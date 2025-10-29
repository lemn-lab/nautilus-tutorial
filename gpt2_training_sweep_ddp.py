#!/usr/bin/env python3

import os
import math
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Union, Dict

import numpy as np
import torch
from datasets import load_dataset, DatasetDict
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    PreTrainedTokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
import wandb
import torch.distributed as dist

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_tokenizer(shakespeare_files, tokenizer_dir, vocab_size=52000, min_frequency=2, special_tokens=None):
    """Train a ByteLevel BPE tokenizer and save it to tokenizer_dir."""
    tokenizer_dir = Path(tokenizer_dir)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    if special_tokens is None:
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

    logger.info("Training tokenizer...")
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=shakespeare_files, vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=special_tokens)
    tokenizer.save_model(str(tokenizer_dir))

    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    # Wrap with PreTrainedTokenizerFast so it's compatible with transformers API
    tokenizer_fast = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json_path),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        vocab_file=str(tokenizer_dir / "vocab.json"),
        merges_file=str(tokenizer_dir / "merges.txt"),
    )

    # ensure pad token id is set
    tokenizer_fast.pad_token = "<pad>"
    tokenizer_fast.save_pretrained(str(tokenizer_dir))
    logger.info(f"Tokenizer saved to {tokenizer_dir}")
    return str(tokenizer_dir)

def setup_ddp():
    """Initialize DDP using environment variables set by torchrun."""
    # torchrun sets these environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # CRITICAL: Set device BEFORE init_process_group
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    
    # Initialize process group AFTER setting device
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    
    return rank, local_rank, world_size

def cleanup_ddp():
    try:
        dist.barrier()
    except Exception:
        pass
    dist.destroy_process_group()


def load_and_tokenize_dataset(text_files, tokenizer_path, block_size=128, test_size=0.1, seed=42, dataset_cache_dir=None):
    """Load raw text dataset and prepare tokenized dataset grouped into block_size chunks."""
    logger.info("Loading dataset (text)...")
    # load_dataset('text') expects newline-separated text or single large file(s).
    data_files = {"train": text_files} if isinstance(text_files, (list, tuple)) else {"train": [text_files]}
    ds = load_dataset("text", data_files=data_files, cache_dir=dataset_cache_dir)
    # Remove empty lines
    def filter_empty(ex):
        return ex["text"] is not None and ex["text"].strip() != ""

    ds = ds.filter(filter_empty)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path, pad_token="<pad>", unk_token="<unk>")
    logger.info("Tokenizing dataset (this may take a while)...")

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized = ds.map(tokenize_function, batched=True, remove_columns=["text"]) # -> batch size is given here implicitly, its 1000 by default 
    # Concatenate and chunk into blocks of block_size
    def group_texts(examples):
        # concatenates all input_ids for a batch and split into chunks
        all_ids = sum(examples["input_ids"], [])
        total_length = len(all_ids)
        # drop the remainder to make chunks exact
        total_length = (total_length // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "attention_mask": []}
        input_ids = [all_ids[i : i + block_size] for i in range(0, total_length, block_size)]
        attention_masks = [[1] * block_size for _ in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_masks}

    lm_datasets = tokenized.map(group_texts, batched=True, remove_columns=tokenized["train"].column_names)
    split = tokenized["train"].train_test_split(test_size=test_size, seed=seed)
    lm_datasets = DatasetDict({"train": split["train"], "test": split["test"]})
    logger.info(f"Tokenized and grouped dataset size: {len(lm_datasets)} blocks")
    return lm_datasets, tokenizer


def make_small_gpt2_config(vocab_size, n_embd=256, n_layer=6, n_head=8, max_position_embeddings=1024):
    """Return a GPT2Config tuned for a small model (random init)."""
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_position_embeddings,
        n_ctx=max_position_embeddings,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
    )
    return config


# ---------------- Training Function for Sweep ----------------
def train_with_sweep_ddp(rank, local_rank, world_size, config=None):
    """DDP-aware GPT2 training"""

    is_main = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    learning_rate_tensor = torch.zeros(1).to(device)
    batch_size_tensor = torch.zeros(1).to(device)
    num_train_epochs_tensor = torch.zeros(1).to(device)
    weight_decay_tensor = torch.zeros(1).to(device)
    gradient_accumulation_steps_tensor = torch.zeros(1).to(device)

    if is_main:
        wandb.init(config=config)
        config = wandb.config
        learning_rate = config.get('learning_rate', config.learning_rate)
        learning_rate_tensor[0] = learning_rate
        batch_size = config.get('per_device_train_batch_size', config.per_device_train_batch_size)
        batch_size_tensor[0] = batch_size
        num_train_epochs = config.get('num_train_epochs', config.num_train_epochs)
        num_train_epochs_tensor[0] = num_train_epochs
        weight_decay = config.get('weight_decay', config.weight_decay)
        weight_decay_tensor[0] = weight_decay
        gradient_accumulation_steps = config.get('gradient_accumulation_steps', config.gradient_accumulation_steps)
        gradient_accumulation_steps_tensor[0] = gradient_accumulation_steps
        print(f"[rank {rank}] WandB config:", config)
    
    dist.broadcast(learning_rate_tensor, src=0)
    dist.broadcast(num_train_epochs_tensor, src=0)
    dist.broadcast(batch_size_tensor, src=0)
    dist.broadcast(gradient_accumulation_steps_tensor, src=0)
    dist.broadcast(weight_decay_tensor, src=0)

    learning_rate = learning_rate_tensor.item()
    num_train_epochs = int(num_train_epochs_tensor.item())
    batch_size = int(batch_size_tensor.item())
    gradient_accumulation_steps = int(gradient_accumulation_steps_tensor.item())
    weight_decay = weight_decay_tensor.item()


    # Dataset & tokenizer
    lm_datasets, tokenizer = load_and_tokenize_dataset(args.shakespeare_files, args.tokenizer_dir, block_size=args.block_size)

    # Model
    model_config = make_small_gpt2_config(vocab_size=len(tokenizer.get_vocab()))
    model = GPT2LMHeadModel(model_config)
    model.resize_token_embeddings(len(tokenizer))

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    report_to = "wandb" if is_main else "none"

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=500,
        weight_decay=weight_decay,
        learning_rate=learning_rate,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=200,
        save_strategy="epoch",
        save_total_limit=3,
        evaluation_strategy="epoch",
        fp16=args.fp16 and torch.cuda.is_available(),
        dataloader_pin_memory=True,
        report_to=report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
    )

    logger.info(f"Starting training: {sum(p.numel() for p in model.parameters()):,} parameters")
    trainer.train()

    # any custom logging should ONLY be done by the main process
    # if is_main:
    #     wandb.log()
    dist.barrier()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    if is_main:
        wandb.finish()

# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train small GPT-2 with W&B sweep")
    parser.add_argument("--shakespeare_files", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, default="./gpt2_shakespeare_small")
    parser.add_argument("--tokenizer_dir", type=str, default="./tokenizer")
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tokenizer_dir, exist_ok=True)

    # Train tokenizer if missing
    if not (Path(args.tokenizer_dir) / "vocab.json").exists():
        train_tokenizer(args.shakespeare_files, args.tokenizer_dir)
    
    wandb.login(key="") # -> fill your wandb key

    # ---------------- W&B Sweep ----------------
    sweep_config = {
        "name": "gpt2-shakespeare",
        "method": "bayes",
        "metric": {"name": "eval/loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 1e-5, "max": 5e-4},
            "per_device_train_batch_size": {"values": [4, 8, 16]},
            "gradient_accumulation_steps": {"values": [1, 2, 4]},
            "num_train_epochs": {"values": [1, 2, 3]},
            "weight_decay": {"min": 0.0, "max": 0.1},
        }
    }

    rank, local_rank, world_size = setup_ddp()

    sweep_id = wandb.sweep(sweep_config, project="gpt2_shakespeare")
    wandb.agent(sweep_id, lambda: train_with_sweep_ddp(rank, local_rank, world_size), count=10)

    cleanup_ddp()