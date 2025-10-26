#!/usr/bin/env python3
"""
train_gpt2_from_scratch_sweep.py

Trains a small GPT-2 style model from scratch using the Shakespeare corpus.
Steps:
  1. Train a byte-level BPE tokenizer on the text file(s)
  2. Load dataset with `datasets`
  3. Tokenize and group text into blocks
  4. Create a small GPT-2 config and model (random init)
  5. Train with Hugging Face Trainer
  6. Along with step 4, 5; run sweep using wandb
"""

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

    tokenized = ds.map(tokenize_function, batched=True, remove_columns=["text"]) # batch size is given here implicitly 
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
def train_with_sweep(config=None):
    wandb.init(config=config)
    config = wandb.config

    # Dataset & tokenizer
    lm_datasets, tokenizer = load_and_tokenize_dataset(args.shakespeare_files, args.tokenizer_dir, block_size=args.block_size)

    # Model
    model_config = make_small_gpt2_config(vocab_size=len(tokenizer.get_vocab()))
    model = GPT2LMHeadModel(model_config)
    model.resize_token_embeddings(len(tokenizer))

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=500,
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=200,
        save_strategy="epoch",
        save_total_limit=3,
        evaluation_strategy="epoch",
        fp16=args.fp16 and torch.cuda.is_available(),
        dataloader_pin_memory=True,
        report_to="wandb",
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
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

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
        "method": "bayes",
        "metric": {"name": "eval_loss", "goal": "minimize"},
        "parameters": {
            "learning_rate": {"min": 1e-5, "max": 5e-4},
            "per_device_train_batch_size": {"values": [4, 8, 16]},
            "gradient_accumulation_steps": {"values": [1, 2, 4]},
            "num_train_epochs": {"values": [1, 2, 3]},
            "weight_decay": {"min": 0.0, "max": 0.1},
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="gpt2_shakespeare")
    wandb.agent(sweep_id, function=train_with_sweep, count=2)