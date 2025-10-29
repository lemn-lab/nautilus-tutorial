#!/usr/bin/env python3
"""
train_gpt2_from_scratch.py

Trains a small GPT-2 style model from scratch using the Shakespeare corpus.
Steps:
  1. Train a byte-level BPE tokenizer on the text file(s)
  2. Load dataset with `datasets`
  3. Tokenize and group text into blocks
  4. Create a small GPT-2 config and model (random init)
  5. Train with Hugging Face Trainer
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

    tokenized = ds.map(tokenize_function, batched=True, remove_columns=["text"]) 
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

    lm_datasets = tokenized.map(group_texts, batched=True, remove_columns=tokenized["train"].column_names) # -> # batch size is given here implicitly. How many lines are taken at once is the batch size here. its 1000
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


def main():
    parser = argparse.ArgumentParser(description="Train small GPT-2 from scratch on Shakespeare")
    parser.add_argument("--shakespeare_files", nargs="+", required=True, help="Path(s) to shakespeare text file(s).")
    parser.add_argument("--output_dir", type=str, default="./gpt2_shakespeare_small", help="Where to save model/tokenizer/checkpoints")
    parser.add_argument("--tokenizer_dir", type=str, default="./tokenizer", help="Where to write trained tokenizer")
    parser.add_argument("--vocab_size", type=int, default=52000, help="Tokenizer vocab size")
    parser.add_argument("--block_size", type=int, default=128, help="Block size for LM training")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Use fp16 mixed precision")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tokenizer_dir, exist_ok=True)

    # 1) Train tokenizer (only if tokenizer_dir is empty)
    tokenizer_path = args.tokenizer_dir
    # If tokenizer files exist, skip training
    if not (Path(tokenizer_path) / "vocab.json").exists():
        tokenizer_path = train_tokenizer(args.shakespeare_files, tokenizer_path, vocab_size=args.vocab_size)
    else:
        logger.info("Tokenizer already exists, loading.")

    # 2) Load & tokenize dataset
    lm_datasets, tokenizer = load_and_tokenize_dataset(args.shakespeare_files, tokenizer_path, block_size=args.block_size)

    # 3) Build model config and initialize model from scratch
    config = make_small_gpt2_config(vocab_size=len(tokenizer.get_vocab()), n_embd=256, n_layer=6, n_head=8, max_position_embeddings=1024)
    logger.info("Config: %s", config.to_json_string())

    model = GPT2LMHeadModel(config)
    # ensure model token embeddings match tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # 4) Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 5) Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=500,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.95,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=200,
        save_strategy="epoch",
        save_total_limit=3,
        evaluation_strategy="epoch",
        fp16=args.fp16 and torch.cuda.is_available(),
        dataloader_pin_memory=True,
        report_to="none",
    )

    # 6) Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
    )

    # Print basic info
    logger.info(f"Starting training: {sum(p.numel() for p in model.parameters()):,} parameters")
    # 7) Train
    trainer.train()
    # 8) Save final model and tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Training complete. Model + tokenizer saved to {args.output_dir}")


if __name__ == "__main__":
    main()