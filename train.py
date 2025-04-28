#accelerate launch train.py --model_name_or_path ./model --train_file ./data.jsonl --output_dir sylvie-finetuned --per_device_train_batch_size 4 --gradient_accumulation_steps 8 --num_train_epochs 3 --max_seq_length 256

import argparse
import os
import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Adafactor
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--max_seq_length", type=int, default=256)
    return p.parse_args()


def main():
    # Reduce fragmentation issues
    os.environ["TORCH_ADAMW_MODE"] = "fused"
    gc.collect()
    torch.cuda.empty_cache()

    args = parse_args()

    # 1) Load JSONL dataset
    data = load_dataset("json", data_files={"train": args.train_file})
    train_ds = data["train"]

    # 2) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # Enable gradient checkpointing and disable cache
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    # 3) Tokenize examples
    def tokenize_fn(example):
        text = example["prompt"] + example["completion"]
        return tokenizer(
            text,
            truncation=True,
            max_length=args.max_seq_length,
        )

    tokenized = train_ds.map(
        tokenize_fn,
        batched=False,
        remove_columns=train_ds.column_names,
    )

    # 4) Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 5) Prepare training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        fp16=True,
        remove_unused_columns=False,
        report_to="none"
    )

    # 6) Use Adafactor with manual learning rate (Option 1)
    optimizer = Adafactor(
        model.parameters(),
        lr=args.learning_rate,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False  # must be False when relative_step=False
    )

    # 7) Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None),  # no HF scheduler
    )

    # 8) Train!
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == "__main__":
    main()
