#!/usr/bin/env python3
"""Single-GPU QLoRA SFT using Qwen's official chat template.

The semantic role layout matches benchmark inference:
  system = row instruction
  user = row input
  assistant = row output

Only assistant tokens, including the assistant end marker, receive loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def messages(row: dict, user_text: str | None = None, include_answer: bool = False) -> list[dict]:
    result = [
        {"role": "system", "content": row["instruction"]},
        {"role": "user", "content": row.get("input", "") if user_text is None else user_text},
    ]
    if include_answer:
        result.append({"role": "assistant", "content": row["output"]})
    return result


def render_prompt(tokenizer, row: dict, user_text: str | None = None) -> list[int]:
    return tokenizer.apply_chat_template(
        messages(row, user_text=user_text),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def render_response(tokenizer, row: dict, prompt_ids: list[int]) -> list[int]:
    full_ids = tokenizer.apply_chat_template(
        messages(row, include_answer=True),
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("Official chat template full conversation does not begin with the generation prompt")
    return full_ids[len(prompt_ids) :]


def assistant_end_ids(tokenizer, row: dict, prompt_ids: list[int]) -> list[int]:
    empty = dict(row)
    empty["output"] = ""
    full_ids = tokenizer.apply_chat_template(
        messages(empty, include_answer=True),
        tokenize=True,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError("Cannot identify assistant end marker")
    suffix = full_ids[len(prompt_ids) :]
    if not suffix:
        raise RuntimeError("Official chat template has no assistant end marker")
    return suffix


def crop_input(tokenizer, row: dict, max_prompt_tokens: int) -> tuple[list[int], bool]:
    prompt_ids = render_prompt(tokenizer, row)
    if len(prompt_ids) <= max_prompt_tokens:
        return prompt_ids, False

    raw_ids = tokenizer(row.get("input", ""), add_special_tokens=False).input_ids

    def candidate(keep: int) -> list[int]:
        if keep <= 0:
            selected = []
        else:
            head = max(1, keep // 4)
            tail = keep - head
            selected = raw_ids[:head] + (raw_ids[-tail:] if tail else [])
        user_text = tokenizer.decode(selected, skip_special_tokens=True)
        return render_prompt(tokenizer, row, user_text=user_text)

    low, high = 0, len(raw_ids)
    best = candidate(0)
    if len(best) > max_prompt_tokens:
        raise RuntimeError("Instruction and chat-template overhead exceed the available prompt budget")
    while low <= high:
        mid = (low + high) // 2
        trial = candidate(mid)
        if len(trial) <= max_prompt_tokens:
            best = trial
            low = mid + 1
        else:
            high = mid - 1
    return best, True


def encode_example(tokenizer, row: dict, max_length: int, max_response_tokens: int):
    original_prompt = render_prompt(tokenizer, row)
    response_ids = render_response(tokenizer, row, original_prompt)
    end_ids = assistant_end_ids(tokenizer, row, original_prompt)
    if response_ids[-len(end_ids) :] != end_ids:
        raise RuntimeError("Assistant response does not end with the official end marker")
    if len(response_ids) > max_response_tokens:
        content_ids = response_ids[: -len(end_ids)]
        keep = max_response_tokens - len(end_ids)
        if keep < 1:
            raise RuntimeError("max_response_tokens is too small for the official assistant end marker")
        response_ids = content_ids[:keep] + end_ids

    prompt_ids, input_trimmed = crop_input(tokenizer, row, max_length - len(response_ids))
    ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids
    if len(ids) > max_length or not any(label != -100 for label in labels):
        raise RuntimeError("Invalid encoded example")
    return ids, labels, input_trimmed


class PackedSFTDataset(IterableDataset):
    def __init__(self, rows, tokenizer, max_length, seed, max_response_tokens=512):
        self.rows = rows
        self.tok = tokenizer
        self.max_length = max_length
        self.seed = seed
        self.max_response_tokens = max_response_tokens

    def __iter__(self):
        epoch = 0
        while True:
            order = list(range(len(self.rows)))
            random.Random(self.seed + epoch).shuffle(order)
            buffer_ids, buffer_labels = [], []
            for index in order:
                ids, labels, _ = encode_example(
                    self.tok,
                    self.rows[index],
                    self.max_length,
                    self.max_response_tokens,
                )
                if buffer_ids and len(buffer_ids) + len(ids) > self.max_length:
                    pad = self.max_length - len(buffer_ids)
                    yield {
                        "input_ids": torch.tensor(buffer_ids + [self.tok.pad_token_id] * pad, dtype=torch.long),
                        "attention_mask": torch.tensor([1] * len(buffer_ids) + [0] * pad, dtype=torch.long),
                        "labels": torch.tensor(buffer_labels + [-100] * pad, dtype=torch.long),
                    }
                    buffer_ids, buffer_labels = [], []
                buffer_ids.extend(ids)
                buffer_labels.extend(labels)
                if len(buffer_ids) == self.max_length:
                    yield {
                        "input_ids": torch.tensor(buffer_ids, dtype=torch.long),
                        "attention_mask": torch.ones(self.max_length, dtype=torch.long),
                        "labels": torch.tensor(buffer_labels, dtype=torch.long),
                    }
                    buffer_ids, buffer_labels = [], []
            if buffer_ids:
                pad = self.max_length - len(buffer_ids)
                yield {
                    "input_ids": torch.tensor(buffer_ids + [self.tok.pad_token_id] * pad, dtype=torch.long),
                    "attention_mask": torch.tensor([1] * len(buffer_ids) + [0] * pad, dtype=torch.long),
                    "labels": torch.tensor(buffer_labels + [-100] * pad, dtype=torch.long),
                }
            epoch += 1


def collate(features):
    return {key: torch.stack([item[key] for item in features]) for key in features[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--condition", required=True, choices=["public", "private", "mixed"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-response-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    args = parser.parse_args()

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("This protocol requires exactly one GPU process")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")

    train_path = Path(args.train_file)
    rows = load_jsonl(train_path)
    if len(rows) != 5600 or any(row.get("domain") != args.condition and args.condition != "mixed" for row in rows):
        raise RuntimeError("Training file does not match the frozen condition")
    if args.condition == "mixed":
        counts = {name: sum(row.get("domain") == name for row in rows) for name in ("public", "private")}
        if counts != {"public": 2800, "private": 2800}:
            raise RuntimeError(f"Mixed balance mismatch: {counts}")

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    if not tokenizer.chat_template:
        raise RuntimeError("Pinned Qwen tokenizer has no official chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    sample_ids, sample_labels, sample_trimmed = encode_example(
        tokenizer, rows[0], args.max_length, args.max_response_tokens
    )
    if sample_trimmed or sample_labels[: len(sample_ids) - sum(x != -100 for x in sample_labels)] != [-100] * (len(sample_ids) - sum(x != -100 for x in sample_labels)):
        raise RuntimeError("Chat-template supervision preflight failed")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    run_config = vars(args) | {
        "version": "0.6.0",
        "serialization": "tokenizer.apply_chat_template",
        "role_layout": {"system": "instruction", "user": "input", "assistant": "output"},
        "enable_thinking": False,
        "train_rows": len(rows),
        "train_file_sha256": sha256(train_path),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest(),
        "sample_total_tokens": len(sample_ids),
        "sample_supervised_tokens": sum(label != -100 for label in sample_labels),
    }
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=False,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    model.print_trainable_parameters()

    dataset = PackedSFTDataset(rows, tokenizer, args.max_length, args.seed, args.max_response_tokens)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=True,
        fp16=False,
        optim="paged_adamw_8bit",
        weight_decay=0.0,
        max_grad_norm=1.0,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=collate)
    trainer.train()
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"CHAT_SFT_COMPLETE {args.condition} {final_dir}", flush=True)


if __name__ == "__main__":
    main()
