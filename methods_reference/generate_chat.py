#!/usr/bin/env python3
"""Generate one v0.6.0 condition with the frozen official Qwen chat protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig, set_seed


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument(
        "--condition",
        required=True,
        choices=["base", "public_chat", "private_chat", "mixed_chat"],
    )
    args = parser.parse_args()

    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    generation = protocol["generation"]
    prompt_config = protocol["inference_format"]
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if protocol["version"] != "0.6.0" or protocol["model"]["revision"] not in args.model:
        raise RuntimeError("Model or protocol version mismatch")
    if (args.condition == "base") != (args.adapter is None):
        raise RuntimeError("Only Base may run without an adapter")

    benchmark_path = Path(args.benchmark)
    if hashlib.sha256(benchmark_path.read_bytes()).hexdigest() != protocol["freeze_hashes"]["pilot_48.jsonl"]:
        raise RuntimeError("Frozen benchmark hash mismatch")
    rows = [json.loads(line) for line in benchmark_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 48 or len({row["item_id"] for row in rows}) != 48:
        raise RuntimeError("Frozen benchmark must contain 48 unique items")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise RuntimeError("Never overwrite generated outputs")

    set_seed(generation["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=True, local_files_only=True, trust_remote_code=False
    )
    if not tokenizer.chat_template:
        raise RuntimeError("Pinned tokenizer must provide an official chat template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer EOS is required")
    chat_template_sha = hashlib.sha256(tokenizer.chat_template.encode("utf-8")).hexdigest()
    if chat_template_sha != protocol["model"]["chat_template_sha256"]:
        raise RuntimeError("Chat template hash mismatch")

    prepared = []
    for row in rows:
        if set(row) != {"item_id", "instruction", "input"}:
            raise RuntimeError(f"Unexpected benchmark schema: {row.get('item_id')}")
        messages = [
            {"role": "system", "content": row["instruction"]},
            {"role": "user", "content": row["input"]},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        if not isinstance(ids, list) or not ids or len(ids) > prompt_config["max_input_tokens"]:
            raise RuntimeError(f"Invalid prompt tokens: {row['item_id']}")
        if tokenizer(rendered, add_special_tokens=False)["input_ids"] != ids:
            raise RuntimeError(f"Rendered/tokenized prompt mismatch: {row['item_id']}")
        prepared.append((row, rendered, ids))

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA GPU required")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quantization,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    config = GenerationConfig(
        max_new_tokens=generation["max_new_tokens"],
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.0,
        use_cache=True,
    )
    effective, _ = model._prepare_generation_config(config, use_model_defaults=False, do_sample=False)
    if effective.do_sample is not False or effective.num_beams != 1 or effective.max_new_tokens != generation["max_new_tokens"]:
        raise RuntimeError("Effective generation settings differ from the frozen protocol")

    started = time.time()
    with output_path.open("x", encoding="utf-8") as stream:
        for index, (row, rendered, input_ids) in enumerate(prepared, 1):
            encoded = torch.tensor([input_ids], dtype=torch.long, device=model.device)
            attention = torch.ones_like(encoded)
            before = time.time()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=encoded,
                    attention_mask=attention,
                    generation_config=config,
                    use_model_defaults=False,
                    do_sample=False,
                )
            generated_ids = generated[0, len(input_ids) :].tolist()
            raw = tokenizer.decode(generated_ids, skip_special_tokens=False)
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            eos_ids = effective.eos_token_id if isinstance(effective.eos_token_id, list) else [effective.eos_token_id]
            if generated_ids and generated_ids[-1] in eos_ids:
                stop_reason = "eos"
            elif len(generated_ids) == generation["max_new_tokens"]:
                stop_reason = "max_new_tokens"
            else:
                stop_reason = "generation_ended_without_declared_stop"
            record = dict(
                row,
                model_output=text,
                model_output_raw_with_special_tokens=raw,
                generated_token_ids=generated_ids,
                adapter=args.adapter or "BASE",
                base_model=protocol["model"]["base"],
                model_snapshot=args.model,
                model_revision=protocol["model"]["revision"],
                condition=args.condition,
                protocol_sha256=protocol_sha,
                chat_template_sha256=chat_template_sha,
                rendered_prompt=rendered,
                rendered_prompt_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                prompt_token_ids_sha256=token_hash(input_ids),
                input_tokens=len(input_ids),
                generated_tokens=len(generated_ids),
                eos_token_ids=eos_ids,
                stop_reason=stop_reason,
                suspected_truncation=stop_reason == "max_new_tokens",
                empty_output=not text.strip(),
                thinking_markup_present="<think>" in text or "</think>" in text,
                effective_do_sample=effective.do_sample,
                effective_num_beams=effective.num_beams,
                effective_temperature=None,
                use_model_defaults=False,
                enable_thinking=False,
                generation_seconds=round(time.time() - before, 3),
                generated_at_unix=time.time(),
            )
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(
                json.dumps(
                    {
                        "condition": args.condition,
                        "completed": index,
                        "total": len(rows),
                        "item_id": row["item_id"],
                        "tokens": len(generated_ids),
                        "stop": stop_reason,
                        "elapsed_seconds": round(time.time() - started),
                    }
                ),
                flush=True,
            )
    print(f"GENERATION_COMPLETE {args.condition}", flush=True)


if __name__ == "__main__":
    main()
