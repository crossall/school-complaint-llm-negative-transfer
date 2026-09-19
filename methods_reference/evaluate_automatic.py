#!/usr/bin/env python3
"""Frozen automatic metrics for the v0.6.0 Base + three Chat-SFT outputs."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ["base", "public_chat", "private_chat", "mixed_chat"]
HEADINGS = ["핵심 요약", "1차 처리 방향", "확인할 사항", "처리 계획", "회신 초안"]
DIRECTIONS = ["안내", "추가확인", "담당자연계"]
NUMBER_MARKER = re.compile(r"(?:(?<=^)|(?<=\n)|(?<=\s))(?:\*\*)?([1-5])\s*[.．)]\s*", re.M)


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sequential_markers(text: str):
    matches = list(NUMBER_MARKER.finditer(text))
    result = []
    after = -1
    for expected in range(1, 6):
        found = next((match for match in matches if int(match.group(1)) == expected and match.start() > after), None)
        if found is None:
            return []
        result.append(found)
        after = found.start()
    return result


def repetition_metrics(text: str):
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if len(line) >= 12]
    line_counts = Counter(lines)
    repeated_line_excess = sum(count - 1 for count in line_counts.values() if count > 1)
    words = re.findall(r"\S+", text)
    grams = Counter(tuple(words[index : index + 4]) for index in range(max(0, len(words) - 3)))
    max_fourgram_frequency = max(grams.values(), default=0)
    repeated_fourgram_excess = sum(count - 1 for count in grams.values() if count > 1)
    repetition_flag = repeated_line_excess > 0 or max_fourgram_frequency >= 3
    return repeated_line_excess, max_fourgram_frequency, repeated_fourgram_excess, repetition_flag


def inspect_text(text: str):
    positions = [text.find(heading) for heading in HEADINGS]
    exact_headings = all(position >= 0 for position in positions)
    markers = sequential_markers(text)
    numbered = len(markers) == 5
    section2 = text[markers[1].end() : markers[2].start()] if numbered else ""
    detected = [label for label in DIRECTIONS if label in section2]
    predicted = detected[0] if len(detected) == 1 else ""
    repeated_line_excess, max_fourgram_frequency, repeated_fourgram_excess, repetition_flag = repetition_metrics(text)
    return {
        "exact_five_headings": exact_headings,
        "exact_headings_in_order": exact_headings and positions == sorted(positions),
        "numbered_sections_1_to_5": numbered,
        "direction_parseable": bool(predicted),
        "predicted_processing_level": predicted,
        "repeated_line_excess": repeated_line_excess,
        "max_fourgram_frequency": max_fourgram_frequency,
        "repeated_fourgram_excess": repeated_fourgram_excess,
        "repetition_flag": repetition_flag,
    }


def main():
    output_dir = RUN_ROOT / "outputs"
    log_dir = RUN_ROOT / "logs"
    gold = {row["item_id"]: row for row in read_jsonl(PACKAGE_ROOT / "benchmark" / "pilot_48_gold.jsonl")}
    benchmark = {row["item_id"]: row for row in read_jsonl(PACKAGE_ROOT / "benchmark" / "pilot_48.jsonl")}
    if set(gold) != set(benchmark) or len(gold) != 48:
        raise RuntimeError("Benchmark/gold freeze mismatch")

    details = []
    rows_by_condition = {}
    for condition in CONDITIONS:
        rows = read_jsonl(output_dir / f"{condition}_pilot_chat.jsonl")
        if len(rows) != 48 or {row["item_id"] for row in rows} != set(benchmark):
            raise RuntimeError(f"Incomplete output: {condition}")
        rows_by_condition[condition] = {row["item_id"]: row for row in rows}
        for row in rows:
            measured = inspect_text(row["model_output"])
            expected = gold[row["item_id"]]["processing_level"]
            details.append(
                {
                    "condition": condition,
                    "item_id": row["item_id"],
                    "domain": gold[row["item_id"]]["domain"],
                    "complexity": gold[row["item_id"]]["complexity"],
                    "expected_processing_level": expected,
                    "predicted_processing_level": measured["predicted_processing_level"],
                    "direction_parseable": measured["direction_parseable"],
                    "direction_correct": measured["predicted_processing_level"] == expected,
                    "exact_five_headings": measured["exact_five_headings"],
                    "exact_headings_in_order": measured["exact_headings_in_order"],
                    "numbered_sections_1_to_5": measured["numbered_sections_1_to_5"],
                    "characters": len(row["model_output"]),
                    "input_tokens": row["input_tokens"],
                    "generated_tokens": row["generated_tokens"],
                    "stop_reason": row["stop_reason"],
                    "empty_output": row["empty_output"],
                    "thinking_markup_present": row["thinking_markup_present"],
                    "repeated_line_excess": measured["repeated_line_excess"],
                    "max_fourgram_frequency": measured["max_fourgram_frequency"],
                    "repeated_fourgram_excess": measured["repeated_fourgram_excess"],
                    "repetition_flag": measured["repetition_flag"],
                }
            )

    for item_id in benchmark:
        for key in ("rendered_prompt_sha256", "prompt_token_ids_sha256", "chat_template_sha256"):
            if len({rows_by_condition[condition][item_id][key] for condition in CONDITIONS}) != 1:
                raise RuntimeError(f"Condition input mismatch: {item_id} {key}")

    summary = {}
    for condition in CONDITIONS:
        current = [row for row in details if row["condition"] == condition]
        summary[condition] = {
            "responses": 48,
            "direction_correct": sum(row["direction_correct"] for row in current),
            "direction_by_expected_level": {
                level: sum(row["direction_correct"] and row["expected_processing_level"] == level for row in current)
                for level in DIRECTIONS
            },
            "numbered_sections_1_to_5": sum(row["numbered_sections_1_to_5"] for row in current),
            "exact_five_headings": sum(row["exact_five_headings"] for row in current),
            "mean_generated_tokens": round(sum(row["generated_tokens"] for row in current) / 48, 3),
            "truncated": sum(row["stop_reason"] == "max_new_tokens" for row in current),
            "empty": sum(row["empty_output"] for row in current),
            "repetition_flag": sum(row["repetition_flag"] for row in current),
        }

    detail_path = log_dir / "automatic_metrics_192.csv"
    with detail_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    report = {
        "version": "0.6.0",
        "scope": "mechanical_metrics_only",
        "conditions": summary,
        "repetition_definition": {
            "repeated_line": "공백 정규화 뒤 길이 12자 이상인 동일 행의 초과 반복",
            "fourgram": "공백 토큰 4-gram 빈도와 초과 반복",
            "flag": "반복 행이 있거나 동일 4-gram이 3회 이상 등장",
        },
        "interpretation_limit": "사실성·처리 적절성·요구 충족성·안전성 또는 전체 응답 품질을 판정하지 않는다.",
    }
    (log_dir / "automatic_metrics_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
