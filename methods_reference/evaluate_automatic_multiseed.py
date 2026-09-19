#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ["base", "public_seed2026", "public_seed2027", "public_seed2028"]
HEADINGS = ["핵심 요약", "1차 처리 방향", "확인할 사항", "처리 계획", "회신 초안"]
DIRECTIONS = ["안내", "추가확인", "담당자연계"]
NUMBER_MARKER = re.compile(r"(?:(?<=^)|(?<=\n)|(?<=\s))(?:\*\*)?([1-5])\s*[.．)]\s*", re.M)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sequential_markers(text: str):
    matches = list(NUMBER_MARKER.finditer(text))
    result, after = [], -1
    for expected in range(1, 6):
        found = next((match for match in matches if int(match.group(1)) == expected and match.start() > after), None)
        if found is None:
            return []
        result.append(found)
        after = found.start()
    return result


def inspect_text(text: str) -> dict:
    positions = [text.find(heading) for heading in HEADINGS]
    markers = sequential_markers(text)
    numbered = len(markers) == 5
    section2 = text[markers[1].end():markers[2].start()] if numbered else ""
    detected = [label for label in DIRECTIONS if label in section2]
    predicted = detected[0] if len(detected) == 1 else ""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if len(line) >= 12]
    repeated_line_excess = sum(count - 1 for count in Counter(lines).values() if count > 1)
    words = re.findall(r"\S+", text)
    grams = Counter(tuple(words[index:index + 4]) for index in range(max(0, len(words) - 3)))
    max_fourgram_frequency = max(grams.values(), default=0)
    return {
        "exact_five_headings": all(position >= 0 for position in positions),
        "numbered_sections_1_to_5": numbered,
        "predicted_processing_level": predicted,
        "repetition_flag": repeated_line_excess > 0 or max_fourgram_frequency >= 3,
    }


def main() -> None:
    gold = {row["item_id"]: row for row in read_jsonl(ROOT / "package/benchmark/pilot_48_gold.jsonl")}
    details = []
    for condition in CONDITIONS:
        rows = read_jsonl(ROOT / f"outputs/{condition}_pilot_chat.jsonl")
        if len(rows) != 48 or {row["item_id"] for row in rows} != set(gold):
            raise RuntimeError(f"Incomplete output: {condition}")
        for row in rows:
            measured = inspect_text(row["model_output"])
            expected = gold[row["item_id"]]["processing_level"]
            details.append({
                "condition": condition,
                "item_id": row["item_id"],
                "domain": gold[row["item_id"]]["domain"],
                "complexity": gold[row["item_id"]]["complexity"],
                "expected_processing_level": expected,
                "predicted_processing_level": measured["predicted_processing_level"],
                "direction_parseable": bool(measured["predicted_processing_level"]),
                "direction_correct": measured["predicted_processing_level"] == expected,
                "exact_five_headings": measured["exact_five_headings"],
                "numbered_sections_1_to_5": measured["numbered_sections_1_to_5"],
                "characters": len(row["model_output"]),
                "input_tokens": row["input_tokens"],
                "generated_tokens": row["generated_tokens"],
                "stop_reason": row["stop_reason"],
                "empty_output": row["empty_output"],
                "repetition_flag": measured["repetition_flag"],
            })
    summary = {}
    for condition in CONDITIONS:
        current = [row for row in details if row["condition"] == condition]
        summary[condition] = {
            "responses": len(current),
            "direction_correct": sum(row["direction_correct"] for row in current),
            "direction_parseable": sum(row["direction_parseable"] for row in current),
            "numbered_sections_1_to_5": sum(row["numbered_sections_1_to_5"] for row in current),
            "exact_five_headings": sum(row["exact_five_headings"] for row in current),
            "mean_generated_tokens": round(sum(row["generated_tokens"] for row in current) / len(current), 3),
            "truncated": sum(row["stop_reason"] == "max_new_tokens" for row in current),
            "empty": sum(row["empty_output"] for row in current),
            "repetition_flag": sum(row["repetition_flag"] for row in current),
        }
    detail_path = ROOT / "logs/automatic_metrics_192.csv"
    with detail_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    report = {
        "version": "0.12.1",
        "scope": "mechanical_metrics_only",
        "conditions": summary,
        "interpretation_limit": "These metrics do not establish semantic response quality or seed replication by themselves."
    }
    (ROOT / "logs/automatic_metrics_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
