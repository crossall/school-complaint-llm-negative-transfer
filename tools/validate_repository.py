from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".csv", ".toml", ".cff", ".txt", ".yml", ".yaml"}
IGNORED_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache"}


def is_ignored(path: Path) -> bool:
    return bool(IGNORED_DIRS.intersection(path.relative_to(ROOT).parts))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    failures: list[str] = []
    required = [
        "README.md", "CITATION.cff", ".zenodo.json", "LICENSE", "LICENSE-DATA.md",
        "MANIFEST.sha256", "data/benchmark/scb48.jsonl",
        "results/public_multiseed/format_compliant_sensitivity.csv",
    ]
    for relative in required:
        if not (ROOT / relative).is_file():
            failures.append(f"missing required file: {relative}")

    forbidden_suffixes = {".safetensors", ".pt", ".pth", ".bin", ".ckpt"}
    for path in ROOT.rglob("*"):
        if is_ignored(path):
            continue
        if path.is_file() and path.suffix.lower() in forbidden_suffixes:
            failures.append(f"model weight file present: {path.relative_to(ROOT)}")
        lowered = path.name.lower()
        if "corpus_chunks_3716" in lowered or "corpus_chunks_multisource_frozen" in lowered:
            failures.append(f"full corpus file present: {path.relative_to(ROOT)}")

    patterns = {
        "local Windows user path": re.compile(r"[A-Za-z]:\\Users\\"),
        "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
        "Hugging Face token": re.compile(r"\bhf_[A-Za-z0-9]{16,}"),
    }
    for path in ROOT.rglob("*"):
        if not path.is_file() or is_ignored(path) or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in patterns.items():
            if pattern.search(text):
                failures.append(f"{label} in {path.relative_to(ROOT)}")

    expected_rows = {
        "data/ratings/experiment1_96.csv": 96,
        "data/ratings/public_multiseed_96.csv": 96,
        "data/ratings/rag_independent_192.csv": 192,
        "data/ratings/rag_adjudicated_96.csv": 96,
        "data/retrieval/v1_top4_192.csv": 192,
        "data/retrieval/v1_relevance_audit_192.csv": 192,
    }
    for relative, expected in expected_rows.items():
        actual = len(pd.read_csv(ROOT / relative))
        if actual != expected:
            failures.append(f"row count {relative}: expected {expected}, got {actual}")

    audit_columns = set(pd.read_csv(ROOT / "data/retrieval/v1_relevance_audit_192.csv", nrows=1).columns)
    if "chunk_text" in audit_columns or "chunk_text_sha256" not in audit_columns:
        failures.append("v1 relevance audit text redaction is invalid")
    for relative, expected in [
        ("data/retrieval/v2/04_v2_selected_contexts_metadata.jsonl", 37),
        ("data/retrieval/v2/06_v2_reranked_top20_metadata.jsonl", 960),
    ]:
        rows = [json.loads(line) for line in (ROOT / relative).read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) != expected:
            failures.append(f"row count {relative}: expected {expected}, got {len(rows)}")
        if any("text" in row or "chunk_text_sha256" not in row for row in rows):
            failures.append(f"v2 passage text redaction is invalid: {relative}")

    for folder in ["experiment1", "public_multiseed", "rag"]:
        for path in (ROOT / "data/model_outputs" / folder).glob("*.jsonl"):
            count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            if count != 48:
                failures.append(f"model output count {path.relative_to(ROOT)}: expected 48, got {count}")

    manifest = ROOT / "MANIFEST.sha256"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected_hash, relative = line.split("  ", 1)
            path = ROOT / relative
            if not path.is_file():
                failures.append(f"manifest file missing: {relative}")
            elif digest(path) != expected_hash:
                failures.append(f"manifest hash mismatch: {relative}")

    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    if zenodo.get("creators", [{}])[0].get("orcid") != "0009-0006-5417-5094":
        failures.append("Zenodo ORCID mismatch")

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        sys.exit(1)
    print("PASS: repository structure, public-scope scan, row counts, metadata, and hashes are valid.")


if __name__ == "__main__":
    main()
