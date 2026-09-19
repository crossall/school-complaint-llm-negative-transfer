# Target-Task Misalignment in LLM Domain Adaptation

Reproducibility package for **“Target-Task Misalignment in LLM Domain Adaptation: Multi-Seed Negative Transfer and Retrieval Failures in School Complaint Response.”**

This release candidate contains the synthetic SCB-48 benchmark, frozen evaluation keys, anonymized expert ratings, compact model outputs, retrieval metadata, and analysis code used for the manuscript. It reproduces the instruction-tuning, multi-seed, RAG, safety, and retrieval results without downloading model weights or running a GPU.

> Release status: `v1.0.0-rc1` (prepared 2026-09-19 KST). Repository: [github.com/crossall/school-complaint-llm-negative-transfer](https://github.com/crossall/school-complaint-llm-negative-transfer). DOI will be inserted after the Zenodo release.

## Quick start

Python 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python run_all.py
python -m unittest discover -s tests -v
```

All regenerated tables and JSON summaries are written under `results/`. The full analysis runs on CPU and does not access the network.

## Newly integrated format-compliance sensitivity analysis

`analysis/analyze_multiseed.py` now treats the format-compliant-only analysis as part of the main reproducibility pipeline. For each Public SFT seed, it retains scenarios where that Public response contained the required five-section structure, pairs those items with the same Base responses, runs a two-sided Wilcoxon signed-rank test with SciPy `method="auto"`, and applies Holm correction across the three seed comparisons.

Expected results:

| Public SFT seed | n | Mean Public−Base | Holm-adjusted p |
|---|---:|---:|---:|
| 2026 | 20 | -0.69375 | 0.0012733 |
| 2027 | 14 | -0.74107 | 0.0046241 |
| 2028 | 10 | -0.75000 | 0.0046241 |

The exact output is `results/public_multiseed/format_compliant_sensitivity.csv`. A regression test prevents this analysis from silently changing.

## Repository map

- `data/benchmark/`: SCB-48 cases, frozen evaluation key, and prespecified 24-item expert subset.
- `data/ratings/`: anonymized independent ratings, response-level means, and adjudicated Experiment 2 safety flags.
- `data/model_outputs/`: compact outputs and reproducibility hashes for Experiment 1, Public multi-seed runs, and RAG conditions.
- `data/retrieval/`: chunk IDs, source metadata, retrieval scores, passage relevance audit, error-path analysis, and frozen Retriever v2 outputs.
- `analysis/`: end-to-end statistical analyses.
- `methods_reference/`: training, generation, and automatic-evaluation scripts retained for method transparency.
- `config/`: frozen experimental protocols.
- `docs/`: data dictionary, provenance, ethics/privacy notes, and DOI release instructions.
- `tests/`: numerical and package-integrity regression tests.

## Reproducibility scope

The included analysis reproduces manuscript tables and inferential statistics from frozen derived data. Training and generation scripts are supplied as method references, but a full model rerun additionally requires the Qwen3-8B checkpoint, compatible GPU software, AI Hub access, and locally prepared training data.

AI Hub raw data and processed training JSONL, model weights/adapters, the full official-document corpus, and direct rater identifiers are intentionally excluded. Retrieval records retain chunk identifiers, locations, scores, and SHA-256 hashes. See `docs/DATA_AVAILABILITY.md` and `docs/THIRD_PARTY_DATA.md`.

## Licenses

Code is licensed under Apache License 2.0. Project-authored benchmark, ratings, metadata, and derived tables are licensed under CC BY 4.0, to the extent the author holds rights in them. Third-party model, dataset, and official-document materials retain their original terms and are not relicensed here. See `LICENSE`, `LICENSE-DATA.md`, and `NOTICE`.

## Citation

Until the DOI is minted, cite the manuscript and this repository as:

> Kim, Taeryeong. (2026). *School Complaint LLM Negative Transfer and Retrieval Failures: Reproducibility Package* (Version 1.0.0). DOI to be assigned.

Machine-readable citation metadata are provided in `CITATION.cff` and `.zenodo.json`.

## Responsible use

SCB-48 is a synthetic research benchmark. The outputs are not operational advice and must not be used to automate school complaint decisions. High-risk cases require review and authorization by qualified personnel.
