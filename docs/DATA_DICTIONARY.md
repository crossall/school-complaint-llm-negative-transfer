# Data dictionary

## Benchmark

`data/benchmark/scb48.jsonl`

- `item_id`: stable identifier, `SCB-P001`–`SCB-P048`.
- `instruction`: frozen system-style task instruction.
- `input`: synthetic complainant statement and scenario facts.

`data/benchmark/scb48_evaluation_key.jsonl`

- `domain`: one of eight complaint domains.
- `processing_level`: target handling level (`안내`, `추가확인`, `담당자연계`).
- `complexity`: simple or complex.
- `gold`: known facts, necessary checks, required actions, critical errors, and acceptable alternatives.

## Expert ratings

`experiment1_96.csv` and `public_multiseed_96.csv` contain one row per response. Important columns:

- `condition`: model/seed condition.
- `handling_correct`: automatic match to the frozen handling label.
- `format_compliant`: all required numbered sections 1–5 were detected.
- `*_rater1`, `*_rater2`: independent 1–5 scores.
- `*_mean`: mean of the two scores.
- `composite_mean`: mean across four rating dimensions after averaging raters.
- `*_both`, `*_either`: conservative and inclusive summaries of independent binary flags. They are not adjudicated multi-seed conclusions.

`rag_independent_192.csv` contains one row per rater × response. `rag_adjudicated_96.csv` contains one row per response and the final consensus safety flags used in Experiment 2.

## Retrieval

- `v1_top4_192.csv`: four hits for each of 48 cases, without passage text.
- `v1_relevance_audit_192.csv`: relevance and misleading-potential judgments; passage text is replaced by SHA-256.
- `v1_error_paths_48.csv`: item-level retrieval failure summary and output linkage coding.
- `v1_context_uptake_96.csv`: Base/Public output differences and lexical context-uptake indicators.
- `v2/02_item_results_48.csv`: frozen Development/Test item outcomes.
- `v2/03_metrics.json`: threshold, split metrics, paired tests, and generation-entry stop decision.

The retrieval audit and output-link coding were exploratory single-analyst judgments, as documented in the manuscript.
