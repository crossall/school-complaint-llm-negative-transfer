# Data availability

## Included

- SCB-48 synthetic complaint scenarios and frozen evaluation keys.
- Prespecified 24-scenario expert-evaluation subset.
- Compact model outputs for all reported conditions.
- Anonymous expert scores and binary flags; Experiment 2 adjudicated safety values.
- Retriever v1 chunk IDs, source titles, locators, scores, and content hashes.
- Retriever v1 relevance/error audits and context-uptake measurements.
- Frozen Retriever v2 development/test results and selected-context metadata.
- Analysis code, expected outputs, configuration records, and provenance hashes.

## Excluded

- AI Hub raw complaint-counseling data and processed training JSONL.
- Qwen base weights and QLoRA adapter weights/checkpoints.
- Full text of the 3,716 official-document chunks.
- Direct rater names or contact details.
- Operational school complaints; none were collected for the study.

## Why the package excludes source corpora

The AI Hub datasets and official documents are third-party resources with their own access and reuse conditions. This package does not redistribute them or place them under the repository licenses. Source identifiers, URLs, cryptographic hashes, and experimental protocols are provided so an authorized researcher can verify provenance and reconstruct inputs where the original terms permit.

## Benchmark privacy status

SCB-48 consists of synthetic research scenarios. It contains no real student, parent, teacher, or school complaint record. The model outputs are responses to those synthetic cases.
