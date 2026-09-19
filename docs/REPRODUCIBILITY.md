# Reproducibility guide

## What the default run reproduces

`python run_all.py` executes four linked analyses:

1. Experiment 1: Base, Public, Private, and Mixed chat-template SFT on the 24-item expert subset.
2. Public multi-seed robustness: Base versus Public seeds 2026, 2027, and 2028, including the format-compliant-only sensitivity analysis.
3. Experiment 2: Base/Public × RAG Off/On expert ratings, prespecified contrasts, cluster-robust sensitivity estimates, reliability, and adjudicated safety flags.
4. Retrieval analysis: Retriever v1 passage audit and frozen Retriever v2 results.

The run reads only files under `data/` and writes only to `results/`. It does not invoke a model, use a GPU, or access the network.

## Software environment

The numerical reference environment is Python 3.11 with NumPy 2.3.3, pandas 2.3.3, and SciPy 1.16.2. These versions are pinned in `requirements.txt` and `pyproject.toml`.

SciPy is pinned because `scipy.stats.wilcoxon(method="auto")` selects an exact or asymptotic calculation based on sample structure. This choice matters for the seed-2028 format-compliant subset (`n=10`). The repository regression test checks the exact values reported in the manuscript.

## Independent verification

```bash
python run_all.py
python -m unittest discover -s tests -v
python tools/validate_repository.py
```

Expected high-level checks include:

- Experiment 1 overall means: Base 4.266, Public 3.490, Private 3.891, Mixed 3.958.
- Public three-seed overall means: 3.490, 3.318, 3.344.
- Format-compliant Public−Base differences: -0.69375, -0.741071, -0.75000.
- RAG overall means: Base Off 4.099, Base On 3.938, Public Off 3.839, Public On 3.120.
- Public any-safety-error count: 2/24 Off and 13/24 On.
- Retriever v1 direct evidence: 9/192 passages; misleading context in 42/48 scenarios.
- Retriever v2 Test exact Gold Hit@4: 0/9; no-context accuracy: 9/15.

## Full training and generation rerun

The `methods_reference/` scripts and `config/` protocols document the training and generation implementation. A full rerun requires:

- authorized access to the two AI Hub complaint-counseling datasets;
- the preprocessing pipeline and locally generated balanced 5,600-row condition files;
- Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`;
- GPU-compatible PyTorch, Transformers, PEFT, Accelerate, and bitsandbytes versions listed in the protocol files;
- official-document source files when reconstructing the 3,716-chunk retrieval corpus.

Those third-party inputs are excluded from this public package. The analysis rerun remains complete because frozen outputs, ratings, retrieval metadata, and derived inputs are included.
