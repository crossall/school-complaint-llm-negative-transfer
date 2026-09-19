from __future__ import annotations

import json

import pandas as pd

from .common import RESULTS, ROOT, write_json


def run() -> dict:
    audit = pd.read_csv(ROOT / "data/retrieval/v1_relevance_audit_192.csv")
    items = pd.read_csv(ROOT / "data/retrieval/v1_error_paths_48.csv")
    uptake = pd.read_csv(ROOT / "data/retrieval/v1_context_uptake_96.csv")
    v2 = json.loads((ROOT / "data/retrieval/v2/03_metrics.json").read_text(encoding="utf-8"))
    counts = audit.relevance_label.value_counts().to_dict()
    by_model = uptake.groupby("model_family").agg(
        context_exclusive_token_rate=("context_exclusive_token_rate", "mean"),
        context_exclusive_3gram_rate=("context_exclusive_3gram_rate", "mean"),
        direction_change_rate=("direction_changed", "mean"),
    ).reset_index()
    result = {
        "v1": {
            "items": int(items.item_id.nunique()),
            "passages": len(audit),
            "relevance_counts": {str(k): int(v) for k, v in counts.items()},
            "direct_evidence_passages": int((audit.relevance_score == 2).sum()),
            "items_without_direct_evidence": int((items.has_direct_chunk == 0).sum()),
            "items_with_misleading_passage": int((items.has_misleading_chunk == 1).sum()),
            "context_uptake_by_model": by_model.to_dict("records"),
        },
        "v2_frozen_output": v2,
    }
    out = RESULTS / "retrieval"
    out.mkdir(parents=True, exist_ok=True)
    by_model.to_csv(out / "context_uptake_by_model.csv", index=False, lineterminator="\n")
    write_json(out / "results.json", result)
    return result


if __name__ == "__main__":
    run()
