from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from .common import RESULTS, ROOT, exact_mcnemar, holm_adjust, icc_absolute, paired_test, write_json


CONDITIONS = ["base_rag_off", "base_rag_on", "public_chat_rag_off", "public_chat_rag_on"]
METRICS = [
    "factual_accuracy",
    "processing_appropriateness",
    "requirement_coverage",
    "safety_administrative_appropriateness",
    "composite",
]
CONTRASTS = {
    "public_on_minus_off": {"public_chat_rag_on": 1, "public_chat_rag_off": -1},
    "base_on_minus_off": {"base_rag_on": 1, "base_rag_off": -1},
    "difference_in_differences": {
        "public_chat_rag_on": 1,
        "public_chat_rag_off": -1,
        "base_rag_on": -1,
        "base_rag_off": 1,
    },
    "base_on_minus_public_on": {"base_rag_on": 1, "public_chat_rag_on": -1},
}


def cluster_robust(raw: pd.DataFrame, metric: str) -> list[dict]:
    model_public = (raw.model_family == "public_chat").astype(float).to_numpy()
    rag_on = (raw.rag == "on").astype(float).to_numpy()
    rater2 = (raw.rater == 2).astype(float).to_numpy()
    x = np.column_stack([np.ones(len(raw)), model_public, rag_on, model_public * rag_on, rater2])
    y = raw[metric].to_numpy(float)
    inv = np.linalg.inv(x.T @ x)
    beta = inv @ x.T @ y
    residuals = y - x @ beta
    clusters = raw.item_id.to_numpy()
    unique = np.unique(clusters)
    meat = np.zeros((x.shape[1], x.shape[1]))
    for cluster in unique:
        idx = np.where(clusters == cluster)[0]
        score = x[idx].T @ residuals[idx]
        meat += np.outer(score, score)
    n, p, g = len(y), x.shape[1], len(unique)
    covariance = (g / (g - 1)) * ((n - 1) / (n - p)) * inv @ meat @ inv
    vectors = {
        "public_on_minus_off": np.array([0, 0, 1, 1, 0], float),
        "base_on_minus_off": np.array([0, 0, 1, 0, 0], float),
        "difference_in_differences": np.array([0, 0, 0, 1, 0], float),
        "base_on_minus_public_on": np.array([0, -1, 0, -1, 0], float),
    }
    rows = []
    for name, vector in vectors.items():
        estimate = float(vector @ beta)
        se = math.sqrt(max(0.0, float(vector @ covariance @ vector)))
        t_value = estimate / se if se else 0.0
        p_value = float(2 * stats.t.sf(abs(t_value), df=g - 1)) if se else 1.0
        rows.append({"metric": metric, "contrast": name, "estimate": estimate, "se_cr1": se, "t": t_value, "df": g - 1, "p": p_value})
    adjusted = holm_adjust([r["p"] for r in rows])
    for row, value in zip(rows, adjusted):
        row["p_holm"] = value
    return rows


def run() -> dict:
    raw = pd.read_csv(ROOT / "data/ratings/rag_independent_192.csv")
    final = pd.read_csv(ROOT / "data/ratings/rag_adjudicated_96.csv")
    assert len(raw) == 192 and len(final) == 96
    means = raw.groupby(["item_id", "condition_id"], as_index=False)[METRICS].mean()
    summary = []
    for condition in CONDITIONS:
        group = means[means.condition_id == condition]
        row = {"condition_id": condition, "n": len(group)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        summary.append(row)

    by_key = {(r.item_id, r.condition_id): r for r in means.itertuples(index=False)}
    items = sorted(means.item_id.unique())
    contrasts = []
    for metric in METRICS:
        metric_rows = []
        for name, weights in CONTRASTS.items():
            differences = np.array([
                sum(weight * float(getattr(by_key[(item, condition)], metric)) for condition, weight in weights.items())
                for item in items
            ])
            row = paired_test(differences)
            row.update({"metric": metric, "contrast": name})
            metric_rows.append(row)
        t_adjusted = holm_adjust([r["paired_t_p"] for r in metric_rows])
        w_adjusted = holm_adjust([r["wilcoxon_p"] for r in metric_rows])
        for row, t_value, w_value in zip(metric_rows, t_adjusted, w_adjusted):
            row["paired_t_p_holm"] = t_value
            row["wilcoxon_p_holm"] = w_value
        contrasts.extend(metric_rows)

    reliability = []
    for metric in METRICS:
        pivot = raw.pivot(index="response_code", columns="rater", values=metric).sort_index()
        a1, ak = icc_absolute(pivot[[1, 2]].to_numpy(float))
        reliability.append({"metric": metric, "icc_a1": a1, "icc_ak": ak})

    cluster = []
    for metric in METRICS:
        cluster.extend(cluster_robust(raw, metric))

    safety_flags = [
        "major_factual_error_final",
        "dangerous_failure_to_handoff_final",
        "privacy_problem_final",
        "unsupported_commitment_or_hallucination_final",
        "any_safety_error_final",
    ]
    safety_counts = []
    for condition in CONDITIONS:
        group = final[final.condition_id == condition]
        row = {"condition_id": condition, "n": len(group)}
        for flag in safety_flags:
            row[flag] = int(group[flag].sum())
        safety_counts.append(row)

    safety_tests = []
    for model, off_condition, on_condition in [
        ("base", "base_rag_off", "base_rag_on"),
        ("public", "public_chat_rag_off", "public_chat_rag_on"),
    ]:
        off = final[final.condition_id == off_condition].set_index("item_id")
        on = final[final.condition_id == on_condition].set_index("item_id")
        for flag in safety_flags:
            test = exact_mcnemar(off.loc[items, flag].to_numpy(bool), on.loc[items, flag].to_numpy(bool))
            safety_tests.append({"model": model, "flag": flag, **test})

    result = {
        "design": {"items": 24, "conditions": CONDITIONS, "raters": 2, "ratings": 192},
        "condition_summary": summary,
        "prespecified_contrasts": contrasts,
        "cluster_robust_sensitivity": cluster,
        "inter_rater_reliability": reliability,
        "adjudicated_safety_counts": safety_counts,
        "safety_mcnemar": safety_tests,
    }
    out = RESULTS / "rag"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / "condition_summary.csv", index=False, lineterminator="\n")
    pd.DataFrame(contrasts).to_csv(out / "prespecified_contrasts.csv", index=False, lineterminator="\n")
    pd.DataFrame(cluster).to_csv(out / "cluster_robust_sensitivity.csv", index=False, lineterminator="\n")
    pd.DataFrame(reliability).to_csv(out / "inter_rater_reliability.csv", index=False, lineterminator="\n")
    pd.DataFrame(safety_counts).to_csv(out / "adjudicated_safety_counts.csv", index=False, lineterminator="\n")
    pd.DataFrame(safety_tests).to_csv(out / "safety_mcnemar.csv", index=False, lineterminator="\n")
    write_json(out / "results.json", result)
    return result


if __name__ == "__main__":
    run()
