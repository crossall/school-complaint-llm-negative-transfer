from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .common import RESULTS, ROOT, cochran_q, holm_adjust, icc_absolute, paired_dz, write_json


CONDITIONS = ["base", "public_chat", "private_chat", "mixed_chat"]
METRICS = [
    "factual_accuracy_mean",
    "handling_appropriateness_mean",
    "requirement_fulfillment_mean",
    "safety_administrative_appropriateness_mean",
    "composite_mean",
]


def run() -> dict:
    data = pd.read_csv(ROOT / "data/ratings/experiment1_96.csv")
    assert len(data) == 96 and data["item_id"].nunique() == 24
    summary_rows = []
    for condition in CONDITIONS:
        group = data[data.condition == condition]
        row = {
            "condition": condition,
            "n": len(group),
            "handling_correct": int(group.handling_correct.sum()),
            "format_compliant": int(group.format_compliant.sum()),
            "mean_generated_tokens": float(group.generated_tokens.mean()),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)

    omnibus, contrasts = [], []
    for metric in METRICS:
        wide = data.pivot(index="item_id", columns="condition", values=metric)[CONDITIONS]
        statistic, p = stats.friedmanchisquare(*(wide[c] for c in CONDITIONS))
        omnibus.append({
            "metric": metric,
            "friedman_chi2": float(statistic),
            "p": float(p),
            "kendall_w": float(statistic / (len(wide) * (len(CONDITIONS) - 1))),
        })
        rows = []
        for condition in CONDITIONS[1:]:
            diff = (wide[condition] - wide.base).to_numpy()
            test = stats.wilcoxon(diff, alternative="two-sided", method="auto")
            rows.append({
                "metric": metric,
                "contrast": f"{condition}-base",
                "mean_difference": float(diff.mean()),
                "wilcoxon_w": float(test.statistic),
                "wilcoxon_p": float(test.pvalue),
                "paired_dz": paired_dz(diff),
            })
        adjusted = holm_adjust([r["wilcoxon_p"] for r in rows])
        for row, value in zip(rows, adjusted):
            row["wilcoxon_p_holm"] = value
        contrasts.extend(rows)

    direction = data.pivot(index="item_id", columns="condition", values="handling_correct")[CONDITIONS]
    q, q_p = cochran_q(direction.to_numpy(bool))

    reliability = []
    prefixes = [m.removesuffix("_mean") for m in METRICS[:-1]]
    for prefix in prefixes:
        matrix = data[[f"{prefix}_rater1", f"{prefix}_rater2"]].to_numpy(float)
        a1, ak = icc_absolute(matrix)
        reliability.append({"metric": prefix, "icc_a1": a1, "icc_ak": ak})
    composite = np.column_stack([
        data[[f"{p}_rater1" for p in prefixes]].mean(axis=1),
        data[[f"{p}_rater2" for p in prefixes]].mean(axis=1),
    ])
    a1, ak = icc_absolute(composite)
    reliability.append({"metric": "composite", "icc_a1": a1, "icc_ak": ak})

    result = {
        "design": {"items": 24, "conditions": CONDITIONS, "raters": 2},
        "condition_summary": summary_rows,
        "omnibus": omnibus,
        "base_contrasts": contrasts,
        "handling_direction_cochran_q": {"q": q, "df": 3, "p": q_p},
        "inter_rater_reliability": reliability,
    }
    out = RESULTS / "experiment1"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out / "condition_summary.csv", index=False, lineterminator="\n")
    pd.DataFrame(omnibus).to_csv(out / "omnibus_tests.csv", index=False, lineterminator="\n")
    pd.DataFrame(contrasts).to_csv(out / "base_contrasts.csv", index=False, lineterminator="\n")
    pd.DataFrame(reliability).to_csv(out / "inter_rater_reliability.csv", index=False, lineterminator="\n")
    write_json(out / "results.json", result)
    return result


if __name__ == "__main__":
    run()
