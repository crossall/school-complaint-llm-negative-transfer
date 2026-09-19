from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .common import RESULTS, ROOT, holm_adjust, icc_absolute, paired_dz, write_json


CONDITIONS = ["base", "public_seed2026", "public_seed2027", "public_seed2028"]
PUBLIC = CONDITIONS[1:]
METRICS = [
    "factual_accuracy_mean",
    "handling_appropriateness_mean",
    "requirement_fulfillment_mean",
    "safety_administrative_appropriateness_mean",
    "composite_mean",
]


def format_compliant_sensitivity(data: pd.DataFrame) -> list[dict]:
    """Compare each Public seed with Base only where that Public response has all five sections.

    scipy==1.16.2 and method='auto' are pinned because the exact/asymptotic
    choice affects the reported p-value for the n=10 seed-2028 subset.
    """
    base = data[data.condition == "base"].set_index("item_id")
    rows = []
    for condition in PUBLIC:
        public = data[(data.condition == condition) & data.format_compliant.astype(bool)].set_index("item_id")
        item_ids = sorted(public.index.intersection(base.index))
        differences = public.loc[item_ids, "composite_mean"].to_numpy() - base.loc[item_ids, "composite_mean"].to_numpy()
        test = stats.wilcoxon(differences, alternative="two-sided", method="auto")
        rows.append({
            "public_condition": condition,
            "restriction": "Public response complied with the required five-section format",
            "n": len(item_ids),
            "mean_public_minus_base": float(differences.mean()),
            "wilcoxon_w": float(test.statistic),
            "wilcoxon_p": float(test.pvalue),
            "paired_dz": paired_dz(differences),
        })
    adjusted = holm_adjust([r["wilcoxon_p"] for r in rows])
    for row, value in zip(rows, adjusted):
        row["wilcoxon_p_holm"] = value
    return rows


def run() -> dict:
    data = pd.read_csv(ROOT / "data/ratings/public_multiseed_96.csv")
    assert len(data) == 96 and data.item_id.nunique() == 24
    summary = []
    for condition in CONDITIONS:
        group = data[data.condition == condition]
        summary.append({
            "condition": condition,
            "n": len(group),
            "composite_mean": float(group.composite_mean.mean()),
            "composite_sd": float(group.composite_mean.std(ddof=1)),
            "handling_mean": float(group.handling_appropriateness_mean.mean()),
            "requirement_mean": float(group.requirement_fulfillment_mean.mean()),
            "safety_admin_mean": float(group.safety_administrative_appropriateness_mean.mean()),
            "factual_mean": float(group.factual_accuracy_mean.mean()),
            "handling_correct": int(group.handling_correct.sum()),
            "format_compliant": int(group.format_compliant.sum()),
            "mean_generated_tokens": float(group.generated_tokens.mean()),
        })

    pairwise = []
    public_omnibus = []
    for metric in METRICS:
        wide = data.pivot(index="item_id", columns="condition", values=metric)[CONDITIONS]
        rows = []
        for condition in PUBLIC:
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
        pairwise.extend(rows)
        statistic, p = stats.friedmanchisquare(*(wide[c] for c in PUBLIC))
        public_omnibus.append({
            "metric": metric,
            "friedman_chi2": float(statistic),
            "p": float(p),
            "kendall_w": float(statistic / (len(wide) * (len(PUBLIC) - 1))),
        })

    sensitivity = format_compliant_sensitivity(data)
    added = data[data.condition.isin(["public_seed2027", "public_seed2028"])]
    prefixes = [m.removesuffix("_mean") for m in METRICS[:-1]]
    reliability = []
    for prefix in prefixes:
        matrix = added[[f"{prefix}_rater1", f"{prefix}_rater2"]].to_numpy(float)
        a1, ak = icc_absolute(matrix)
        reliability.append({"metric": prefix, "icc_a1": a1, "icc_ak": ak})
    composite = np.column_stack([
        added[[f"{p}_rater1" for p in prefixes]].mean(axis=1),
        added[[f"{p}_rater2" for p in prefixes]].mean(axis=1),
    ])
    a1, ak = icc_absolute(composite)
    reliability.append({"metric": "composite", "icc_a1": a1, "icc_ak": ak})

    result = {
        "design": {"items": 24, "public_training_seeds": [2026, 2027, 2028], "same_base": True},
        "condition_summary": summary,
        "base_contrasts": pairwise,
        "public_seed_omnibus": public_omnibus,
        "format_compliant_sensitivity": sensitivity,
        "added_session_reliability": reliability,
    }
    out = RESULTS / "public_multiseed"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(out / "condition_summary.csv", index=False, lineterminator="\n")
    pd.DataFrame(pairwise).to_csv(out / "base_contrasts.csv", index=False, lineterminator="\n")
    pd.DataFrame(public_omnibus).to_csv(out / "public_seed_omnibus.csv", index=False, lineterminator="\n")
    pd.DataFrame(sensitivity).to_csv(out / "format_compliant_sensitivity.csv", index=False, lineterminator="\n")
    pd.DataFrame(reliability).to_csv(out / "added_session_reliability.csv", index=False, lineterminator="\n")
    write_json(out / "results.json", result)
    return result


if __name__ == "__main__":
    run()
