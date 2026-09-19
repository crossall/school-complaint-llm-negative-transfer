from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * float(pvalues[index])))
        adjusted[index] = running
    return adjusted.tolist()


def paired_dz(differences: np.ndarray) -> float | None:
    differences = np.asarray(differences, dtype=float)
    sd = float(differences.std(ddof=1))
    return float(differences.mean() / sd) if sd else None


def paired_test(differences: np.ndarray) -> dict[str, float | int | None]:
    differences = np.asarray(differences, dtype=float)
    n = len(differences)
    mean = float(differences.mean())
    sd = float(differences.std(ddof=1))
    se = sd / math.sqrt(n)
    t_stat = mean / se if se else 0.0
    t_p = float(2 * stats.t.sf(abs(t_stat), df=n - 1)) if se else 1.0
    critical = float(stats.t.ppf(0.975, df=n - 1))
    if np.allclose(differences, 0):
        w_stat, w_p = 0.0, 1.0
    else:
        test = stats.wilcoxon(differences, alternative="two-sided", method="auto")
        w_stat, w_p = float(test.statistic), float(test.pvalue)
    return {
        "n": n,
        "mean_difference": mean,
        "sd_difference": sd,
        "ci95_low": mean - critical * se,
        "ci95_high": mean + critical * se,
        "paired_dz": paired_dz(differences),
        "paired_t": t_stat,
        "paired_t_p": t_p,
        "wilcoxon_w": w_stat,
        "wilcoxon_p": w_p,
    }


def icc_absolute(matrix: np.ndarray) -> tuple[float, float]:
    """McGraw-Wong absolute-agreement ICC(A,1) and ICC(A,k)."""
    x = np.asarray(matrix, dtype=float)
    n, k = x.shape
    grand = x.mean()
    row_mean = x.mean(axis=1)
    col_mean = x.mean(axis=0)
    msr = k * np.square(row_mean - grand).sum() / (n - 1)
    msc = n * np.square(col_mean - grand).sum() / (k - 1)
    residual = x - row_mean[:, None] - col_mean[None, :] + grand
    mse = np.square(residual).sum() / ((n - 1) * (k - 1))
    a1 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
    ak = (msr - mse) / (msr + (msc - mse) / n)
    return float(a1), float(ak)


def exact_mcnemar(a: np.ndarray, b: np.ndarray) -> dict[str, int | float]:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    off_only = int(np.sum(a & ~b))
    on_only = int(np.sum(~a & b))
    discordant = off_only + on_only
    p = float(stats.binomtest(min(off_only, on_only), discordant, 0.5).pvalue) if discordant else 1.0
    return {"off_only": off_only, "on_only": on_only, "discordant": discordant, "exact_p": p}


def cochran_q(binary_matrix: np.ndarray) -> tuple[float, float]:
    x = np.asarray(binary_matrix, dtype=float)
    n, k = x.shape
    column_sums = x.sum(axis=0)
    row_sums = x.sum(axis=1)
    denominator = k * row_sums.sum() - np.square(row_sums).sum()
    q = (k - 1) * (k * np.square(column_sums).sum() - column_sums.sum() ** 2) / denominator
    return float(q), float(stats.chi2.sf(q, k - 1))
