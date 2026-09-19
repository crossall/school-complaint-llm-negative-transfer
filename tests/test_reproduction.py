from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.analyze_experiment1 import run as run_experiment1
from analysis.analyze_multiseed import run as run_multiseed
from analysis.analyze_rag import run as run_rag
from analysis.analyze_retrieval import run as run_retrieval


ROOT = Path(__file__).resolve().parents[1]


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.exp1 = run_experiment1()
        cls.multi = run_multiseed()
        cls.rag = run_rag()
        cls.retrieval = run_retrieval()

    def test_experiment1_means(self) -> None:
        found = {row["condition"]: row["composite_mean_mean"] for row in self.exp1["condition_summary"]}
        expected = {"base": 4.265625, "public_chat": 3.4895833333333335, "private_chat": 3.890625, "mixed_chat": 3.9583333333333335}
        for key, value in expected.items():
            self.assertAlmostEqual(found[key], value, places=12)

    def test_format_compliant_sensitivity(self) -> None:
        rows = self.multi["format_compliant_sensitivity"]
        self.assertEqual([row["n"] for row in rows], [20, 14, 10])
        np.testing.assert_allclose(
            [row["mean_public_minus_base"] for row in rows],
            [-0.69375, -0.7410714285714286, -0.75],
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            [row["wilcoxon_p_holm"] for row in rows],
            [0.0012733152106562185, 0.0046241135067514386, 0.0046241135067514386],
            rtol=0,
            atol=1e-12,
        )

    def test_multiseed_negative_transfer(self) -> None:
        rows = [row for row in self.multi["base_contrasts"] if row["metric"] == "composite_mean"]
        self.assertTrue(all(row["mean_difference"] < 0 for row in rows))
        self.assertTrue(all(row["wilcoxon_p_holm"] < 0.001 for row in rows))

    def test_rag_means_and_safety(self) -> None:
        means = {row["condition_id"]: row["composite_mean"] for row in self.rag["condition_summary"]}
        self.assertAlmostEqual(means["base_rag_off"], 4.098958333333333, places=12)
        self.assertAlmostEqual(means["public_chat_rag_on"], 3.1197916666666665, places=12)
        counts = {row["condition_id"]: row["any_safety_error_final"] for row in self.rag["adjudicated_safety_counts"]}
        self.assertEqual(counts, {"base_rag_off": 2, "base_rag_on": 5, "public_chat_rag_off": 2, "public_chat_rag_on": 13})
        public_any = next(row for row in self.rag["safety_mcnemar"] if row["model"] == "public" and row["flag"] == "any_safety_error_final")
        self.assertAlmostEqual(public_any["exact_p"], 0.00341796875, places=14)

    def test_retrieval_results(self) -> None:
        v1 = self.retrieval["v1"]
        self.assertEqual(v1["direct_evidence_passages"], 9)
        self.assertEqual(v1["items_without_direct_evidence"], 40)
        self.assertEqual(v1["items_with_misleading_passage"], 42)
        test = self.retrieval["v2_frozen_output"]["splits"]["test"]
        self.assertEqual(test["v2_gold_hit_at_4"], 0.0)
        self.assertEqual(test["v2_no_context_accuracy"], 0.6)

    def test_data_shapes(self) -> None:
        expected = {
            "data/ratings/experiment1_96.csv": 96,
            "data/ratings/public_multiseed_96.csv": 96,
            "data/ratings/rag_independent_192.csv": 192,
            "data/ratings/rag_adjudicated_96.csv": 96,
            "data/retrieval/v1_top4_192.csv": 192,
            "data/retrieval/v1_relevance_audit_192.csv": 192,
        }
        for relative, rows in expected.items():
            self.assertEqual(len(pd.read_csv(ROOT / relative)), rows, relative)


if __name__ == "__main__":
    unittest.main()
