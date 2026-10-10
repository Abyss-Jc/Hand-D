"""Assisted curation only uses verifiably out-of-training predictions."""
import unittest

from handd_core.review_assessment import rank_suggestions


class ReviewAssessmentTests(unittest.TestCase):
    def test_disagreement_first_then_low_validation_derived_margin(self):
        rows = [
            {"sample_id": "a", "true_label": "Fist", "predicted_label": "Idle",
             "top2_margin": .82, "trained_on_this_sample": False},
            {"sample_id": "b", "true_label": "Fist", "predicted_label": "Fist",
             "top2_margin": .03, "trained_on_this_sample": False},
            {"sample_id": "c", "true_label": "Fist", "predicted_label": "Fist",
             "top2_margin": .67, "trained_on_this_sample": False},
            {"sample_id": "d", "true_label": "Fist", "predicted_label": "Idle",
             "top2_margin": .001, "trained_on_this_sample": True},
        ]
        result = rank_suggestions(rows, margin_threshold=.10)
        self.assertEqual([r["sample_id"] for r in result], ["a", "b"])
        self.assertEqual([r["reason"] for r in result], ["model_disagreement", "low_margin"])

    def test_invalid_or_in_sample_scores_never_influence_review_queue(self):
        with self.assertRaises(ValueError):
            rank_suggestions([{"sample_id": "a", "top2_margin": float("nan")}], .10)
        with self.assertRaises(ValueError):
            rank_suggestions([], margin_threshold=1.2)
        self.assertEqual(rank_suggestions([
            {"sample_id": "a", "true_label": "Fist", "predicted_label": "Idle",
             "top2_margin": .2, "trained_on_this_sample": True},
        ], .1), [])
