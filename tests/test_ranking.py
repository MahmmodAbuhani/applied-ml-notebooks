import math
import unittest

import numpy as np

from ml_portfolio.ranking import (
    average_precision_lift,
    calibration_by_score_band,
    cumulative_gains_frame,
    expanding_window_splits,
    fixed_budget_table,
    source_order_split_indices,
)


class RankingHelperTests(unittest.TestCase):
    def test_average_precision_lift_normalizes_for_target_prevalence(self):
        lift = average_precision_lift(
            y_true=[1, 0, 0, 1],
            scores=[0.9, 0.8, 0.7, 0.6],
        )

        self.assertAlmostEqual(lift, 1.5)

    def test_average_precision_lift_is_nan_without_positive_targets(self):
        lift = average_precision_lift(
            y_true=[0, 0, 0],
            scores=[0.9, 0.4, 0.1],
        )

        self.assertTrue(math.isnan(lift))

    def test_source_order_split_uses_frozen_cutoff_without_overlap(self):
        early, late = source_order_split_indices(20, train_fraction=0.75)

        self.assertEqual(early.tolist(), list(range(15)))
        self.assertEqual(late.tolist(), list(range(15, 20)))
        self.assertEqual(set(early).intersection(set(late)), set())

    def test_expanding_window_splits_preserve_chronology(self):
        splits = list(expanding_window_splits(20, n_splits=4, min_train_size=8))

        self.assertEqual(len(splits), 4)
        self.assertEqual(splits[0][0].tolist(), list(range(8)))
        self.assertEqual(splits[0][1].tolist(), [8, 9, 10])
        self.assertEqual(splits[-1][0].tolist(), list(range(17)))
        self.assertEqual(splits[-1][1].tolist(), [17, 18, 19])
        for train_index, validation_index in splits:
            self.assertLess(max(train_index), min(validation_index))
            self.assertEqual(set(train_index).intersection(set(validation_index)), set())

    def test_expanding_window_splits_reject_invalid_small_inputs(self):
        with self.assertRaisesRegex(ValueError, "at least one validation row"):
            list(expanding_window_splits(10, n_splits=4, min_train_size=7))

    def test_fixed_budget_table_reports_lift_and_capture(self):
        table = fixed_budget_table(
            y_true=[0, 1, 0, 1],
            scores=[0.9, 0.8, 0.7, 0.1],
            budget_shares=[0.25, 0.50, 1.00],
        )

        self.assertEqual(table["contacts"].tolist(), [1, 2, 4])
        self.assertEqual(table["responders"].tolist(), [0, 1, 2])
        self.assertAlmostEqual(table.loc[1, "response_rate"], 0.5)
        self.assertAlmostEqual(table.loc[1, "lift_vs_base"], 1.0)
        self.assertAlmostEqual(table.loc[1, "responders_captured_share"], 0.5)

    def test_fixed_budget_table_handles_zero_positive_targets(self):
        table = fixed_budget_table(
            y_true=[0, 0, 0, 0],
            scores=[0.9, 0.8, 0.7, 0.1],
            budget_shares=[0.5],
        )

        self.assertEqual(table.loc[0, "responders"], 0)
        self.assertEqual(table.loc[0, "response_rate"], 0.0)
        self.assertEqual(table.loc[0, "responders_captured_share"], 0.0)
        self.assertTrue(math.isnan(table.loc[0, "lift_vs_base"]))

    def test_cumulative_gains_frame_orders_scores_and_handles_zero_positives(self):
        frame = cumulative_gains_frame(
            y_true=[0, 0, 0],
            scores=[0.2, 0.9, 0.4],
            label="zero positive",
        )

        self.assertEqual(frame["score"].tolist(), [0.9, 0.4, 0.2])
        self.assertTrue(np.allclose(frame["responders_captured_share"], 0.0))

    def test_calibration_by_score_band_uses_score_ordered_bins(self):
        frame = calibration_by_score_band(
            y_true=[0, 1, 0, 1],
            scores=[0.1, 0.9, 0.4, 0.8],
            n_bins=2,
        )

        self.assertEqual(frame["score_band"].tolist(), [1, 2])
        self.assertEqual(frame["rows"].tolist(), [2, 2])
        self.assertAlmostEqual(frame.loc[0, "mean_score"], 0.85)
        self.assertAlmostEqual(frame.loc[0, "response_rate"], 1.0)
        self.assertAlmostEqual(frame.loc[1, "mean_score"], 0.25)
        self.assertAlmostEqual(frame.loc[1, "response_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
