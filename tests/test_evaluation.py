import math
import unittest

from sklearn.linear_model import LogisticRegression

import ml_portfolio
from ml_portfolio import evaluation
from ml_portfolio.evaluation import classification_summary, regression_summary


class EvaluationHelperTests(unittest.TestCase):
    def test_repeated_classification_summary_is_public_package_api(self):
        self.assertIs(
            getattr(ml_portfolio, "repeated_classification_summary", None),
            evaluation.repeated_classification_summary,
        )

    def test_repeated_classification_summary_reports_deterministic_uncertainty(self):
        X = [[-3], [-2], [-1], [-0.5], [0.5], [1], [2], [3], [-2.5], [-1.5], [1.5], [2.5]]
        y = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
        estimator = LogisticRegression(max_iter=1000, random_state=42)

        summary_function = getattr(evaluation, "repeated_classification_summary", None)
        self.assertIsNotNone(summary_function, "repeated classification helper is missing")

        summary = summary_function(
            estimator,
            X,
            y,
            n_splits=3,
            n_repeats=2,
            random_state=42,
        )
        repeated = summary_function(
            estimator,
            X,
            y,
            n_splits=3,
            n_repeats=2,
            random_state=42,
        )

        self.assertEqual(summary, repeated)
        self.assertEqual(summary["n_splits"], 3)
        self.assertEqual(summary["n_repeats"], 2)
        self.assertEqual(summary["n_folds"], 6)
        self.assertGreaterEqual(summary["macro_f1_std"], 0.0)
        self.assertIn("macro_f1_mean", summary)

    def test_classification_summary_reports_standard_metrics(self):
        summary = classification_summary(
            y_true=[0, 0, 1, 1],
            y_pred=[0, 1, 1, 1],
            model_name="toy classifier",
        )

        self.assertEqual(summary["model"], "toy classifier")
        self.assertEqual(summary["n_samples"], 4)
        self.assertAlmostEqual(summary["accuracy"], 0.75)
        self.assertAlmostEqual(summary["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(summary["macro_f1"], 0.7333333333333334)
        self.assertAlmostEqual(summary["weighted_f1"], 0.7333333333333334)

    def test_regression_summary_reports_error_and_fit_metrics(self):
        summary = regression_summary(
            y_true=[3, -0.5, 2, 7],
            y_pred=[2.5, 0, 2, 8],
            model_name="toy regressor",
        )

        self.assertEqual(summary["model"], "toy regressor")
        self.assertEqual(summary["n_samples"], 4)
        self.assertAlmostEqual(summary["rmse"], math.sqrt(0.375))
        self.assertAlmostEqual(summary["mae"], 0.5)
        self.assertAlmostEqual(summary["r2"], 0.9486081370449679)


if __name__ == "__main__":
    unittest.main()
