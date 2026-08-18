import unittest

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from ml_portfolio.bank_marketing import (
    LEAKY_FEATURE,
    BankCandidateSpec,
    build_bank_pipeline,
    default_bank_candidate_specs,
    default_policy_config,
    prepare_bank_modeling_data,
    run_forward_bank_validation,
    select_bank_recipe,
)


def _toy_bank_frame(rows: int = 40) -> pd.DataFrame:
    index = np.arange(rows)
    target = (index % 4 == 0).astype(int)
    return pd.DataFrame(
        {
            "age": 30 + index,
            "job": np.where(index % 2 == 0, "admin.", "technician"),
            "marital": "married",
            "education": "university.degree",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": np.where(index < rows // 2, "cellular", "telephone"),
            "month": np.where(index < rows // 2, "may", "nov"),
            "day_of_week": "mon",
            "duration": 100 + index * 3,
            "campaign": 1 + (index % 3),
            "pdays": np.where(index % 7 == 0, 999, 6),
            "previous": index % 2,
            "poutcome": np.where(index % 4 == 0, "success", "nonexistent"),
            "emp.var.rate": -1.8 + (index % 4) * 0.1,
            "cons.price.idx": 92.0 + (index % 3) * 0.2,
            "cons.conf.idx": -40.0 + (index % 5),
            "euribor3m": 1.0 + (index % 6) * 0.2,
            "nr.employed": 5000.0 + (index % 7) * 10,
            "y": np.where(target == 1, "yes", "no"),
        }
    )


class BankMarketingHelperTests(unittest.TestCase):
    def test_forest_candidates_are_class_weighted_sklearn_estimators(self) -> None:
        forest_specs = [
            spec
            for spec in default_bank_candidate_specs()
            if spec.params and spec.params.get("family") == "random_forest"
        ]

        self.assertEqual(len(forest_specs), 4)
        for spec in forest_specs:
            with self.subTest(candidate=spec.name):
                self.assertIsInstance(spec.estimator, RandomForestClassifier)
                self.assertEqual(spec.estimator.class_weight, "balanced")
                self.assertTrue(spec.name.startswith("Class-weighted random forest"))

    def test_selection_retains_fold_prevalence_ap_lift_and_pooled_diagnostics(self):
        X, y, feature_roles = prepare_bank_modeling_data(_toy_bank_frame())
        candidates = [
            BankCandidateSpec(
                name="dummy prior",
                estimator=DummyClassifier(strategy="prior"),
            )
        ]

        selection = select_bank_recipe(
            X.iloc[:30],
            y.iloc[:30],
            numeric_features=feature_roles["numeric_features"],
            categorical_features=feature_roles["categorical_features"],
            candidate_specs=candidates,
            n_splits=4,
            min_train_size=12,
        )

        self.assertEqual(len(selection.fold_results), 4)
        self.assertEqual(
            selection.fold_results.columns.tolist(),
            [
                "model",
                "fold",
                "train_rows",
                "validation_rows",
                "validation_positive_rate",
                "average_precision",
                "ap_lift_over_base_rate",
                "roc_auc",
            ],
        )
        self.assertTrue(
            np.allclose(
                selection.fold_results["average_precision"],
                selection.fold_results["validation_positive_rate"],
            )
        )
        self.assertTrue(
            np.allclose(selection.fold_results["ap_lift_over_base_rate"], 1.0)
        )
        self.assertAlmostEqual(
            float(selection.cv_results.loc[0, "cv_ap_lift_over_base_mean"]),
            1.0,
        )
        self.assertAlmostEqual(
            selection.early_oof_average_precision,
            average_precision_score(
                selection.early_oof_labels,
                selection.early_oof_scores,
            ),
        )
        self.assertAlmostEqual(
            selection.early_oof_base_positive_rate,
            float(np.mean(selection.early_oof_labels)),
        )
        self.assertAlmostEqual(
            selection.early_oof_roc_auc,
            roc_auc_score(selection.early_oof_labels, selection.early_oof_scores),
        )

    def test_prepare_bank_modeling_data_excludes_duration(self):
        X, y, feature_roles = prepare_bank_modeling_data(_toy_bank_frame())

        self.assertNotIn(LEAKY_FEATURE, X.columns)
        self.assertNotIn(LEAKY_FEATURE, feature_roles["numeric_features"])
        self.assertEqual(int(y.sum()), 10)

    def test_pre_call_pipeline_transformed_features_exclude_duration(self):
        X, y, feature_roles = prepare_bank_modeling_data(_toy_bank_frame())
        pipeline = build_bank_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=500),
            numeric_features=feature_roles["numeric_features"],
            categorical_features=feature_roles["categorical_features"],
        )

        pipeline.fit(X, y)
        transformed_names = pipeline.named_steps["preprocess"].get_feature_names_out()

        self.assertFalse(any(LEAKY_FEATURE in name for name in transformed_names))

    def test_pre_call_pipeline_keeps_all_missing_numeric_features(self):
        frame = _toy_bank_frame()
        frame["pdays"] = 999
        X, y, feature_roles = prepare_bank_modeling_data(frame)
        pipeline = build_bank_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=500),
            numeric_features=feature_roles["numeric_features"],
            categorical_features=feature_roles["categorical_features"],
        )

        pipeline.fit(X, y)
        transformed_names = pipeline.named_steps["preprocess"].get_feature_names_out()

        self.assertTrue(any("pdays_clean" in name for name in transformed_names))

    def test_late_label_mutation_does_not_change_selection_or_policy(self):
        frame = _toy_bank_frame()
        X, y, feature_roles = prepare_bank_modeling_data(frame)
        candidates = [
            BankCandidateSpec(
                name="dummy majority",
                estimator=DummyClassifier(strategy="prior"),
            ),
            BankCandidateSpec(
                name="balanced logistic C=1.0",
                estimator=LogisticRegression(class_weight="balanced", C=1.0, max_iter=500),
            ),
        ]

        first = run_forward_bank_validation(
            X,
            y,
            numeric_features=feature_roles["numeric_features"],
            categorical_features=feature_roles["categorical_features"],
            candidate_specs=candidates,
            min_train_size=12,
        )
        mutated_y = y.copy()
        mutated_y.iloc[int(len(y) * 0.75) :] = 1 - mutated_y.iloc[int(len(y) * 0.75) :]
        second = run_forward_bank_validation(
            X,
            mutated_y,
            numeric_features=feature_roles["numeric_features"],
            categorical_features=feature_roles["categorical_features"],
            candidate_specs=candidates,
            min_train_size=12,
        )

        self.assertEqual(first.selection.selected_name, second.selection.selected_name)
        self.assertEqual(first.selection.selected_params, second.selection.selected_params)
        self.assertEqual(first.policy_config, second.policy_config)
        self.assertEqual(first.outer_split.early_indices.tolist(), second.outer_split.early_indices.tolist())
        self.assertEqual(first.outer_split.late_indices.tolist(), second.outer_split.late_indices.tolist())
        self.assertEqual(first.late_metrics["n_samples"], second.late_metrics["n_samples"])

    def test_default_policy_config_is_fixed_budget_only(self):
        policy = default_policy_config()

        self.assertEqual(policy["primary_budget_share"], 0.10)
        self.assertEqual(policy["budget_shares"], (0.01, 0.05, 0.10, 0.20, 0.30))
        self.assertNotIn("threshold_metric", policy)


if __name__ == "__main__":
    unittest.main()
