import unittest

import pandas as pd

from ml_portfolio.penguins import (
    FEATURE_COLUMNS,
    PenguinSample,
    explain_penguin_prediction,
    fit_penguin_model,
    predict_penguin_sample,
    prepare_penguin_modeling_data,
    validate_penguin_columns,
)


def _toy_penguins() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "species": [
                "Adelie",
                "Adelie",
                "Chinstrap",
                "Chinstrap",
                "Gentoo",
                "Gentoo",
                "Adelie",
                "Chinstrap",
                "Gentoo",
            ],
            "island": [
                "Torgersen",
                "Torgersen",
                "Dream",
                "Dream",
                "Biscoe",
                "Biscoe",
                "Dream",
                "Dream",
                "Biscoe",
            ],
            "bill_length_mm": [39.1, 38.7, 46.5, 50.1, 46.1, 49.8, 40.3, 49.0, 47.2],
            "bill_depth_mm": [18.7, 19.0, 17.9, 19.5, 13.2, 15.0, 18.2, 18.1, 14.2],
            "flipper_length_mm": [181, 185, 192, 196, 211, 220, 190, 195, 215],
            "body_mass_g": [3750, 3650, 3500, 3900, 4500, 5000, 3800, 3700, 4700],
            "sex": ["male", "female", "female", "male", "female", "male", "female", "female", "male"],
            "year": [2007, 2008, 2007, 2008, 2007, 2008, 2009, 2009, 2009],
        }
    )


class PenguinModelingTests(unittest.TestCase):
    def test_validate_penguin_columns_rejects_missing_required_feature(self):
        data = _toy_penguins().drop(columns=["bill_length_mm"])

        with self.assertRaisesRegex(ValueError, "bill_length_mm"):
            validate_penguin_columns(data)

    def test_prepare_penguin_modeling_data_drops_incomplete_rows_and_casts_year(self):
        data = _toy_penguins()
        data.loc[0, "sex"] = None

        X, y = prepare_penguin_modeling_data(data)

        self.assertEqual(list(X.columns), FEATURE_COLUMNS)
        self.assertEqual(len(X), 8)
        self.assertEqual(len(y), 8)
        self.assertEqual(X["year"].dtype, object)

    def test_predict_penguin_sample_returns_prediction_and_probabilities(self):
        model = fit_penguin_model(_toy_penguins())
        sample = PenguinSample(
            island="Biscoe",
            bill_length_mm=48.0,
            bill_depth_mm=14.5,
            flipper_length_mm=215.0,
            body_mass_g=4700.0,
            sex="female",
            year="2008",
        )

        result = predict_penguin_sample(model, sample)

        self.assertIn(result.predicted_species, {"Adelie", "Chinstrap", "Gentoo"})
        self.assertEqual(set(result.probabilities), {"Adelie", "Chinstrap", "Gentoo"})
        self.assertAlmostEqual(sum(result.probabilities.values()), 1.0)

    def test_explain_penguin_prediction_returns_ranked_feature_contributions(self):
        model = fit_penguin_model(_toy_penguins())
        sample = PenguinSample(
            island="Dream",
            bill_length_mm=49.0,
            bill_depth_mm=18.1,
            flipper_length_mm=195.0,
            body_mass_g=3700.0,
            sex="female",
            year="2009",
        )

        explanation = explain_penguin_prediction(model, sample, top_n=3)

        self.assertEqual(len(explanation), 3)
        self.assertGreaterEqual(explanation[0]["abs_contribution"], explanation[1]["abs_contribution"])
        self.assertIn("feature", explanation[0])
        self.assertIn("direction", explanation[0])
        self.assertTrue(all("_" not in str(row["feature"]) for row in explanation))


if __name__ == "__main__":
    unittest.main()
