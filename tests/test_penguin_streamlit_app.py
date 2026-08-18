import unittest

import pandas as pd

from demo.penguin_streamlit_app import (
    _contribution_chart,
    _contribution_summary,
    _out_of_distribution_messages,
    _probability_chart,
    _probability_summary,
    _training_distribution_summary,
    _training_distribution_chart,
)
from ml_portfolio.penguins import PenguinSample


class PenguinStreamlitAppTests(unittest.TestCase):
    def test_probability_chart_fixes_x_axis_to_probability_domain(self):
        chart = _probability_chart(
            {
                "Adelie": 0.05,
                "Chinstrap": 0.86,
                "Gentoo": 0.09,
            }
        )

        spec = chart.to_dict()

        self.assertEqual(spec["mark"]["type"], "bar")
        self.assertEqual(spec["encoding"]["x"]["field"], "probability")
        self.assertEqual(spec["encoding"]["x"]["scale"]["domain"], [0, 1])
        self.assertEqual(spec["encoding"]["x"]["axis"]["format"], ".0%")
        self.assertEqual(spec["encoding"]["y"]["field"], "species")

    def test_training_distribution_chart_layers_data_and_user_sample(self):
        data = pd.DataFrame(
            {
                "species": ["Adelie", "Chinstrap"],
                "bill_length_mm": [38.5, 50.1],
                "flipper_length_mm": [181.0, 201.0],
            }
        )
        sample = PenguinSample(
            island="Dream",
            bill_length_mm=45.2,
            bill_depth_mm=16.4,
            flipper_length_mm=196.0,
            body_mass_g=4150.0,
            sex="female",
            year="2008",
        )

        spec = _training_distribution_chart(data, sample).to_dict()

        self.assertEqual(len(spec["layer"]), 2)
        self.assertEqual(spec["layer"][0]["encoding"]["x"]["field"], "bill_length_mm")
        self.assertEqual(spec["layer"][0]["encoding"]["y"]["field"], "flipper_length_mm")
        self.assertEqual(spec["layer"][0]["encoding"]["color"]["field"], "species")
        self.assertEqual(spec["layer"][1]["encoding"]["x"]["field"], "bill_length_mm")
        self.assertEqual(spec["layer"][1]["encoding"]["y"]["field"], "flipper_length_mm")
        self.assertIn("scale", spec["layer"][0]["encoding"]["x"])
        self.assertIn("scale", spec["layer"][0]["encoding"]["y"])
        self.assertEqual(spec["layer"][0]["encoding"]["x"]["scale"]["domain"], [37.34, 51.26])
        self.assertEqual(spec["layer"][0]["encoding"]["y"]["scale"]["domain"], [179.0, 203.0])

    def test_contribution_chart_sets_finite_numeric_domain(self):
        spec = _contribution_chart(
            [
                {
                    "feature": "bill_length_mm",
                    "contribution": 0.25,
                    "abs_contribution": 0.25,
                    "direction": "supports",
                },
                {
                    "feature": "island_Dream",
                    "contribution": -0.10,
                    "abs_contribution": 0.10,
                    "direction": "pulls against",
                },
            ]
        ).to_dict()

        self.assertIn("scale", spec["encoding"]["x"])
        self.assertEqual(spec["encoding"]["x"]["scale"]["domain"], [-0.135, 0.285])

    def test_out_of_distribution_messages_name_feature_and_observed_range(self):
        sample = PenguinSample(
            island="Dream",
            bill_length_mm=70.0,
            bill_depth_mm=16.4,
            flipper_length_mm=196.0,
            body_mass_g=4150.0,
            sex="female",
            year="2008",
        )
        ranges = {
            "bill_length_mm": (32.1, 44.4, 59.6),
            "bill_depth_mm": (13.1, 17.3, 21.5),
            "flipper_length_mm": (172.0, 197.0, 231.0),
            "body_mass_g": (2700.0, 4050.0, 6300.0),
        }

        messages = _out_of_distribution_messages(sample, ranges)

        self.assertEqual(
            messages,
            ["Bill length (mm) is outside the observed range: 32.1 to 59.6."],
        )

    def test_probability_summary_exposes_sorted_values_as_text(self):
        summary = _probability_summary(
            {
                "Adelie": 0.05,
                "Chinstrap": 0.86,
                "Gentoo": 0.09,
            }
        )

        self.assertEqual(
            summary,
            "Probability readout: Chinstrap 86.0%; Gentoo 9.0%; Adelie 5.0%.",
        )

    def test_training_summary_describes_the_marked_input(self):
        sample = PenguinSample(
            island="Dream",
            bill_length_mm=45.2,
            bill_depth_mm=16.4,
            flipper_length_mm=196.0,
            body_mass_g=4150.0,
            sex="female",
            year="2008",
        )

        summary = _training_distribution_summary(sample)

        self.assertIn("diamond marks the current input", summary)
        self.assertIn("45.2 mm bill length", summary)
        self.assertIn("196 mm flipper length", summary)

    def test_contribution_summary_is_textual_and_noncausal(self):
        summary = _contribution_summary(
            [
                {
                    "feature": "bill length mm",
                    "contribution": 0.25,
                    "abs_contribution": 0.25,
                    "direction": "supports",
                },
                {
                    "feature": "island Dream",
                    "contribution": -0.10,
                    "abs_contribution": 0.10,
                    "direction": "pulls against",
                },
            ]
        )

        self.assertEqual(
            summary,
            "Largest model-internal contribution: bill length mm supports "
            "the prediction (+0.250 logit). This is not a causal explanation.",
        )


if __name__ == "__main__":
    unittest.main()
