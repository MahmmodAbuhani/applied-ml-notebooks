from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ml_portfolio.plotting import ACCENT, apply_portfolio_style, save_figure


class PlottingHelperTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_apply_portfolio_style_sets_expected_accent_cycle(self):
        apply_portfolio_style()

        first_color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]

        self.assertEqual(first_color, ACCENT)

    def test_save_figure_writes_to_assets_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])

            output_path = save_figure(fig, "example", project_root=Path(tmp))

            self.assertEqual(output_path, Path(tmp) / "assets" / "figures" / "example.png")
            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)

    def test_save_figure_honors_explicit_evidence_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "evidence-figures"
            fig, ax = plt.subplots()
            ax.plot([0, 1], [1, 0])

            with patch.dict(os.environ, {"ML_PORTFOLIO_FIGURE_DIR": str(output_dir)}):
                output_path = save_figure(fig, "example")

            self.assertEqual(output_path, output_dir / "example.png")
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
