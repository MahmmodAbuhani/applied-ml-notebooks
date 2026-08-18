import subprocess
import sys
import unittest
from pathlib import Path


class PredictPenguinSpeciesCliTests(unittest.TestCase):
    def test_cli_accepts_local_data_url_for_network_free_smoke_test(self):
        fixture_url = (Path(__file__).parent / "fixtures" / "penguins_toy.csv").resolve().as_uri()

        result = subprocess.run(
            [
                sys.executable,
                "demo/predict_penguin_species.py",
                "--data-url",
                fixture_url,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("Predicted species:", result.stdout)
        self.assertIn("Class probabilities:", result.stdout)
        self.assertIn("Boundary: educational public-data demo", result.stdout)


if __name__ == "__main__":
    unittest.main()
