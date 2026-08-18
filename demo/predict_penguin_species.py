"""Train a Palmer Penguins demo model and predict one sample."""

from __future__ import annotations

import argparse

from ml_portfolio.penguins import (
    PenguinSample,
    explain_penguin_prediction,
    fit_penguin_model,
    load_penguin_data,
    predict_penguin_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--island", default="Dream")
    parser.add_argument("--bill-length-mm", type=float, default=45.2)
    parser.add_argument("--bill-depth-mm", type=float, default=16.4)
    parser.add_argument("--flipper-length-mm", type=float, default=196.0)
    parser.add_argument("--body-mass-g", type=float, default=4150.0)
    parser.add_argument("--sex", default="female", choices=["female", "male"])
    parser.add_argument("--year", default="2008")
    parser.add_argument(
        "--data-url",
        default=None,
        help="Optional CSV URL for smoke tests. Defaults to the pinned public Penguins CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = PenguinSample(
        island=args.island,
        bill_length_mm=args.bill_length_mm,
        bill_depth_mm=args.bill_depth_mm,
        flipper_length_mm=args.flipper_length_mm,
        body_mass_g=args.body_mass_g,
        sex=args.sex,
        year=str(args.year),
    )
    data = load_penguin_data(args.data_url, min_bytes=100) if args.data_url else None
    model = fit_penguin_model(data)
    prediction = predict_penguin_sample(model, sample)
    explanation = explain_penguin_prediction(model, sample, top_n=5)

    print(f"Predicted species: {prediction.predicted_species}")
    print("Class probabilities:")
    for species, probability in sorted(prediction.probabilities.items()):
        print(f"  {species}: {probability:.3f}")
    print()
    print("Top model-internal logit contributions:")
    for row in explanation:
        print(f"  {row['feature']}: {row['direction']} ({row['contribution']:+.3f})")
    print()
    print("Boundary: educational public-data demo; not a field identification system.")


if __name__ == "__main__":
    main()
