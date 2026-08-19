"""Streamlit demo for the Palmer Penguins classifier."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from ml_portfolio.penguins import (
    DATA_URL,
    PenguinSample,
    categorical_options,
    default_penguin_sample,
    explain_penguin_prediction,
    feature_ranges,
    fit_penguin_model,
    load_penguin_data,
    predict_penguin_sample,
)


ACCENT = "#2F6F73"
HIGHLIGHT = "#C2703D"
REPO_URL = "https://github.com/MahmmodAbuhani/applied-ml-notebooks"
NOTEBOOK_URL = f"{REPO_URL}/blob/main/notebooks/palmer_penguins_end_to_end.ipynb"
MODEL_CARD_URL = f"{REPO_URL}/blob/main/reports/palmer_penguins_model_card.md"
STATIC_EXPLORER_URL = f"{REPO_URL}/blob/main/site/index.html"
NUMERIC_LABELS = {
    "bill_length_mm": "Bill length (mm)",
    "bill_depth_mm": "Bill depth (mm)",
    "flipper_length_mm": "Flipper length (mm)",
    "body_mass_g": "Body mass (g)",
}


@st.cache_data(show_spinner=False)
def cached_penguin_data() -> pd.DataFrame:
    """Load the pinned public CSV once per Streamlit session."""

    return load_penguin_data()


@st.cache_resource(show_spinner=False)
def cached_penguin_model(data: pd.DataFrame):
    """Train the demo model from the public CSV."""

    return fit_penguin_model(data)


def _probability_table(probabilities: dict[str, float]) -> pd.DataFrame:
    rows = [
        {"species": species, "probability": probability}
        for species, probability in sorted(
            probabilities.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]
    return pd.DataFrame(rows)


def _probability_summary(probabilities: dict[str, float]) -> str:
    """Return a short text alternative for the probability chart."""

    values = _probability_table(probabilities)
    readout = "; ".join(
        f"{row.species} {row.probability:.1%}"
        for row in values.itertuples(index=False)
    )
    return f"Model probabilities, not calibrated field confidence: {readout}."


def _probability_chart(probabilities: dict[str, float]) -> alt.Chart:
    probability_table = _probability_table(probabilities)
    return (
        alt.Chart(probability_table)
        .mark_bar(color=ACCENT)
        .encode(
            x=alt.X(
                "probability:Q",
                title="Probability",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            y=alt.Y("species:N", title="Species", sort="-x"),
            tooltip=[
                alt.Tooltip("species:N", title="Species"),
                alt.Tooltip("probability:Q", title="Probability", format=".1%"),
            ],
        )
        .properties(height=130)
    )


def _finite_domain(values: object) -> list[float]:
    numeric_values = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric_values.empty:
        return [0.0, 1.0]

    lower = float(numeric_values.min())
    upper = float(numeric_values.max())
    span = upper - lower
    padding = span * 0.1 if span else max(abs(lower) * 0.1, 1.0)
    return [round(lower - padding, 3), round(upper + padding, 3)]


def _training_distribution_summary(sample: PenguinSample) -> str:
    """Return a short text alternative for the training-context chart."""

    return (
        "Training-context plot: colored points show observed penguins; the diamond "
        "marks the current input at "
        f"{sample.bill_length_mm:.1f} mm bill length and "
        f"{sample.flipper_length_mm:.0f} mm flipper length."
    )


def _training_distribution_chart(data: pd.DataFrame, sample: PenguinSample) -> alt.LayerChart:
    chart_data = data[["species", "bill_length_mm", "flipper_length_mm"]].dropna().copy()
    sample_data = pd.DataFrame(
        [
            {
                "label": "Current input",
                "bill_length_mm": sample.bill_length_mm,
                "flipper_length_mm": sample.flipper_length_mm,
            }
        ]
    )
    bill_domain = _finite_domain(
        [*chart_data["bill_length_mm"].tolist(), sample.bill_length_mm]
    )
    flipper_domain = _finite_domain(
        [*chart_data["flipper_length_mm"].tolist(), sample.flipper_length_mm]
    )

    observed_points = (
        alt.Chart(chart_data)
        .mark_circle(size=62, opacity=0.62)
        .encode(
            x=alt.X(
                "bill_length_mm:Q",
                title="Bill length (mm)",
                scale=alt.Scale(domain=bill_domain),
            ),
            y=alt.Y(
                "flipper_length_mm:Q",
                title="Flipper length (mm)",
                scale=alt.Scale(domain=flipper_domain),
            ),
            color=alt.Color("species:N", title="Observed species"),
            tooltip=[
                alt.Tooltip("species:N", title="Species"),
                alt.Tooltip("bill_length_mm:Q", title="Bill length", format=".1f"),
                alt.Tooltip("flipper_length_mm:Q", title="Flipper length", format=".0f"),
            ],
        )
    )
    current_input = (
        alt.Chart(sample_data)
        .mark_point(
            shape="diamond",
            size=260,
            filled=True,
            color=HIGHLIGHT,
            stroke="white",
            strokeWidth=1.5,
        )
        .encode(
            x=alt.X(
                "bill_length_mm:Q",
                title="Bill length (mm)",
                scale=alt.Scale(domain=bill_domain),
            ),
            y=alt.Y(
                "flipper_length_mm:Q",
                title="Flipper length (mm)",
                scale=alt.Scale(domain=flipper_domain),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Point"),
                alt.Tooltip("bill_length_mm:Q", title="Bill length", format=".1f"),
                alt.Tooltip("flipper_length_mm:Q", title="Flipper length", format=".0f"),
            ],
        )
    )
    return (observed_points + current_input).properties(height=280)


def _contribution_chart(explanation: list[dict[str, object]]) -> alt.Chart:
    contribution_table = pd.DataFrame(explanation).copy()
    contribution_domain = _finite_domain(contribution_table["contribution"])
    return (
        alt.Chart(contribution_table)
        .mark_bar()
        .encode(
            x=alt.X(
                "contribution:Q",
                title="Logit contribution",
                scale=alt.Scale(domain=contribution_domain),
            ),
            y=alt.Y(
                "feature:N",
                title="Feature",
                sort=alt.EncodingSortField(field="abs_contribution", order="descending"),
            ),
            color=alt.Color(
                "direction:N",
                title="Direction",
                scale=alt.Scale(
                    domain=["supports", "pulls against"],
                    range=[ACCENT, HIGHLIGHT],
                ),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("direction:N", title="Direction"),
                alt.Tooltip("contribution:Q", title="Contribution", format="+.3f"),
            ],
        )
        .properties(height=210)
    )


def _contribution_summary(explanation: list[dict[str, object]]) -> str:
    """Return a short text alternative for the contribution chart."""

    if not explanation:
        return "No model-internal contributions are available."

    top = max(explanation, key=lambda row: abs(float(row["contribution"])))
    feature = str(top["feature"])
    direction = str(top["direction"])
    contribution = float(top["contribution"])
    return (
        f"Largest model-internal contribution: {feature} {direction} the prediction "
        f"({contribution:+.3f} logit). This is not a causal explanation."
    )


def _out_of_distribution_messages(
    sample: PenguinSample,
    ranges: dict[str, tuple[float, float, float]],
) -> list[str]:
    messages = []
    for feature, (observed_min, _, observed_max) in ranges.items():
        value = float(getattr(sample, feature))
        if value < observed_min or value > observed_max:
            label = NUMERIC_LABELS[feature]
            messages.append(
                f"{label} is outside the observed range: "
                f"{observed_min:.1f} to {observed_max:.1f}."
            )
    return messages


def _slider_bounds(observed_min: float, observed_max: float) -> tuple[float, float]:
    span = observed_max - observed_min
    padding = span * 0.12
    return observed_min - padding, observed_max + padding


def main() -> None:
    st.set_page_config(
        page_title="Penguin Species Demo",
        page_icon=":material/science:",
        layout="wide",
    )

    st.title("Penguin Species Demo")
    st.caption(
        "Educational public-data demo. Hosted on Streamlit Community Cloud; "
        "not a production decision system."
    )

    data = cached_penguin_data()
    model = cached_penguin_model(data)
    ranges = feature_ranges(data)
    options = categorical_options(data)
    defaults = default_penguin_sample()

    st.markdown(
        """
        The Palmer Penguins dataset is small, curated, and easy to audit.

        This app shows inputs, model probabilities, feature contributions, and limits.

        Contributions describe the fitted model's calculation. They are not biological
        causes or calibrated field confidence.
        """
    )

    controls, results = st.columns([0.42, 0.58], gap="large")

    with controls:
        st.subheader("Penguin Inputs")
        st.caption("Morphology values are numeric; island, sex, and year are collection context.")

        island = st.selectbox(
            "Island",
            options["island"],
            index=options["island"].index(defaults.island),
        )
        sex = st.selectbox(
            "Sex",
            options["sex"],
            index=options["sex"].index(defaults.sex),
        )
        year = st.selectbox(
            "Year",
            options["year"],
            index=options["year"].index(defaults.year),
        )

        bill_length = st.slider(
            "Bill length (mm)",
            min_value=_slider_bounds(ranges["bill_length_mm"][0], ranges["bill_length_mm"][2])[0],
            max_value=_slider_bounds(ranges["bill_length_mm"][0], ranges["bill_length_mm"][2])[1],
            value=defaults.bill_length_mm,
            step=0.1,
        )
        bill_depth = st.slider(
            "Bill depth (mm)",
            min_value=_slider_bounds(ranges["bill_depth_mm"][0], ranges["bill_depth_mm"][2])[0],
            max_value=_slider_bounds(ranges["bill_depth_mm"][0], ranges["bill_depth_mm"][2])[1],
            value=defaults.bill_depth_mm,
            step=0.1,
        )
        flipper_length = st.slider(
            "Flipper length (mm)",
            min_value=_slider_bounds(
                ranges["flipper_length_mm"][0],
                ranges["flipper_length_mm"][2],
            )[0],
            max_value=_slider_bounds(
                ranges["flipper_length_mm"][0],
                ranges["flipper_length_mm"][2],
            )[1],
            value=defaults.flipper_length_mm,
            step=1.0,
        )
        body_mass = st.slider(
            "Body mass (g)",
            min_value=_slider_bounds(ranges["body_mass_g"][0], ranges["body_mass_g"][2])[0],
            max_value=_slider_bounds(ranges["body_mass_g"][0], ranges["body_mass_g"][2])[1],
            value=defaults.body_mass_g,
            step=50.0,
        )

    sample = PenguinSample(
        island=island,
        bill_length_mm=bill_length,
        bill_depth_mm=bill_depth,
        flipper_length_mm=flipper_length,
        body_mass_g=body_mass,
        sex=sex,
        year=year,
    )
    prediction = predict_penguin_sample(model, sample)
    explanation = explain_penguin_prediction(model, sample, top_n=6)
    ood_messages = _out_of_distribution_messages(sample, ranges)

    with results:
        st.subheader("Prediction")
        st.metric("Predicted species", prediction.predicted_species)
        for message in ood_messages:
            st.warning(message)

        st.caption(_probability_summary(prediction.probabilities))
        st.altair_chart(_probability_chart(prediction.probabilities), width="stretch")
        st.caption(_training_distribution_summary(sample))
        st.altair_chart(_training_distribution_chart(data, sample), width="stretch")

        st.subheader("Model-Internal Explanation")
        st.caption(
            "Top contributions to the predicted species logit. These are not causal "
            "claims about biology or collection conditions."
        )
        st.caption(_contribution_summary(explanation))
        st.altair_chart(_contribution_chart(explanation), width="stretch")

    st.subheader("Data And Model Boundary")
    st.markdown(
        f"""
        - Source: [pinned Palmer Penguins CSV]({DATA_URL}).
        - Training: the model is fit from public data when the app starts.
        - App code: this repository does not persist inputs, write them to a
          database, or send them to a model API.
        - Hosting: this public app runs on Streamlit Community Cloud. It may wake on
          demand and remains an educational interface, not a production service.
        - Scope: educational classifier demo, not a
          wildlife field tool or production service.
        - Review path: read the [Penguins notebook]({NOTEBOOK_URL}) for
          holdout metrics, repeated training-only checks, feature ablation, and
          error review.
        - Further reading: [model card]({MODEL_CARD_URL}), [repository]({REPO_URL}),
          and [static explorer source]({STATIC_EXPLORER_URL}).
        """
    )


if __name__ == "__main__":
    main()
