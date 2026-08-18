import {
  explainPrediction,
  predictPenguin,
  validateModelArtifact,
} from "./model.mjs";

const MODEL_URL = "./model/penguins-logistic-v1.json";
const numberFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 });
const percentFormat = new Intl.NumberFormat("en-US", {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

function controlId(name) {
  return `input-${name.replaceAll("_", "-")}`;
}

function appendText(parent, tagName, text, className) {
  const element = document.createElement(tagName);
  element.textContent = text;
  if (className) element.className = className;
  parent.append(element);
  return element;
}

function configureControls(model) {
  for (const feature of model.inputs.numeric) {
    const input = document.querySelector(`#${controlId(feature.name)}`);
    const output = document.querySelector(`#${controlId(feature.name)}-value`);
    const range = document.querySelector(`#${controlId(feature.name)}-range`);
    input.min = String(feature.control_min);
    input.max = String(feature.control_max);
    input.step = String(feature.step);
    input.value = String(feature.default);
    output.value = numberFormat.format(feature.default);
    output.textContent = numberFormat.format(feature.default);
    range.textContent =
      `Observed: ${numberFormat.format(feature.observed_min)} to ` +
      `${numberFormat.format(feature.observed_max)} ${feature.unit}`;
    input.addEventListener("input", () => {
      output.value = numberFormat.format(Number(input.value));
      output.textContent = numberFormat.format(Number(input.value));
    });
  }

  for (const feature of model.inputs.categorical) {
    const select = document.querySelector(`#${controlId(feature.name)}`);
    select.replaceChildren();
    for (const optionValue of feature.options) {
      const option = document.createElement("option");
      option.value = optionValue;
      option.textContent = optionValue[0].toUpperCase() + optionValue.slice(1);
      option.selected = optionValue === feature.default;
      select.append(option);
    }
  }
}

function readInput(form) {
  return Object.fromEntries(
    [...form.elements]
      .filter((element) => element.name)
      .map((element) => [
        element.name,
        element.dataset.kind === "numeric" ? Number(element.value) : element.value,
      ]),
  );
}

function renderProbabilities(model, prediction) {
  const container = document.querySelector("#probabilities");
  container.replaceChildren();
  for (const species of model.classes) {
    const probability = prediction.probabilities[species];
    const row = document.createElement("div");
    row.className = "probability-row";
    const header = document.createElement("div");
    header.className = "bar-label";
    appendText(header, "span", species);
    appendText(header, "span", percentFormat.format(probability), "bar-value");
    const track = document.createElement("div");
    track.className = "bar-track";
    track.setAttribute("role", "meter");
    track.setAttribute("aria-label", `${species} probability`);
    track.setAttribute("aria-valuemin", "0");
    track.setAttribute("aria-valuemax", "100");
    track.setAttribute("aria-valuenow", String(probability * 100));
    const fill = document.createElement("div");
    fill.className = `bar-fill species-${species.toLowerCase()}`;
    fill.style.setProperty("--bar-width", `${probability * 100}%`);
    track.append(fill);
    row.append(header, track);
    container.append(row);
  }
}

function renderWarnings(warnings) {
  const container = document.querySelector("#range-warnings");
  container.replaceChildren();
  container.hidden = warnings.length === 0;
  if (warnings.length === 0) return;
  appendText(container, "strong", "Outside the observed training range");
  const list = document.createElement("ul");
  for (const warning of warnings) appendText(list, "li", warning);
  container.append(list);
}

function readableFeature(feature) {
  return feature
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function renderExplanation(explanation) {
  const container = document.querySelector("#contributions");
  container.replaceChildren();
  const maximum = Math.max(...explanation.map((item) => Math.abs(item.contribution)), 0.001);
  for (const item of explanation) {
    const row = document.createElement("div");
    row.className = "contribution-row";
    const header = document.createElement("div");
    header.className = "bar-label";
    appendText(header, "span", readableFeature(item.feature));
    appendText(
      header,
      "span",
      `${item.direction} ${item.contribution >= 0 ? "+" : ""}${item.contribution.toFixed(3)}`,
      `bar-value direction-${item.contribution >= 0 ? "positive" : "negative"}`,
    );
    const track = document.createElement("div");
    track.className = "contribution-track";
    const fill = document.createElement("div");
    fill.className = `contribution-fill ${item.contribution >= 0 ? "positive" : "negative"}`;
    fill.style.setProperty("--contribution-width", `${(Math.abs(item.contribution) / maximum) * 50}%`);
    track.append(fill);
    row.append(header, track);
    container.append(row);
  }
}

function render(model, form, announce = false) {
  const input = readInput(form);
  const prediction = predictPenguin(model, input);
  const explanation = explainPrediction(model, input, prediction.predicted_species, 5);
  document.querySelector("#predicted-species").textContent = prediction.predicted_species;
  renderProbabilities(model, prediction);
  renderWarnings(prediction.warnings);
  renderExplanation(explanation);
  if (announce) {
    document.querySelector("#prediction-status").textContent =
      `Prediction updated to ${prediction.predicted_species}. ` +
      `${prediction.warnings.length} range warnings.`;
  }
}

async function start() {
  const response = await fetch(MODEL_URL, { cache: "no-store" });
  if (!response.ok) throw new Error(`Model artifact request failed with ${response.status}.`);
  const model = await response.json();
  validateModelArtifact(model);
  const form = document.querySelector("#penguin-inputs");
  configureControls(model);
  form.addEventListener("input", () => render(model, form, false));
  form.addEventListener("change", () => render(model, form, true));
  render(model, form, false);
  document.querySelector("#model-id").textContent = model.artifact_id;
  document.querySelector("#training-rows").textContent = String(model.training_rows);
}

start().catch((error) => {
  document.querySelector("#app-error").hidden = false;
  document.querySelector("#app-error-detail").textContent = error.message;
});
