const EXPECTED_NUMERIC_FEATURES = [
  "bill_length_mm",
  "bill_depth_mm",
  "flipper_length_mm",
  "body_mass_g",
];
const EXPECTED_CATEGORICAL_FEATURES = ["island", "sex", "year"];

function assertFiniteArray(values, label) {
  if (!Array.isArray(values) || values.some((value) => !Number.isFinite(value))) {
    throw new TypeError(`${label} must contain only finite numbers.`);
  }
}

export function validateModelArtifact(model) {
  if (!model || typeof model !== "object") {
    throw new TypeError("Model artifact must be an object.");
  }
  if (model.schema_version !== 1 || model.artifact_id !== "penguins-logistic-v1") {
    throw new TypeError("Unsupported model artifact identity or schema.");
  }
  if (!Array.isArray(model.classes) || model.classes.length !== 3) {
    throw new TypeError("Model artifact must define three classes.");
  }
  if (
    JSON.stringify(model.preprocessing?.numeric_features) !==
      JSON.stringify(EXPECTED_NUMERIC_FEATURES) ||
    JSON.stringify(model.preprocessing?.categorical_features) !==
      JSON.stringify(EXPECTED_CATEGORICAL_FEATURES)
  ) {
    throw new TypeError("Model artifact feature order is unsupported.");
  }

  const transformedCount = model.transformed_features?.length;
  if (!Number.isInteger(transformedCount) || transformedCount < 7) {
    throw new TypeError("Model artifact transformed features are incomplete.");
  }
  assertFiniteArray(model.preprocessing.imputer_statistics, "Imputer statistics");
  assertFiniteArray(model.preprocessing.scaler_mean, "Scaler means");
  assertFiniteArray(model.preprocessing.scaler_scale, "Scaler scales");
  assertFiniteArray(model.classifier?.intercepts, "Classifier intercepts");

  if (
    !Array.isArray(model.classifier?.coefficients) ||
    model.classifier.coefficients.length !== model.classes.length
  ) {
    throw new TypeError("Classifier coefficient rows must match the three classes.");
  }
  model.classifier.coefficients.forEach((row, index) => {
    assertFiniteArray(row, `Classifier coefficient row ${index}`);
    if (row.length !== transformedCount) {
      throw new TypeError("Classifier coefficient width must match transformed features.");
    }
  });
  return true;
}

function validateInput(model, input) {
  for (const feature of model.inputs.numeric) {
    if (!Number.isFinite(input[feature.name])) {
      throw new TypeError(`${feature.label} must be a finite number.`);
    }
  }
  for (const feature of model.inputs.categorical) {
    if (!feature.options.includes(String(input[feature.name]))) {
      throw new TypeError(`${feature.label} is not a supported category.`);
    }
  }
}

function transformInput(model, input) {
  const transformed = [];
  const preprocessing = model.preprocessing;

  preprocessing.numeric_features.forEach((name, index) => {
    const value = Number.isFinite(input[name])
      ? input[name]
      : preprocessing.imputer_statistics[index];
    transformed.push(
      (value - preprocessing.scaler_mean[index]) / preprocessing.scaler_scale[index],
    );
  });

  preprocessing.categorical_features.forEach((name, featureIndex) => {
    const selected = String(input[name]);
    preprocessing.categorical_options[featureIndex].forEach((category) => {
      transformed.push(selected === category ? 1 : 0);
    });
  });
  return transformed;
}

function stableSoftmax(logits) {
  const maximum = Math.max(...logits);
  const exponents = logits.map((value) => Math.exp(value - maximum));
  const denominator = exponents.reduce((sum, value) => sum + value, 0);
  return exponents.map((value) => value / denominator);
}

function rangeWarnings(model, input) {
  const warnings = [];
  for (const feature of model.inputs.numeric) {
    const value = input[feature.name];
    if (value < feature.observed_min || value > feature.observed_max) {
      warnings.push(
        `${feature.label} (${feature.unit}) is outside the observed training range of ` +
          `${feature.observed_min.toFixed(1)} to ${feature.observed_max.toFixed(1)}.`,
      );
    }
  }
  return warnings;
}

export function predictPenguin(model, input) {
  validateModelArtifact(model);
  validateInput(model, input);
  const transformed = transformInput(model, input);
  const logits = model.classifier.coefficients.map((coefficients, classIndex) =>
    coefficients.reduce(
      (sum, coefficient, featureIndex) =>
        sum + coefficient * transformed[featureIndex],
      model.classifier.intercepts[classIndex],
    ),
  );
  const probabilityValues = stableSoftmax(logits);
  const winnerIndex = probabilityValues.reduce(
    (best, value, index, values) => (value > values[best] ? index : best),
    0,
  );
  return {
    predicted_species: model.classes[winnerIndex],
    probabilities: Object.fromEntries(
      model.classes.map((species, index) => [species, probabilityValues[index]]),
    ),
    logits,
    warnings: rangeWarnings(model, input),
  };
}

export function explainPrediction(model, input, predictedSpecies, limit = 5) {
  validateModelArtifact(model);
  validateInput(model, input);
  const classIndex = model.classes.indexOf(predictedSpecies);
  if (classIndex < 0) {
    throw new TypeError("Predicted species is not present in the model artifact.");
  }
  const transformed = transformInput(model, input);
  return transformed
    .map((value, index) => {
      const contribution = value * model.classifier.coefficients[classIndex][index];
      return {
        index,
        feature: model.transformed_features[index]
          .replace("numeric__", "")
          .replace("categorical__", ""),
        contribution,
        direction: contribution >= 0 ? "supports" : "pulls against",
      };
    })
    .sort((left, right) => {
      const magnitudeDifference =
        Math.abs(right.contribution) - Math.abs(left.contribution);
      return magnitudeDifference === 0 ? left.index - right.index : magnitudeDifference;
    })
    .slice(0, limit)
    .map(({ index: _index, ...row }) => row);
}
