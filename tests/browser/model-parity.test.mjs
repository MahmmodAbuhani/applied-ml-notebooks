import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  explainPrediction,
  predictPenguin,
  validateModelArtifact,
} from "../../site/js/model.mjs";

const root = new URL("../../", import.meta.url);
const model = JSON.parse(
  await readFile(new URL("site/model/penguins-logistic-v1.json", root), "utf8"),
);
const fixtures = JSON.parse(
  await readFile(
    new URL("tests/browser/fixtures/penguins_reference_cases.json", root),
    "utf8",
  ),
);

function assertClose(actual, expected, tolerance, label) {
  assert.ok(
    Math.abs(actual - expected) <= tolerance,
    `${label}: expected ${expected}, received ${actual}`,
  );
}

test("the committed browser artifact satisfies its runtime schema", () => {
  assert.doesNotThrow(() => validateModelArtifact(model));
  assert.equal(model.schema_version, 1);
  assert.equal(model.artifact_id, "penguins-logistic-v1");
  assert.deepEqual(model.classes, ["Adelie", "Chinstrap", "Gentoo"]);
});

test("browser inference matches Python reference outputs across fixtures and boundaries", () => {
  assert.ok(fixtures.cases.length >= 42, "expected representative and boundary fixtures");

  for (const fixture of fixtures.cases) {
    const actual = predictPenguin(model, fixture.input);
    assert.equal(actual.predicted_species, fixture.expected.predicted_species, fixture.id);
    assert.deepEqual(actual.warnings, fixture.expected.warnings, `${fixture.id}: warnings`);

    for (const species of model.classes) {
      assertClose(
        actual.probabilities[species],
        fixture.expected.probabilities[species],
        1e-10,
        `${fixture.id}: ${species} probability`,
      );
    }
    actual.logits.forEach((value, index) => {
      assertClose(value, fixture.expected.logits[index], 1e-10, `${fixture.id}: logit ${index}`);
    });
    assertClose(
      Object.values(actual.probabilities).reduce((sum, value) => sum + value, 0),
      1,
      1e-12,
      `${fixture.id}: probability sum`,
    );
  }
});

test("the explanation is deterministic, limited, signed, and tied to the predicted class", () => {
  const fixture = fixtures.cases.find(({ id }) => id === "default");
  assert.ok(fixture, "default fixture must exist");

  const prediction = predictPenguin(model, fixture.input);
  const explanation = explainPrediction(model, fixture.input, prediction.predicted_species, 5);

  assert.equal(explanation.length, 5);
  assert.deepEqual(explanation, fixture.expected.top_contributions);
  for (const item of explanation) {
    assert.ok(Number.isFinite(item.contribution));
    assert.equal(
      item.direction,
      item.contribution >= 0 ? "supports" : "pulls against",
    );
  }
});

test("malformed artifacts and non-finite input fail closed", () => {
  assert.throws(
    () => validateModelArtifact({ ...model, classes: ["Adelie"] }),
    /three classes/i,
  );
  assert.throws(
    () => predictPenguin(model, { ...fixtures.cases[0].input, body_mass_g: Infinity }),
    /finite number/i,
  );
});
