import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const root = new URL("../../", import.meta.url);
const styles = await readFile(new URL("site/styles.css", root), "utf8");

test("responsive grid children can shrink inside narrow viewports", () => {
  assert.match(
    styles,
    /\.hero-grid\s*>\s*\*,\s*\.section-heading\s*>\s*\*,\s*\.explorer-grid\s*>\s*\*\s*\{[^}]*min-width:\s*0;/s,
  );
});

test("the hero heading wraps within the mobile content box", () => {
  assert.match(styles, /h1\s*\{[^}]*overflow-wrap:\s*anywhere;/s);
  assert.match(styles, /@media\s*\(max-width:\s*600px\)[\s\S]*?h1\s*\{[^}]*max-width:\s*100%;/s);
});

test("mobile grids use a zero-minimum track", () => {
  assert.match(
    styles,
    /@media\s*\(max-width:\s*900px\)[\s\S]*?\.hero-grid, \.section-heading, \.explorer-grid\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\);/s,
  );
});
