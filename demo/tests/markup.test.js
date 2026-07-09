"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const demoRoot = path.resolve(__dirname, "..");
const markup = fs.readFileSync(path.join(demoRoot, "index.html"), "utf8");
const app = fs.readFileSync(path.join(demoRoot, "app.js"), "utf8");

test("provenance markup has unique, script-addressable fields", () => {
  const ids = [...markup.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]);
  assert.equal(new Set(ids).size, ids.length, "index.html contains duplicate IDs");
  for (const id of [
    "provenance-panel",
    "acceptance-badge",
    "provenance-model-revision",
    "provenance-lens-sha",
    "provenance-corpus-sha",
    "provenance-prompt-count",
    "provenance-target-layer",
    "provenance-source-layers",
    "provenance-acceptance-tier",
    "provenance-acceptance-status",
    "acceptance-reasons",
    "acceptance-checks",
  ]) {
    assert.ok(ids.includes(id), `missing #${id}`);
    assert.match(app, new RegExp(`"${id}"`), `app.js does not bind #${id}`);
  }
});

test("provenance policy loads before the explorer and smoke is never auto-loaded", () => {
  const provenanceIndex = markup.indexOf('src="provenance.js"');
  const appIndex = markup.indexOf('src="app.js"');
  assert.ok(provenanceIndex >= 0 && provenanceIndex < appIndex);
  assert.match(app, /\.\/fixtures\/nemotron-3-nano\.recorded\.json/);
  assert.doesNotMatch(app, /smoke-recorded\.json/);
});

test("the accepted pilot has an explicit opt-in URL but cannot become the default", () => {
  assert.match(app, /\.\/fixtures\/nemotron-3-nano\.pilot-recorded\.json/);
  assert.match(app, /URLSearchParams\(globalThis\.location\.search\)/);
  assert.match(app, /requested !== "pilot"/);
  assert.match(app, /summary\.finalDefaultEligible/);
  assert.match(app, /example !== "modulation-topic"/);
});
