"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const {
  aggregateReadoutRows,
  buildInterventionPayload,
  conditionViewModel,
  parseLayerList,
  progressPercent,
  readoutEntries,
  selectedPositions,
  visibleWhitespace,
} = require("../app.js");

test("API sequence tokens preserve leading whitespace and numeric IDs", () => {
  const apiCondition = {
    name: "clean",
    completion: " fish",
    tokens: [
      { position: 0, id: 1, text: "<s>", is_generated: false, is_bos: true },
      { position: 1, id: 48291, text: " fish", is_generated: true, is_bos: false },
    ],
    readouts: { layers: [34], jacobian: {}, logit: {} },
  };

  const view = conditionViewModel(apiCondition);
  assert.equal(view.tokens[1].text, " fish");
  assert.equal(view.tokens[1].id, 48291);
  assert.equal(typeof view.tokens[1].id, "number");
  assert.equal(visibleWhitespace(view.tokens[1].text), "␠fish");
  assert.equal(apiCondition.tokens[1].text, " fish", "render labels must not mutate API text");
});

test("API readout entries preserve exact decoded pieces and numeric IDs", () => {
  const condition = {
    readouts: {
      jacobian: {
        "34": [
          [{ id: 70001, text: "\nanswer", probability: 0.4, logit: 12.5 }],
          [{ id: 192, text: " coral", probability: 0.25, logit: 8.75 }],
        ],
      },
    },
  };

  const entries = readoutEntries(condition, "jacobian", 34, 1);
  assert.deepEqual(entries[0], {
    id: 192,
    text: " coral",
    probability: 0.25,
    logit: 8.75,
  });
  assert.equal(typeof entries[0].id, "number");
  assert.equal(visibleWhitespace(entries[0].text), "␠coral");
});

test("Unicode and zero-width whitespace readouts remain visibly identifiable", () => {
  assert.equal(visibleWhitespace("\u202f"), "⍽");
  assert.equal(visibleWhitespace("\u00a0answer"), "⍽answer");
  assert.equal(visibleWhitespace("\u200b"), "⟨ZWSP⟩");
  assert.equal(visibleWhitespace("\u200d"), "⟨ZWJ⟩");
});

test("position scopes derive all, prompt, generated, and deduplicated custom positions", () => {
  const condition = conditionViewModel({
    tokens: [
      { position: 0, id: 1, text: "<s>", is_generated: false, is_bos: true },
      { position: 1, id: 2, text: " prompt", is_generated: false, is_bos: false },
      { position: 2, id: 3, text: " answer", is_generated: true, is_bos: false },
    ],
  });

  assert.deepEqual(selectedPositions(condition, { mode: "all" }), [0, 1, 2]);
  assert.deepEqual(selectedPositions(condition, { mode: "prompt" }), [0, 1]);
  assert.deepEqual(selectedPositions(condition, { mode: "generated" }), [2]);
  assert.deepEqual(selectedPositions(condition, { mode: "custom", custom: [2, 1, 2, 999] }), [1, 2]);
});

test("aggregate readout rows count every selected position and layer without overwriting", () => {
  const condition = {
    readouts: {
      jacobian: {
        "26": [
          [{ id: 70001, text: " coral", probability: 0.1, logit: 4 }],
          [{ id: 70001, text: " coral", probability: 0.4, logit: 7 }],
        ],
        "34": [
          [
            { id: 80002, text: " reef", probability: 0.3, logit: 6 },
            { id: 70001, text: " coral", probability: 0.2, logit: 5 },
          ],
          [{ id: 70001, text: " coral", probability: 0.25, logit: 5.5 }],
        ],
        "40": [[], []],
      },
    },
  };

  const rows = aggregateReadoutRows(condition, "jacobian", [26, 34, 40], [0, 1, 1]);
  const coral = rows.find((row) => row.id === 70001);

  assert.ok(coral);
  assert.equal(typeof coral.id, "number");
  assert.equal(coral.text, " coral");
  assert.equal(coral.occurrenceCount, 4);
  assert.equal(coral.positionCount, 2);
  assert.equal(coral.layerCount, 2);
  assert.deepEqual(coral.cells.map((cell) => cell.count), [2, 2, 0]);
  assert.deepEqual([...coral.positionCounts.entries()], [[0, 2], [1, 2]]);
  assert.equal(coral.cells[0].bestOccurrence.position, 1);
  assert.equal(coral.anchor.layer, 26, "equal layer counts choose the lowest writable layer");
  assert.equal(coral.anchor.position, 1, "the source occurrence retains its exact position");
});

test("aggregate readout rows preserve leading whitespace and keep equal text with different IDs separate", () => {
  const condition = {
    readouts: {
      logit: {
        "26": [[
          { id: 11, text: " Spider", probability: 0.4, logit: 8 },
          { id: 12, text: " Spider", probability: 0.3, logit: 7 },
        ]],
      },
    },
  };

  const rows = aggregateReadoutRows(condition, "logit", [26], 0);

  assert.equal(rows.length, 2);
  assert.deepEqual(rows.map((row) => row.id), [11, 12]);
  assert.deepEqual(rows.map((row) => row.text), [" Spider", " Spider"]);
  assert.equal(condition.readouts.logit["26"][0][0].text, " Spider");
});

test("aggregate readout rows sort deterministically by occurrence count then numeric ID", () => {
  const condition = {
    readouts: {
      jacobian: {
        "26": [[
          { id: 40, text: "forty", probability: 0.1, logit: 1 },
          { id: 30, text: "thirty", probability: 0.25, logit: 3 },
          { id: 20, text: "twenty", probability: 0.25, logit: 3 },
          { id: 10, text: "ten", probability: 0.99, logit: 9 },
        ]],
        "34": [[
          { id: 40, text: "forty", probability: 0.1, logit: 1 },
          { id: 30, text: "thirty", probability: 0.25, logit: 3 },
          { id: 20, text: "twenty", probability: 0.25, logit: 3 },
        ]],
      },
    },
  };

  const rows = aggregateReadoutRows(condition, "jacobian", [26, 34], 0);

  assert.deepEqual(
    rows.map((row) => row.id),
    [20, 30, 40, 10],
    "raw top-k occurrence count wins first and numeric ID breaks ties",
  );
});

test("aggregate readout rows retain original entry indexes and prefer a writable anchor over layer 51", () => {
  const condition = {
    readouts: {
      logit: {
        "51": [[
          { id: 999, text: "other", probability: 0.01, logit: 1 },
          { id: 42010, text: " Spider", probability: 0.9, logit: 12 },
        ]],
        "50": [[
          { id: 111, text: "first", probability: 0.2, logit: 3 },
          { id: 222, text: "second", probability: 0.15, logit: 2 },
          { id: 42010, text: " Spider", probability: 0.1, logit: 1.5 },
        ]],
      },
    },
  };

  const rows = aggregateReadoutRows(condition, "logit", [51, 50], 0);
  const spider = rows.find((row) => row.id === 42010);

  assert.ok(spider);
  assert.equal(spider.cells[0].bestOccurrence.entryIndex, 1);
  assert.equal(spider.cells[1].bestOccurrence.entryIndex, 2);
  assert.equal(spider.anchor.layer, 50);
  assert.equal(spider.anchor.position, 0);
  assert.equal(spider.anchor.entryIndex, 2);
  assert.equal(spider.anchor.entry.id, 42010);
});

test("swap payload uses typed source identity instead of the clicked readout", () => {
  const base = {
    prompt: "Discuss reefs.",
    layers: [26, 34, 40],
    top_k: 8,
    max_new_tokens: 8,
    sampling: false,
    temperature: 1,
    top_p: 1,
    seed: 0,
  };
  const selected = {
    lensType: "jacobian",
    entry: { id: 192, text: " coral" },
  };
  const payload = buildInterventionPayload(base, selected, {
    mode: "swap",
    layers: "26,34,40",
    strength: "-0.1",
    applyToGenerated: false,
    source: {
      text: " Spider",
      token_ids: [42010],
      is_single_token: true,
      pieces: [{ id: 42010, text: " Spider" }],
    },
    target: {
      text: " fish",
      token_ids: [15591],
      is_single_token: true,
      pieces: [{ id: 15591, text: " fish" }],
    },
  });

  assert.deepEqual(payload.intervention.source_token_ids, [42010]);
  assert.equal(typeof payload.intervention.source_token_ids[0], "number");
  assert.deepEqual(payload.source_token_texts, [" Spider"]);
  assert.equal(payload.intervention.target_token_id, 15591);
  assert.equal(typeof payload.intervention.target_token_id, "number");
  assert.equal(payload.target_token_text, " fish");
  assert.equal(payload.prompt, base.prompt);
  assert.deepEqual(selected.entry, { id: 192, text: " coral" }, "payload construction must not mutate the clicked readout");
});

for (const { text, id } of [
  { text: "Spider", id: 84369 },
  { text: " Spider", id: 42010 },
]) {
  test(`swap preserves exact typed source ${JSON.stringify(text)} and ID ${id}`, () => {
    const payload = buildInterventionPayload(
      { prompt: "Discuss reefs.", layers: [26] },
      { lensType: "jacobian", entry: { id: 192, text: " coral" } },
      {
        mode: "swap",
        layers: "26",
        strength: -0.1,
        applyToGenerated: false,
        source: {
          text,
          token_ids: [id],
          is_single_token: true,
          pieces: [{ id, text }],
        },
        target: {
          text: " fish",
          token_ids: [15591],
          is_single_token: true,
          pieces: [{ id: 15591, text: " fish" }],
        },
      },
    );

    assert.deepEqual(payload.intervention.source_token_ids, [id]);
    assert.deepEqual(payload.source_token_texts, [text]);
    assert.equal(payload.intervention.target_token_id, 15591);
    assert.equal(payload.target_token_text, " fish");
  });
}

test("missing typed swap source fails closed", () => {
  assert.throws(() => buildInterventionPayload(
    { prompt: "x", layers: [26] },
    { lensType: "jacobian", entry: { id: 84369, text: "Spider" } },
    {
      mode: "swap",
      layers: "26",
      strength: -0.1,
      applyToGenerated: false,
      source: null,
      target: { text: " fish", token_ids: [15591], is_single_token: true },
    },
  ), /Swap source must tokenize to exactly one token/);
});

test("multi-token typed swap source fails closed", () => {
  assert.throws(() => buildInterventionPayload(
    { prompt: "x", layers: [26] },
    { lensType: "jacobian", entry: { id: 84369, text: "Spider" } },
    {
      mode: "swap",
      layers: "26",
      strength: -0.1,
      applyToGenerated: false,
      source: {
        text: "spider",
        token_ids: [2308, 2506],
        is_single_token: false,
        pieces: [{ id: 2308, text: "sp" }, { id: 2506, text: "ider" }],
      },
      target: { text: " fish", token_ids: [15591], is_single_token: true },
    },
  ), /Swap source must tokenize to exactly one token/);
});

test("multi-token swap target fails closed", () => {
  assert.throws(() => buildInterventionPayload(
    { prompt: "x", layers: [34] },
    { lensType: "jacobian", entry: { id: 192, text: " coral" } },
    {
      mode: "swap",
      layers: "34",
      strength: -0.1,
      applyToGenerated: false,
      source: { text: " Spider", token_ids: [42010], is_single_token: true },
      target: { text: "two tokens", token_ids: [3, 4], is_single_token: false },
    },
  ), /exactly one token/);
});

test("layer ranges canonicalize and accept all 52 readout layers", () => {
  assert.deepEqual(parseLayerList("40, 26-28,27"), [26, 27, 28, 40]);
  assert.deepEqual(
    parseLayerList("0-51"),
    Array.from({ length: 52 }, (_, layer) => layer),
  );
  assert.throws(() => parseLayerList("0-52"), /0–51/);
});

test("interventions accept all 51 writable layers and reject read-only layer 51", () => {
  const payload = buildInterventionPayload(
    { prompt: "x", layers: [13, 50] },
    { lensType: "jacobian", entry: { id: 84369, text: "Spider" } },
    {
      mode: "steer",
      layers: "0-50",
      strength: -0.1,
      applyToGenerated: false,
      source: null,
      target: null,
    },
  );

  assert.deepEqual(
    payload.intervention.layers,
    Array.from({ length: 51 }, (_, layer) => layer),
  );
  assert.throws(
    () => buildInterventionPayload(
      { prompt: "x", layers: [13, 50] },
      { lensType: "jacobian", entry: { id: 84369, text: "Spider" } },
      {
        mode: "steer",
        layers: "0-51",
        strength: -0.1,
        applyToGenerated: false,
        source: null,
        target: null,
      },
    ),
    /0–50/,
  );
});

test("baseline and paired progress remain monotonic through broad readouts", () => {
  const baseline = [
    { status: "running", phase: "starting" },
    { status: "running", phase: "tokenize" },
    { status: "running", phase: "directions", condition: "clean" },
    { status: "running", phase: "generate", condition: "clean", current: 32, total: 32 },
    { status: "running", phase: "capture", condition: "clean" },
    { status: "running", phase: "readout", condition: "clean", current: 1, total: 38 },
    { status: "running", phase: "readout", condition: "clean", current: 38, total: 38 },
    { status: "complete", phase: "complete" },
  ].map((status) => progressPercent(status, "baseline"));
  const paired = [
    { status: "running", phase: "directions", condition: "clean" },
    { status: "running", phase: "generate", condition: "clean", current: 32, total: 32 },
    { status: "running", phase: "capture", condition: "clean" },
    { status: "running", phase: "readout", condition: "clean", current: 38, total: 38 },
    { status: "running", phase: "directions", condition: "intervened" },
    { status: "running", phase: "generate", condition: "intervened", current: 32, total: 32 },
    { status: "running", phase: "capture", condition: "intervened" },
    { status: "running", phase: "readout", condition: "intervened", current: 38, total: 38 },
    { status: "complete", phase: "complete" },
  ].map((status) => progressPercent(status, "intervention"));

  for (const values of [baseline, paired]) {
    assert.ok(values.every(Number.isFinite));
    assert.deepEqual([...values].sort((left, right) => left - right), values);
    assert.equal(values.at(-1), 100);
  }
});
