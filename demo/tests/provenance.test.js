"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const provenance = require("../provenance.js");

const SHA_A = "a".repeat(64);
const SHA_B = "b".repeat(64);
const SHA_C = "c".repeat(64);

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

function acceptance(overrides = {}) {
  return {
    schema: provenance.ACCEPTANCE_SCHEMA,
    tier: "paper-scale",
    status: "accepted",
    is_final: true,
    exportable: true,
    checks: Object.fromEntries(provenance.REQUIRED_ACCEPTANCE_CHECKS.map((check) => [check, true])),
    reasons: [],
    ...overrides,
  };
}

function fixture({
  nPrompts = provenance.PAPER_SCALE_PROMPTS,
  decision = acceptance(),
  example = provenance.PAPER_SCALE_EXAMPLE,
  lensOverrides = {},
} = {}) {
  return {
    _fixture: {
      schema: provenance.FIXTURE_SCHEMA,
      mode: "recorded",
      example,
      model_revision: provenance.PINNED_MODEL_REVISION,
      lens_sha256: SHA_A,
      acceptance: decision,
      provenance: {
        model: { id: provenance.PINNED_MODEL_ID, revision: provenance.PINNED_MODEL_REVISION },
        tokenizer: { id: provenance.PINNED_MODEL_ID, revision: provenance.PINNED_MODEL_REVISION },
        lens: {
          sha256: SHA_A,
          corpus_manifest_sha256: provenance.PAPER_SCALE_CORPUS_MANIFEST_SHA256,
          adaptation_source_sha256: SHA_C,
          upstream_jlens_commit: provenance.PINNED_UPSTREAM_COMMIT,
          n_prompts: nPrompts,
          source_layers: [...provenance.PAPER_SCALE_SOURCE_LAYERS],
          target_layer: provenance.PAPER_SCALE_TARGET_LAYER,
          dataset: { ...provenance.PINNED_DATASET },
          storage_dtype: "float32",
          runtime: {
            python: "3.12.3",
            torch: "2.9.0a0+50eac811a6.nv25.9",
            cuda_runtime: "13.0",
            packages: { transformers: provenance.REQUIRED_TRANSFORMERS_VERSION },
          },
          fit: { ...provenance.PAPER_SCALE_FIT },
          acceptance: clone(decision),
          ...lensOverrides,
        },
        prompt: { example, sha256: SHA_B },
        exporter: {
          adaptation_source_sha256: SHA_C,
          upstream_jlens_commit: provenance.PINNED_UPSTREAM_COMMIT,
        },
      },
    },
    meta: { model: provenance.PINNED_MODEL_ID },
  };
}

test("only exact accepted paper-scale metadata is final-default eligible", () => {
  const candidate = fixture();
  assert.equal(provenance.isFinalDefaultEligible(candidate), true);
  assert.deepEqual(provenance.acceptanceSummary(candidate), {
    tier: "paper-scale",
    status: "accepted",
    label: "Paper Scale · Accepted",
    badgeStatus: "final",
    finalDefaultEligible: true,
    underPaperScale: false,
    reasons: [],
    checks: provenance.REQUIRED_ACCEPTANCE_CHECKS.map((check) => ({
      label: check.replace(/_/g, " ").replace(/\b\w/g, (character) => character.toUpperCase()),
      status: "passed",
      detail: "",
    })),
  });
});

test("smoke recordings remain non-final and opt-in", () => {
  const candidate = fixture({
    nPrompts: 8,
    decision: acceptance({ tier: "smoke", status: "non-final", is_final: false, reasons: ["Smoke fit only"] }),
  });
  assert.doesNotThrow(() => provenance.validateRecordedProvenance(candidate));
  assert.equal(provenance.isFinalDefaultEligible(candidate), false);
  const summary = provenance.acceptanceSummary(candidate);
  assert.equal(summary.badgeStatus, "nonfinal");
  assert.equal(summary.finalDefaultEligible, false);
  assert.deepEqual(summary.reasons, ["Smoke fit only"]);
});

test("an accepted pilot is valid but remains non-final and opt-in", () => {
  const candidate = fixture({
    nPrompts: 100,
    decision: acceptance({ tier: "pilot", is_final: false, reasons: ["Canonical pilot completed"] }),
  });
  assert.doesNotThrow(() => provenance.validateRecordedProvenance(candidate));
  assert.equal(provenance.isFinalDefaultEligible(candidate), false);
  const summary = provenance.acceptanceSummary(candidate);
  assert.equal(summary.status, "accepted");
  assert.equal(summary.tier, "pilot");
  assert.equal(summary.badgeStatus, "nonfinal");
  assert.equal(summary.finalDefaultEligible, false);
});

test("recordings without an acceptance decision stay non-final", () => {
  const candidate = fixture();
  delete candidate._fixture.acceptance;
  delete candidate._fixture.provenance.lens.acceptance;
  assert.equal(provenance.isFinalDefaultEligible(candidate), false);
  assert.equal(provenance.acceptanceSummary(candidate).status, "undeclared");
});

test("paper-scale labels cannot bypass the fixed prompt floor", () => {
  const candidate = fixture({ nPrompts: provenance.PAPER_SCALE_PROMPTS - 1 });
  assert.throws(() => provenance.validateRecordedProvenance(candidate), /at least 1,000 fitting prompts/);
  assert.equal(provenance.isFinalDefaultEligible(candidate), false);
});

test("paper-scale default eligibility requires exactly 1,000 prompts", () => {
  const candidate = fixture({ nPrompts: provenance.PAPER_SCALE_PROMPTS + 1 });
  assert.doesNotThrow(() => provenance.validateRecordedProvenance(candidate));
  assert.equal(provenance.isFinalDefaultEligible(candidate), false);
});

test("every canonical paper-scale lens field is enforced before auto-load", () => {
  const cases = [
    ["source layers", { source_layers: provenance.PAPER_SCALE_SOURCE_LAYERS.slice(0, -1) }],
    ["target layer", { target_layer: provenance.PAPER_SCALE_TARGET_LAYER - 1 }],
    ["corpus manifest", { corpus_manifest_sha256: SHA_B }],
    ["dataset id", { dataset: { ...provenance.PINNED_DATASET, id: "other/dataset" } }],
    ["dataset config", { dataset: { ...provenance.PINNED_DATASET, config: "other-config" } }],
    ["dataset revision", { dataset: { ...provenance.PINNED_DATASET, revision: SHA_A } }],
    ["dataset split", { dataset: { ...provenance.PINNED_DATASET, split: "test" } }],
    ["storage dtype", { storage_dtype: "float16" }],
    [
      "Transformers version",
      {
        runtime: {
          python: "3.12.3",
          torch: "2.9.0a0+50eac811a6.nv25.9",
          cuda_runtime: "13.0",
          packages: { transformers: "5.9.0" },
        },
      },
    ],
    ["dimension batch", { fit: { ...provenance.PAPER_SCALE_FIT, dim_batch: 16 } }],
    ["sequence length", { fit: { ...provenance.PAPER_SCALE_FIT, max_seq_len: 64 } }],
    ["skip count", { fit: { ...provenance.PAPER_SCALE_FIT, skip_first: 0 } }],
    ["model dtype", { fit: { ...provenance.PAPER_SCALE_FIT, dtype: "float16" } }],
    ["compile mode", { fit: { ...provenance.PAPER_SCALE_FIT, compile_blocks: true } }],
    ["Mamba flag", { fit: { ...provenance.PAPER_SCALE_FIT, disable_mamba_kernels: true } }],
    ["Mamba backend", { fit: { ...provenance.PAPER_SCALE_FIT, mamba_backend: "torch" } }],
    ["patched layers", { fit: { ...provenance.PAPER_SCALE_FIT, patched_mamba_layers: 23 } }],
  ];
  for (const [label, lensOverrides] of cases) {
    const candidate = fixture({ lensOverrides });
    assert.doesNotThrow(() => provenance.validateRecordedProvenance(candidate), label);
    assert.equal(provenance.isFinalDefaultEligible(candidate), false, label);
  }
});

test("only the exact focus example can become the startup fixture", () => {
  const candidate = fixture({ example: "modulation-topic-neutral" });
  assert.doesNotThrow(() => provenance.validateRecordedProvenance(candidate));
  assert.equal(provenance.isFinalDefaultEligible(candidate), false);

  const inconsistent = fixture();
  inconsistent._fixture.provenance.prompt.example = "modulation-topic-neutral";
  assert.equal(provenance.isFinalDefaultEligible(inconsistent), false);
});

test("top-level and lens acceptance decisions must be deeply identical", () => {
  const candidate = fixture();
  candidate._fixture.provenance.lens.acceptance = acceptance({
    tier: "smoke",
    status: "non-final",
    is_final: false,
    reasons: ["Smoke only"],
  });
  assert.throws(
    () => provenance.validateRecordedProvenance(candidate),
    /must exactly match provenance\.lens\.acceptance/,
  );
  assert.equal(provenance.isFinalDefaultEligible(candidate), false);
});

test("accepted fixtures must pass every exported acceptance check", () => {
  const checks = Object.fromEntries(provenance.REQUIRED_ACCEPTANCE_CHECKS.map((check) => [check, true]));
  checks.fp32_storage = false;
  const candidate = fixture({ decision: acceptance({ checks }) });
  assert.throws(() => provenance.validateRecordedProvenance(candidate), /pass every required acceptance check/);
  assert.equal(provenance.isFinalDefaultEligible(candidate), false);
});

test("accepted paper-scale fixtures must explicitly be final", () => {
  const candidate = fixture({ decision: acceptance({ is_final: false }) });
  assert.throws(() => provenance.validateRecordedProvenance(candidate), /must declare is_final=true/);
  assert.equal(provenance.isFinalDefaultEligible(candidate), false);
});

test("unknown acceptance values fail closed", () => {
  const candidate = fixture({ decision: acceptance({ status: "approved" }) });
  assert.throws(() => provenance.validateRecordedProvenance(candidate), /accepted or non-final/);
  assert.equal(provenance.isFinalDefaultEligible(candidate), false);
});

test("illustrative fixtures are never eligible", () => {
  const candidate = { _fixture: { mode: "illustrative" }, meta: { model: provenance.PINNED_MODEL_ID } };
  assert.equal(provenance.validateRecordedProvenance(candidate), null);
  assert.equal(provenance.isFinalDefaultEligible(candidate), false);
});
