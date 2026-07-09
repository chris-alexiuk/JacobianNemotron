(function attachNemotronFixtureProvenance(root, factory) {
  "use strict";

  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.NemotronFixtureProvenance = api;
})(typeof globalThis === "object" ? globalThis : this, function buildNemotronFixtureProvenance() {
  "use strict";

  const FIXTURE_SCHEMA = "nemotron-jlens-fixture/v1";
  const ACCEPTANCE_SCHEMA = "nemotron-jlens-acceptance/v1";
  const PINNED_MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16";
  const PINNED_MODEL_REVISION = "cbd3fa9f933d55ef16a84236559f4ee2a0526848";
  const PINNED_UPSTREAM_COMMIT = "581d398613e5602a5af361e1c34d3a92ea82ba8e";
  const REQUIRED_TRANSFORMERS_VERSION = "4.57.3";
  const PAPER_SCALE_PROMPTS = 1000;
  const PAPER_SCALE_CORPUS_MANIFEST_SHA256 = "c75fc7ee5d92335f0620a08c6f87d210fc3b3fe4f3d6bcfae7d0cd864a88d63e";
  const PAPER_SCALE_EXAMPLE = "modulation-topic";
  const PAPER_SCALE_SOURCE_LAYERS = Object.freeze(Array.from({ length: 51 }, (_, index) => index));
  const PAPER_SCALE_TARGET_LAYER = 51;
  const PINNED_DATASET = Object.freeze({
    id: "Salesforce/wikitext",
    config: "wikitext-103-raw-v1",
    revision: "b08601e04326c79dfdd32d625aee71d232d685c3",
    split: "train",
  });
  const PAPER_SCALE_FIT = Object.freeze({
    dim_batch: 8,
    max_seq_len: 128,
    skip_first: 16,
    dtype: "bfloat16",
    compile_blocks: false,
    disable_mamba_kernels: false,
    mamba_backend: "fused-or-auto",
    patched_mamba_layers: 0,
  });
  const ACCEPTANCE_STATUSES = new Set(["accepted", "non-final"]);
  const ACCEPTANCE_TIERS = new Set(["smoke", "pilot", "paper-scale"]);
  const REQUIRED_ACCEPTANCE_CHECKS = Object.freeze([
    "pinned_model",
    "pinned_dataset",
    "known_manifest",
    "complete_prompt_set",
    "full_layer_coverage",
    "fixed_fit_settings",
    "pinned_transformers",
    "fp32_storage",
  ]);
  const SHA256 = /^[0-9a-f]{64}$/;

  function isPlainObject(value) {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
  }

  function isSha256(value) {
    return typeof value === "string" && SHA256.test(value);
  }

  function deepEqual(left, right) {
    if (Object.is(left, right)) return true;
    if (Array.isArray(left) || Array.isArray(right)) {
      return (
        Array.isArray(left) &&
        Array.isArray(right) &&
        left.length === right.length &&
        left.every((value, index) => deepEqual(value, right[index]))
      );
    }
    if (!isPlainObject(left) || !isPlainObject(right)) return false;
    const leftKeys = Object.keys(left).sort();
    const rightKeys = Object.keys(right).sort();
    return (
      leftKeys.length === rightKeys.length &&
      leftKeys.every((key, index) => key === rightKeys[index] && deepEqual(left[key], right[key]))
    );
  }

  function hasExactFields(value, expected) {
    return isPlainObject(value) && Object.entries(expected).every(([key, expectedValue]) => value[key] === expectedValue);
  }

  function isCanonicalPaperScaleContract(recorded) {
    const { info, lens, prompt } = recorded;
    return Boolean(
      lens.n_prompts === PAPER_SCALE_PROMPTS &&
        deepEqual(lens.source_layers, PAPER_SCALE_SOURCE_LAYERS) &&
        lens.target_layer === PAPER_SCALE_TARGET_LAYER &&
        lens.corpus_manifest_sha256 === PAPER_SCALE_CORPUS_MANIFEST_SHA256 &&
        hasExactFields(lens.dataset, PINNED_DATASET) &&
        lens.storage_dtype === "float32" &&
        isPlainObject(lens.runtime) &&
        isPlainObject(lens.runtime.packages) &&
        lens.runtime.packages.transformers === REQUIRED_TRANSFORMERS_VERSION &&
        hasExactFields(lens.fit, PAPER_SCALE_FIT) &&
        info.example === PAPER_SCALE_EXAMPLE &&
        prompt.example === PAPER_SCALE_EXAMPLE
    );
  }

  function validateAcceptance(value) {
    if (value === undefined) return null;
    if (!isPlainObject(value)) throw new Error("_fixture.acceptance must be an object when present.");
    if (value.schema !== ACCEPTANCE_SCHEMA) {
      throw new Error(`Recorded fixture acceptance must use ${ACCEPTANCE_SCHEMA}.`);
    }
    if (!ACCEPTANCE_STATUSES.has(value.status)) {
      throw new Error("Recorded fixture acceptance status must be accepted or non-final.");
    }
    if (!ACCEPTANCE_TIERS.has(value.tier)) {
      throw new Error("Recorded fixture acceptance tier must be smoke, pilot, or paper-scale.");
    }
    if (typeof value.is_final !== "boolean" || value.exportable !== true) {
      throw new Error("Recorded fixture acceptance must declare boolean is_final and exportable=true.");
    }
    if (
      !isPlainObject(value.checks) ||
      !REQUIRED_ACCEPTANCE_CHECKS.every((check) => typeof value.checks[check] === "boolean")
    ) {
      throw new Error("Recorded fixture acceptance is missing required boolean checks.");
    }
    if (!Array.isArray(value.reasons) || !value.reasons.every((reason) => typeof reason === "string")) {
      throw new Error("Recorded fixture acceptance reasons must be an array of strings.");
    }
    const accepted = value.status === "accepted";
    if (value.is_final && (!accepted || value.tier !== "paper-scale")) {
      throw new Error("Only an accepted paper-scale fixture may declare is_final=true.");
    }
    if (value.status === "non-final" && value.is_final) {
      throw new Error("A non-final acceptance decision must declare is_final=false.");
    }
    if (accepted && value.tier === "paper-scale" && !value.is_final) {
      throw new Error("An accepted paper-scale fixture must declare is_final=true.");
    }
    if (
      accepted &&
      value.tier === "paper-scale" &&
      REQUIRED_ACCEPTANCE_CHECKS.some((check) => value.checks[check] !== true)
    ) {
      throw new Error("Accepted paper-scale fixtures must pass every required acceptance check.");
    }
    return value;
  }

  function validateRecordedProvenance(fixture) {
    const info = fixture && fixture._fixture;
    if (!info || info.mode !== "recorded") return null;
    const provenance = info.provenance;
    const model = provenance && provenance.model;
    const tokenizer = provenance && provenance.tokenizer;
    const lens = provenance && provenance.lens;
    const prompt = provenance && provenance.prompt;
    const exporter = provenance && provenance.exporter;

    if (
      info.schema !== FIXTURE_SCHEMA ||
      (fixture.meta && fixture.meta.model) !== PINNED_MODEL_ID ||
      info.model_revision !== PINNED_MODEL_REVISION ||
      !model ||
      model.id !== PINNED_MODEL_ID ||
      model.revision !== PINNED_MODEL_REVISION ||
      !tokenizer ||
      tokenizer.id !== PINNED_MODEL_ID ||
      tokenizer.revision !== PINNED_MODEL_REVISION
    ) {
      throw new Error("Recorded fixtures must declare the exact pinned Nemotron model and tokenizer revision.");
    }

    if (
      !lens ||
      !prompt ||
      !exporter ||
      !isSha256(info.lens_sha256) ||
      info.lens_sha256 !== lens.sha256 ||
      !isSha256(lens.corpus_manifest_sha256) ||
      !isSha256(lens.adaptation_source_sha256) ||
      lens.upstream_jlens_commit !== PINNED_UPSTREAM_COMMIT ||
      !Number.isInteger(lens.n_prompts) ||
      lens.n_prompts <= 0 ||
      !Array.isArray(lens.source_layers) ||
      !lens.source_layers.length ||
      !lens.source_layers.every((layer) => Number.isInteger(layer) && layer >= 0) ||
      new Set(lens.source_layers).size !== lens.source_layers.length ||
      !Number.isInteger(lens.target_layer) ||
      lens.target_layer < 0 ||
      !isSha256(prompt.sha256) ||
      typeof prompt.example !== "string" ||
      !prompt.example ||
      !isSha256(exporter.adaptation_source_sha256) ||
      exporter.upstream_jlens_commit !== PINNED_UPSTREAM_COMMIT
    ) {
      throw new Error("Recorded fixture provenance is missing or malformed.");
    }

    const hasTopLevelAcceptance = Object.prototype.hasOwnProperty.call(info, "acceptance");
    const hasLensAcceptance = Object.prototype.hasOwnProperty.call(lens, "acceptance");
    if (
      hasTopLevelAcceptance !== hasLensAcceptance ||
      (hasTopLevelAcceptance && !deepEqual(info.acceptance, lens.acceptance))
    ) {
      throw new Error("Recorded fixture acceptance must exactly match provenance.lens.acceptance.");
    }
    const acceptance = validateAcceptance(info.acceptance);
    if (
      acceptance &&
      acceptance.status === "accepted" &&
      acceptance.tier === "paper-scale" &&
      lens.n_prompts < PAPER_SCALE_PROMPTS
    ) {
      throw new Error(
        `Paper-scale acceptance requires at least ${PAPER_SCALE_PROMPTS.toLocaleString("en")} fitting prompts.`,
      );
    }

    return { info, provenance, model, tokenizer, lens, prompt, exporter, acceptance };
  }

  function isFinalDefaultEligible(fixture) {
    try {
      const recorded = validateRecordedProvenance(fixture);
      return Boolean(
        recorded &&
          recorded.acceptance &&
          recorded.acceptance.schema === ACCEPTANCE_SCHEMA &&
          recorded.acceptance.status === "accepted" &&
          recorded.acceptance.tier === "paper-scale" &&
          recorded.acceptance.is_final === true &&
          recorded.acceptance.exportable === true &&
          REQUIRED_ACCEPTANCE_CHECKS.every((check) => recorded.acceptance.checks[check] === true) &&
          isCanonicalPaperScaleContract(recorded),
      );
    } catch (_error) {
      return false;
    }
  }

  function words(value) {
    return String(value || "")
      .replace(/[-_]+/g, " ")
      .replace(/\b\w/g, (character) => character.toUpperCase());
  }

  function normalizeReasons(value) {
    if (typeof value === "string" && value.trim()) return [value.trim()];
    if (!Array.isArray(value)) return [];
    return value.filter((reason) => typeof reason === "string" && reason.trim()).map((reason) => reason.trim());
  }

  function normalizeCheckEntry(label, value) {
    if (typeof value === "boolean") {
      return { label: words(label), status: value ? "passed" : "failed", detail: "" };
    }
    if (typeof value === "string") {
      return { label: words(label), status: value, detail: "" };
    }
    if (!isPlainObject(value)) return null;
    const entryLabel = value.label || value.name || value.check || label;
    const rawStatus = value.status !== undefined ? value.status : value.ok;
    const status =
      typeof rawStatus === "boolean" ? (rawStatus ? "passed" : "failed") : String(rawStatus || "reported");
    const detail = value.detail || value.reason || value.description || "";
    return { label: words(entryLabel), status, detail: typeof detail === "string" ? detail : String(detail) };
  }

  function normalizeChecks(value) {
    if (Array.isArray(value)) {
      return value
        .map((entry, index) => {
          if (typeof entry === "string") return { label: entry, status: "reported", detail: "" };
          return normalizeCheckEntry(`Check ${index + 1}`, entry);
        })
        .filter(Boolean);
    }
    if (!isPlainObject(value)) return [];
    return Object.entries(value)
      .map(([label, entry]) => normalizeCheckEntry(label, entry))
      .filter(Boolean);
  }

  function acceptanceSummary(fixture) {
    const recorded = validateRecordedProvenance(fixture);
    if (!recorded) return null;
    const { acceptance, lens } = recorded;
    if (!acceptance) {
      return {
        tier: "undeclared",
        status: "undeclared",
        label: "Acceptance undeclared",
        badgeStatus: "undeclared",
        finalDefaultEligible: false,
        reasons: [],
        checks: [],
      };
    }

    const finalDefaultEligible = isFinalDefaultEligible(fixture);
    return {
      tier: acceptance.tier,
      status: acceptance.status,
      label: `${words(acceptance.tier)} · ${words(acceptance.status)}`,
      badgeStatus: finalDefaultEligible ? "final" : "nonfinal",
      finalDefaultEligible,
      underPaperScale: lens.n_prompts < PAPER_SCALE_PROMPTS,
      reasons: normalizeReasons(acceptance.reasons !== undefined ? acceptance.reasons : acceptance.reason),
      checks: normalizeChecks(acceptance.checks),
    };
  }

  return Object.freeze({
    FIXTURE_SCHEMA,
    ACCEPTANCE_SCHEMA,
    PINNED_MODEL_ID,
    PINNED_MODEL_REVISION,
    PINNED_UPSTREAM_COMMIT,
    REQUIRED_TRANSFORMERS_VERSION,
    PAPER_SCALE_PROMPTS,
    PAPER_SCALE_CORPUS_MANIFEST_SHA256,
    PAPER_SCALE_EXAMPLE,
    PAPER_SCALE_SOURCE_LAYERS,
    PAPER_SCALE_TARGET_LAYER,
    PINNED_DATASET,
    PAPER_SCALE_FIT,
    REQUIRED_ACCEPTANCE_CHECKS,
    isSha256,
    validateAcceptance,
    validateRecordedProvenance,
    isFinalDefaultEligible,
    acceptanceSummary,
    normalizeReasons,
    normalizeChecks,
  });
});
