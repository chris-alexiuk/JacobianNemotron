(() => {
  "use strict";

  const Provenance = globalThis.NemotronFixtureProvenance;
  if (!Provenance) throw new Error("provenance.js must load before app.js");

  const FINAL_FIXTURE_URL = "./fixtures/nemotron-3-nano.recorded.json";
  const PILOT_FIXTURE_URL = "./fixtures/nemotron-3-nano.pilot-recorded.json";
  const SAMPLE_FIXTURE_URL = "./fixtures/nemotron-3-nano.sample.json";
  const JACOBIAN = "JACOBIAN_LENS";
  const LOGIT = "LOGIT_LENS";
  const PAPER_SCALE_PROMPTS = Provenance.PAPER_SCALE_PROMPTS;
  const DIFF = "DIFF";
  const LOCK_COLORS = ["#0c9db3", "#d9732f", "#7758d8", "#198a65"];
  const MAX_LOCKS = LOCK_COLORS.length;

  const state = {
    fixture: null,
    source: "Bundled sample",
    mode: JACOBIAN,
    layerStart: 0,
    layerEnd: 0,
    topN: 6,
    maxTopN: 6,
    selectedPositions: new Set(),
    hoveredPosition: null,
    previewToken: null,
    lockedTokens: [],
  };

  const el = {};
  let toastTimer = null;

  document.addEventListener("DOMContentLoaded", boot);

  function boot() {
    [
      "fixture-input",
      "import-button",
      "export-button",
      "fixture-pill",
      "fixture-status",
      "fixture-notice-title",
      "fixture-notice-body",
      "fixture-source",
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
      "acceptance-details",
      "acceptance-reasons",
      "acceptance-checks",
      "model-name",
      "run-summary",
      "mode-tabs",
      "layer-start",
      "layer-end",
      "layer-start-output",
      "layer-end-output",
      "layer-range-label",
      "top-n",
      "top-n-limit",
      "loading-state",
      "error-state",
      "error-message",
      "workspace",
      "transcript-content",
      "selection-summary",
      "clear-selection",
      "readout-position",
      "layer-readout",
      "sidebar-title",
      "sidebar-count",
      "sidebar-description",
      "locked-tokens",
      "layer-axis-label",
      "token-list",
      "tooltip",
      "toast",
    ].forEach((id) => {
      el[id] = document.getElementById(id);
    });

    bindControls();
    loadRequestedFixture();
  }

  async function loadRequestedFixture() {
    const requested = new URLSearchParams(globalThis.location.search).get("fixture");
    if (requested !== "pilot") {
      await loadBundledFixture();
      return;
    }

    try {
      const response = await fetch(PILOT_FIXTURE_URL, { cache: "no-store" });
      if (!response.ok) throw new Error(`Pilot fixture request returned HTTP ${response.status}.`);
      const fixture = validateFixture(await response.json());
      const summary = Provenance.acceptanceSummary(fixture);
      const example = fixture._fixture.provenance.prompt.example;
      if (
        !summary ||
        summary.tier !== "pilot" ||
        summary.status !== "accepted" ||
        summary.finalDefaultEligible ||
        example !== "modulation-topic"
      ) {
        throw new Error("The opt-in pilot fixture is not an accepted non-final focus recording.");
      }
      setFixture(fixture, "Bundled accepted pilot · explicit opt-in");
    } catch (error) {
      showLoadError(error);
    }
  }

  function bindControls() {
    el["mode-tabs"].addEventListener("click", (event) => {
      const button = event.target.closest("[data-mode]");
      if (!button || !state.fixture) return;
      setMode(button.dataset.mode);
    });

    el["mode-tabs"].addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      const tabs = [...el["mode-tabs"].querySelectorAll("[data-mode]")];
      const current = tabs.findIndex((tab) => tab.dataset.mode === state.mode);
      let next = current;
      if (event.key === "ArrowLeft") next = (current - 1 + tabs.length) % tabs.length;
      if (event.key === "ArrowRight") next = (current + 1) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      event.preventDefault();
      setMode(tabs[next].dataset.mode);
      tabs[next].focus();
    });

    el["layer-start"].addEventListener("input", () => {
      state.layerStart = Number(el["layer-start"].value);
      if (state.layerStart > state.layerEnd) state.layerEnd = state.layerStart;
      renderControls();
      renderAnalysis();
    });

    el["layer-end"].addEventListener("input", () => {
      state.layerEnd = Number(el["layer-end"].value);
      if (state.layerEnd < state.layerStart) state.layerStart = state.layerEnd;
      renderControls();
      renderAnalysis();
    });

    el["top-n"].addEventListener("change", updateTopN);
    el["top-n"].addEventListener("input", updateTopN);

    el["clear-selection"].addEventListener("click", () => {
      state.selectedPositions.clear();
      state.hoveredPosition = null;
      updateTranscriptStates();
      renderAnalysis();
    });

    el["import-button"].addEventListener("click", () => el["fixture-input"].click());
    el["fixture-input"].addEventListener("change", importFixture);
    el["export-button"].addEventListener("click", exportFixture);

    document.addEventListener("keydown", (event) => {
      if (event.key !== "Escape" || !state.fixture) return;
      if (state.previewToken) {
        state.previewToken = null;
      } else if (state.hoveredPosition !== null) {
        state.hoveredPosition = null;
      } else if (state.lockedTokens.length) {
        state.lockedTokens = [];
        renderLockedTokens();
        renderSidebar();
      } else {
        state.selectedPositions.clear();
      }
      updateTranscriptStates();
      renderAnalysis();
    });
  }

  async function loadBundledFixture() {
    try {
      const response = await fetch(FINAL_FIXTURE_URL, { cache: "no-store" });
      if (response.ok) {
        const fixture = await response.json();
        validateFixture(fixture);
        if (Provenance.isFinalDefaultEligible(fixture)) {
          setFixture(fixture, "Bundled accepted recording");
          return;
        }
      }
    } catch (_error) {
      // Missing, malformed, and non-final recordings never displace the sample.
    }

    try {
      const response = await fetch(SAMPLE_FIXTURE_URL, { cache: "no-store" });
      if (!response.ok) throw new Error(`Fixture request returned HTTP ${response.status}.`);
      const fixture = await response.json();
      setFixture(fixture, "Bundled sample");
    } catch (error) {
      showLoadError(error);
    }
  }

  function validateRecordedProvenance(fixture) {
    Provenance.validateRecordedProvenance(fixture);
  }

  function validateFixture(fixture) {
    if (!fixture || typeof fixture !== "object") throw new Error("The JSON root must be an object.");
    if (!fixture.meta || fixture.meta.kind !== "meta") throw new Error("Missing a valid meta message.");
    validateRecordedProvenance(fixture);
    if (!fixture.meta.layers_by_type || typeof fixture.meta.layers_by_type !== "object") {
      throw new Error("meta.layers_by_type is required.");
    }
    for (const type of [JACOBIAN, LOGIT]) {
      const layers = fixture.meta.layers_by_type[type];
      if (
        !Array.isArray(layers) ||
        !layers.length ||
        !layers.every(Number.isFinite) ||
        new Set(layers).size !== layers.length
      ) {
        throw new Error(`meta.layers_by_type.${type} must contain layer numbers.`);
      }
    }
    const jacobianLayers = fixture.meta.layers_by_type[JACOBIAN];
    const logitLayers = new Set(fixture.meta.layers_by_type[LOGIT]);
    if (jacobianLayers.length !== logitLayers.size || !jacobianLayers.every((layer) => logitLayers.has(layer))) {
      throw new Error("Jacobian and logit readouts must declare identical layer sets for Difference mode.");
    }
    if (!Array.isArray(fixture.tokens) || !fixture.tokens.length) {
      throw new Error("tokens must be a non-empty array.");
    }
    const positions = new Set();
    for (const token of fixture.tokens) {
      if (
        token.kind !== "token" ||
        !Number.isInteger(token.position) ||
        typeof token.token !== "string" ||
        typeof token.is_generated !== "boolean"
      ) {
        throw new Error("Each token needs kind, integer position, token text, and a boolean is_generated flag.");
      }
      if (positions.has(token.position)) throw new Error(`Duplicate token position ${token.position}.`);
      positions.add(token.position);
      if (!Array.isArray(token.results)) throw new Error(`Position ${token.position} is missing results.`);
      for (const type of [JACOBIAN, LOGIT]) {
        const result = token.results.find((item) => item.type === type);
        const layerCount = fixture.meta.layers_by_type[type].length;
        if (!result || !Array.isArray(result.top_tokens) || result.top_tokens.length !== layerCount) {
          throw new Error(`Position ${token.position} has an invalid ${type} token matrix.`);
        }
        if (!Array.isArray(result.top_probs) || result.top_probs.length !== layerCount) {
          throw new Error(`Position ${token.position} has an invalid ${type} probability matrix.`);
        }
        result.top_tokens.forEach((row, layerIndex) => {
          const probabilities = result.top_probs[layerIndex];
          if (!Array.isArray(row) || !row.every((value) => typeof value === "string")) {
            throw new Error(`Position ${token.position} has an invalid ${type} token row.`);
          }
          if (
            !Array.isArray(probabilities) ||
            probabilities.length !== row.length ||
            !probabilities.every((value) => Number.isFinite(value) && value >= 0 && value <= 1)
          ) {
            throw new Error(`Position ${token.position} has an invalid ${type} probability row.`);
          }
        });
        validateTracked(result.tracked, layerCount, token.position, type);
      }
    }
    validateInitialUi(fixture._fixture && fixture._fixture.ui, positions);
    return fixture;
  }

  function validateTracked(tracked, layerCount, position, type) {
    if (tracked === undefined) return;
    if (!tracked || typeof tracked !== "object") {
      throw new Error(`Position ${position} has invalid ${type} tracked-token data.`);
    }
    const entries = Array.isArray(tracked)
      ? tracked.map((entry) => [entry?.token, entry])
      : Object.entries(tracked);
    for (const [key, entry] of entries) {
      if (!entry || typeof entry !== "object" || Array.isArray(entry)) {
        throw new Error(`Position ${position} has an invalid ${type} tracked-token entry.`);
      }
      const label = entry.token === undefined && !Array.isArray(tracked) ? key : entry.token;
      if (typeof label !== "string" || !tokenKey(label)) {
        throw new Error(`Position ${position} has an invalid ${type} tracked token label.`);
      }
      if (entry.id !== undefined && (!Number.isInteger(entry.id) || entry.id < 0)) {
        throw new Error(`Position ${position} has an invalid ${type} tracked token id.`);
      }
      for (const [field, valid] of [
        ["ranks", (value) => value === null || (Number.isInteger(value) && value >= 1)],
        ["probs", (value) => value === null || (Number.isFinite(value) && value >= 0 && value <= 1)],
      ]) {
        if (!Array.isArray(entry[field]) || entry[field].length !== layerCount || !entry[field].every(valid)) {
          throw new Error(`Position ${position} has invalid ${type} tracked ${field}.`);
        }
      }
    }
  }

  function validateInitialUi(ui, positions) {
    if (ui === undefined) return;
    if (!ui || typeof ui !== "object" || Array.isArray(ui)) throw new Error("_fixture.ui must be an object.");
    const selected = ui.selected_positions;
    if (
      selected !== undefined &&
      (!Array.isArray(selected) || !selected.every((position) => Number.isInteger(position) && positions.has(position)))
    ) {
      throw new Error("_fixture.ui.selected_positions must contain valid token positions.");
    }
    const locked = ui.locked_tokens;
    if (
      locked !== undefined &&
      (!Array.isArray(locked) ||
        locked.length > MAX_LOCKS ||
        !locked.every((token) => typeof token === "string" && tokenKey(token)))
    ) {
      throw new Error(`_fixture.ui.locked_tokens must contain at most ${MAX_LOCKS} non-empty strings.`);
    }
  }

  function setFixture(rawFixture, source) {
    const fixture = validateFixture(rawFixture);
    state.fixture = fixture;
    state.source = source;
    state.mode = JACOBIAN;
    state.maxTopN = deriveMaxTopN(fixture);
    state.topN = Math.max(1, Math.min(Number(fixture.meta.top_n) || state.maxTopN, state.maxTopN));
    const initialUi = fixture._fixture && fixture._fixture.ui;
    state.selectedPositions = new Set(initialUi?.selected_positions || []);
    state.hoveredPosition = null;
    state.previewToken = null;
    state.lockedTokens = (initialUi?.locked_tokens || []).map((token) => ({ key: tokenKey(token), display: token }));
    const layers = availableLayers();
    state.layerStart = 0;
    state.layerEnd = Math.max(0, layers.length - 1);

    el["loading-state"].hidden = true;
    el["error-state"].hidden = true;
    el.workspace.hidden = false;
    renderControls();
    renderTranscript();
    renderAnalysis();
  }

  function deriveMaxTopN(fixture) {
    let max = 1;
    for (const token of fixture.tokens) {
      for (const result of token.results) {
        for (const row of result.top_tokens) max = Math.max(max, row.length);
      }
    }
    return max;
  }

  function showLoadError(error) {
    el["loading-state"].hidden = true;
    el["error-state"].hidden = false;
    el["error-message"].textContent = error instanceof Error ? error.message : String(error);
    el["fixture-source"].textContent = "Fixture unavailable";
  }

  function importFixture(event) {
    const [file] = event.target.files;
    event.target.value = "";
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const fixture = JSON.parse(String(reader.result));
        setFixture(fixture, `Imported · ${file.name}`);
        showToast(`Loaded ${file.name}`);
      } catch (error) {
        showToast(`Import failed: ${error instanceof Error ? error.message : String(error)}`);
      }
    };
    reader.onerror = () => showToast(`Could not read ${file.name}.`);
    reader.readAsText(file);
  }

  function exportFixture() {
    if (!state.fixture) return;
    const fixture = JSON.parse(JSON.stringify(state.fixture));
    fixture._fixture = fixture._fixture || {};
    fixture._fixture.ui = {
      selected_positions: [...state.selectedPositions].sort((a, b) => a - b),
      locked_tokens: state.lockedTokens.map((token) => token.display),
    };
    const blob = new Blob([`${JSON.stringify(fixture, null, 2)}\n`], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "nemotron-3-nano-lens-fixture.json";
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
    showToast("Fixture exported as JSON.");
  }

  function setMode(mode) {
    if (![JACOBIAN, LOGIT, DIFF].includes(mode)) return;
    state.mode = mode;
    const layers = availableLayers();
    state.layerStart = Math.min(state.layerStart, layers.length - 1);
    state.layerEnd = Math.min(Math.max(state.layerStart, state.layerEnd), layers.length - 1);
    state.previewToken = null;
    renderControls();
    updateTranscriptStates();
    renderAnalysis();
  }

  function updateTopN() {
    if (!state.fixture) return;
    const parsed = Number.parseInt(el["top-n"].value, 10);
    if (!Number.isFinite(parsed)) return;
    state.topN = Math.max(1, Math.min(parsed, state.maxTopN));
    el["top-n"].value = String(state.topN);
    updateTranscriptStates();
    renderAnalysis();
  }

  function renderControls() {
    if (!state.fixture) return;
    const { meta, done } = state.fixture;
    const layers = availableLayers();
    const lastIndex = Math.max(0, layers.length - 1);
    state.layerStart = Math.min(state.layerStart, lastIndex);
    state.layerEnd = Math.min(Math.max(state.layerStart, state.layerEnd), lastIndex);
    const visible = visibleLayers();

    renderFixtureDisclosure();
    renderProvenancePanel();
    el["fixture-source"].textContent = `${state.source} · ${state.fixture.tokens.length} positions`;
    el["model-name"].textContent = meta.model;
    el["model-name"].title = meta.model;
    const vocab = done && Number.isFinite(done.vocab_size) ? ` · ${formatNumber(done.vocab_size)} vocab` : "";
    el["run-summary"].textContent = `${state.fixture.tokens.length} positions · ${layers.length} sampled layers${vocab}`;

    for (const tab of el["mode-tabs"].querySelectorAll("[data-mode]")) {
      const active = tab.dataset.mode === state.mode;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
    }

    for (const range of [el["layer-start"], el["layer-end"]]) {
      range.max = String(lastIndex);
      range.disabled = !lastIndex;
    }
    el["layer-start"].value = String(state.layerStart);
    el["layer-end"].value = String(state.layerEnd);
    el["layer-start-output"].value = `L${layers[state.layerStart]}`;
    el["layer-start-output"].textContent = `L${layers[state.layerStart]}`;
    el["layer-end-output"].value = `L${layers[state.layerEnd]}`;
    el["layer-end-output"].textContent = `L${layers[state.layerEnd]}`;
    el["layer-range-label"].textContent = `L${visible[0]}–L${visible[visible.length - 1]} · ${visible.length} sampled`;

    el["top-n"].max = String(state.maxTopN);
    el["top-n"].value = String(state.topN);
    el["top-n-limit"].textContent = `of ${state.maxTopN}`;
  }

  function renderFixtureDisclosure() {
    const fixtureMeta = state.fixture._fixture || {};
    const mode =
      fixtureMeta.mode === "illustrative" ? "illustrative" : fixtureMeta.mode === "recorded" ? "recorded" : "fixture";
    el["fixture-pill"].dataset.mode = mode;
    if (mode === "illustrative") {
      el["fixture-pill"].dataset.acceptance = "illustrative";
      el["fixture-status"].textContent = "Illustrative fixture";
      el["fixture-pill"].title = "Hand-authored UI data; not a Nemotron measurement";
      el["fixture-notice-title"].textContent = "Illustrative UI sample — not model output";
      el["fixture-notice-body"].textContent =
        fixtureMeta.description || "Every displayed readout is hand-authored and must not be used as scientific evidence.";
    } else if (mode === "recorded") {
      const summary = Provenance.acceptanceSummary(state.fixture);
      const nPrompts = fixtureMeta.provenance.lens.n_prompts;
      const underScale = nPrompts < PAPER_SCALE_PROMPTS;
      el["fixture-pill"].dataset.acceptance = summary.badgeStatus;
      el["fixture-status"].textContent = summary.finalDefaultEligible
        ? "Accepted · paper-scale"
        : summary.tier === "undeclared"
          ? `Recorded · ${formatNumber(nPrompts)} prompts`
          : `Recorded · ${summary.tier}`;
      el["fixture-pill"].title = "Replaying imported model readouts; no live inference is running";
      if (summary.finalDefaultEligible) {
        el["fixture-notice-title"].textContent = "Accepted paper-scale recorded replay";
        el["fixture-notice-body"].textContent =
          `The exported metadata marks this ${formatNumber(nPrompts)}-prompt artifact paper-scale and accepted. ` +
          "The browser displays that declaration but has not independently verified the run, checksums, or scientific result.";
      } else if (underScale) {
        el["fixture-notice-title"].textContent = "Recorded under-scale replay — not a final result";
        el["fixture-notice-body"].textContent =
          `These are model readouts from a lens fitted on ${formatNumber(nPrompts)} prompts. ` +
          `They can validate the pipeline and interface, but are not the paper-scale ${formatNumber(PAPER_SCALE_PROMPTS)}-prompt result or evidence for directed modulation. ` +
          "The browser checked declaration shape and pinned identifiers; it did not independently verify the run or provenance.";
      } else {
        el["fixture-notice-title"].textContent = "Recorded replay — not accepted as final";
        el["fixture-notice-body"].textContent =
          `This recording is ${summary.tier === "undeclared" ? "missing acceptance metadata" : `marked ${summary.tier} / ${summary.status}`}. ` +
          "It remains opt-in and cannot replace the illustrative default. The browser has not independently verified its claims.";
      }
    } else {
      el["fixture-pill"].dataset.acceptance = "undeclared";
      el["fixture-status"].textContent = "Fixture replay";
      el["fixture-pill"].title = "Fixture provenance was not declared";
      el["fixture-notice-title"].textContent = "Fixture with undeclared provenance";
      el["fixture-notice-body"].textContent =
        "No live inference is running. Treat these values as unverified until provenance is supplied.";
    }
  }

  function renderProvenancePanel() {
    const recorded = Provenance.validateRecordedProvenance(state.fixture);
    el["provenance-panel"].hidden = !recorded;
    if (!recorded) return;

    const summary = Provenance.acceptanceSummary(state.fixture);
    const { info, lens } = recorded;
    el["acceptance-badge"].textContent = summary.label;
    el["acceptance-badge"].dataset.status = summary.badgeStatus;
    el["provenance-model-revision"].textContent = info.model_revision;
    el["provenance-lens-sha"].textContent = lens.sha256;
    el["provenance-corpus-sha"].textContent = lens.corpus_manifest_sha256;
    el["provenance-prompt-count"].textContent = formatNumber(lens.n_prompts);
    el["provenance-target-layer"].textContent = `L${lens.target_layer}`;
    el["provenance-source-layers"].textContent = formatLayerSet(lens.source_layers);
    el["provenance-acceptance-tier"].textContent = summary.tier;
    el["provenance-acceptance-status"].textContent = summary.status;

    const reasons = el["acceptance-reasons"];
    reasons.hidden = summary.reasons.length === 0;
    reasons.textContent = summary.reasons.length ? `Reasons: ${summary.reasons.join(" · ")}` : "";

    const checks = el["acceptance-checks"];
    checks.replaceChildren();
    for (const check of summary.checks) {
      const item = document.createElement("li");
      const label = document.createElement("span");
      const status = document.createElement("strong");
      label.textContent = check.label;
      status.textContent = check.status;
      item.dataset.status = check.status.toLocaleLowerCase();
      item.append(label, status);
      if (check.detail) {
        const detail = document.createElement("small");
        detail.textContent = check.detail;
        item.append(detail);
      }
      checks.append(item);
    }
    checks.hidden = summary.checks.length === 0;
    el["acceptance-details"].hidden = summary.reasons.length === 0 && summary.checks.length === 0;
  }

  function formatLayerSet(layers) {
    const sorted = [...layers].sort((left, right) => left - right);
    const ranges = [];
    let start = sorted[0];
    let end = start;
    for (const layer of sorted.slice(1)) {
      if (layer === end + 1) {
        end = layer;
      } else {
        ranges.push(start === end ? `L${start}` : `L${start}–L${end}`);
        start = layer;
        end = layer;
      }
    }
    ranges.push(start === end ? `L${start}` : `L${start}–L${end}`);
    return `${ranges.join(", ")} · ${formatNumber(sorted.length)} total`;
  }

  function availableLayers() {
    if (!state.fixture) return [];
    const byType = state.fixture.meta.layers_by_type;
    const types = state.mode === JACOBIAN ? [JACOBIAN] : state.mode === LOGIT ? [LOGIT] : [JACOBIAN, LOGIT];
    return [...new Set(types.flatMap((type) => byType[type] || []))].sort((a, b) => a - b);
  }

  function visibleLayers() {
    const layers = availableLayers();
    return layers.slice(state.layerStart, state.layerEnd + 1);
  }

  function renderTranscript() {
    el["transcript-content"].replaceChildren();
    const segments = transcriptSegments();
    for (const segment of segments) {
      const turn = document.createElement("div");
      turn.className = `transcript-turn${segment.generated ? " generated" : ""}`;
      const label = document.createElement("div");
      label.className = "turn-label";
      label.textContent = segment.label;
      const run = document.createElement("div");
      run.className = "token-run";

      for (const token of segment.tokens) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = `source-token${token.is_generated ? " generated" : ""}`;
        button.dataset.position = String(token.position);
        button.textContent = displayToken(token.token);
        button.setAttribute(
          "aria-label",
          `Position ${token.position}, ${token.is_generated ? "generated" : "prompt"} token ${spokenToken(token.token)}`,
        );
        button.setAttribute("aria-pressed", "false");
        button.addEventListener("mouseenter", () => {
          state.hoveredPosition = token.position;
          showTooltip(button, `Position ${token.position} · ${token.is_generated ? "generated" : "prompt"} · ${spokenToken(token.token)}`);
          updateTranscriptStates();
          renderAnalysis();
        });
        button.addEventListener("mousemove", () => positionTooltip(button));
        button.addEventListener("mouseleave", () => {
          state.hoveredPosition = null;
          hideTooltip();
          updateTranscriptStates();
          renderAnalysis();
        });
        button.addEventListener("focus", () => {
          state.hoveredPosition = token.position;
          updateTranscriptStates();
          renderAnalysis();
        });
        button.addEventListener("blur", () => {
          state.hoveredPosition = null;
          updateTranscriptStates();
          renderAnalysis();
        });
        button.addEventListener("click", (event) => selectPosition(token.position, event.shiftKey));
        run.append(button);
      }
      turn.append(label, run);
      el["transcript-content"].append(turn);
    }
    updateTranscriptStates();
  }

  function transcriptSegments() {
    const configured = state.fixture._fixture && state.fixture._fixture.transcript;
    if (Array.isArray(configured) && configured.length) {
      return configured
        .map((segment) => ({
          label: String(segment.label || "Sequence"),
          generated: Boolean(segment.generated),
          tokens: state.fixture.tokens.filter(
            (token) => token.position >= Number(segment.start) && token.position <= Number(segment.end),
          ),
        }))
        .filter((segment) => segment.tokens.length);
    }
    const prompt = state.fixture.tokens.filter((token) => !token.is_generated);
    const generated = state.fixture.tokens.filter((token) => token.is_generated);
    return [
      { label: "Prompt", generated: false, tokens: prompt },
      { label: "Generated completion", generated: true, tokens: generated },
    ].filter((segment) => segment.tokens.length);
  }

  function selectPosition(position, additive) {
    if (additive) {
      if (state.selectedPositions.has(position)) state.selectedPositions.delete(position);
      else state.selectedPositions.add(position);
    } else if (state.selectedPositions.size === 1 && state.selectedPositions.has(position)) {
      state.selectedPositions.clear();
    } else {
      state.selectedPositions.clear();
      state.selectedPositions.add(position);
    }
    updateTranscriptStates();
    renderAnalysis();
  }

  function updateTranscriptStates() {
    if (!state.fixture) return;
    const highlights = highlightedTokens();
    for (const button of el["transcript-content"].querySelectorAll(".source-token")) {
      const position = Number(button.dataset.position);
      const selected = state.selectedPositions.has(position);
      button.classList.toggle("is-selected", selected);
      button.classList.toggle("is-hovered", state.hoveredPosition === position);
      button.setAttribute("aria-pressed", String(selected));
      const token = tokenAt(position);
      const match = highlights.find((highlight) => token && tokenContainsPrediction(token, highlight.key));
      button.classList.toggle("token-match", Boolean(match));
      if (match) button.style.setProperty("--match-color", match.color);
      else button.style.removeProperty("--match-color");
    }

    const count = state.selectedPositions.size;
    el["selection-summary"].textContent = count ? `${count} position${count === 1 ? "" : "s"} pinned` : "All positions";
    el["clear-selection"].hidden = !count;
  }

  function renderAnalysis() {
    if (!state.fixture) return;
    renderControls();
    renderLockedTokens();
    renderSidebar();
    renderReadout();
  }

  function focusTokens() {
    if (state.hoveredPosition !== null) {
      const token = tokenAt(state.hoveredPosition);
      return token ? [token] : [];
    }
    if (state.selectedPositions.size) {
      return state.fixture.tokens.filter((token) => state.selectedPositions.has(token.position));
    }
    return state.fixture.tokens;
  }

  function renderSidebar() {
    const scope = focusTokens();
    const layers = visibleLayers();
    const rows = prioritizeLockedRows(aggregateRows(scope, layers)).slice(0, 48);
    const modeLabel = state.mode === JACOBIAN ? "Jacobian tokens" : state.mode === LOGIT ? "Logit tokens" : "Largest differences";
    el["sidebar-title"].textContent = modeLabel;
    el["sidebar-count"].textContent = String(rows.length);
    el["layer-axis-label"].textContent = layers.length === 1 ? `L${layers[0]}` : `L${layers[0]} → L${layers[layers.length - 1]}`;
    if (state.hoveredPosition !== null) {
      el["sidebar-description"].textContent = `Readouts for hovered position ${state.hoveredPosition}, ranked over ${layers.length} sampled layer${layers.length === 1 ? "" : "s"}.`;
    } else if (state.selectedPositions.size) {
      el["sidebar-description"].textContent = `Ranked across ${state.selectedPositions.size} pinned positions and ${layers.length} sampled layers.`;
    } else {
      el["sidebar-description"].textContent = `Ranked across all positions and ${layers.length} visible sampled layers.`;
    }

    el["token-list"].replaceChildren();
    if (!rows.length) {
      const empty = document.createElement("p");
      empty.className = "empty-readout";
      empty.textContent = "No readouts in this range.";
      el["token-list"].append(empty);
      return;
    }

    for (const row of rows) {
      const lockedIndex = state.lockedTokens.findIndex((token) => token.key === row.key);
      const button = document.createElement("button");
      button.type = "button";
      button.className = `token-row${lockedIndex >= 0 ? " is-locked" : ""}${row.differenceUnavailable ? " is-unavailable" : ""}`;
      button.style.setProperty("--layer-count", String(layers.length));
      if (lockedIndex >= 0) button.style.setProperty("--lock-color", LOCK_COLORS[lockedIndex]);
      button.setAttribute("aria-pressed", String(lockedIndex >= 0));
      button.setAttribute("aria-label", `${lockedIndex >= 0 ? "Unlock" : "Lock"} token ${spokenToken(row.display)}. ${row.detail}`);

      const identity = document.createElement("span");
      identity.className = "token-identity";
      const marker = document.createElement("i");
      marker.className = "lock-marker";
      marker.setAttribute("aria-hidden", "true");
      const copy = document.createElement("span");
      copy.className = "token-copy";
      const name = document.createElement("strong");
      name.textContent = displayToken(row.display);
      name.title = spokenToken(row.display);
      const detail = document.createElement("small");
      detail.textContent = row.detail;
      copy.append(name, detail);
      identity.append(marker, copy);

      const bars = document.createElement("span");
      bars.className = "layer-bars";
      const maxLayerValue = Math.max(0.0001, ...row.byLayer.map((value) => Math.abs(value)));
      row.byLayer.forEach((value, index) => {
        const bar = document.createElement("span");
        bar.className = `layer-bar${value < 0 ? " is-negative" : ""}`;
        const fill = document.createElement("i");
        fill.style.setProperty("--weight", String(Math.abs(value) / maxLayerValue));
        bar.title = row.layerDetails?.[index] || `Layer ${layers[index]} · ${state.mode === DIFF ? signed(value) : value.toFixed(2)}`;
        bar.append(fill);
        bars.append(bar);
      });
      button.append(identity, bars);

      button.addEventListener("mouseenter", () => {
        state.previewToken = { key: row.key, display: row.display };
        button.classList.add("is-previewed");
        updateTranscriptStates();
      });
      button.addEventListener("mouseleave", () => {
        state.previewToken = null;
        button.classList.remove("is-previewed");
        updateTranscriptStates();
      });
      button.addEventListener("focus", () => {
        state.previewToken = { key: row.key, display: row.display };
        updateTranscriptStates();
      });
      button.addEventListener("blur", () => {
        state.previewToken = null;
        updateTranscriptStates();
      });
      button.addEventListener("click", () => toggleTokenLock(row));
      el["token-list"].append(button);
    }
  }

  function aggregateRows(tokens, layers) {
    if (state.mode === DIFF) {
      const jacobian = aggregateType(JACOBIAN, tokens, layers);
      const logit = aggregateType(LOGIT, tokens, layers);
      const keys = new Set([...jacobian.keys(), ...logit.keys()]);
      return [...keys]
        .map((key) => {
          const a = materializeEntry(key, jacobian.get(key), layers);
          const b = materializeEntry(key, logit.get(key), layers);
          const hasTrackedRow = a.hasTrackedRow || b.hasTrackedRow;
          const exactProbabilityDifference =
            a.hasTrackedRow &&
            b.hasTrackedRow &&
            layers.every(
              (_, index) =>
                a.probabilityCounts[index] > 0 &&
                a.probabilityCounts[index] === b.probabilityCounts[index],
            );
          if (hasTrackedRow && !exactProbabilityDifference) {
            return {
              key,
              display: a.display || b.display || key,
              score: 0,
              byLayer: layers.map(() => 0),
              detail: "tracked Δp unavailable · exact probabilities are missing or unmatched",
              layerDetails: layers.map((layer) => `Layer ${layer} · tracked Δp unavailable`),
              isTracked: true,
              differenceUnavailable: true,
            };
          }
          const score = a.score - b.score;
          const tracked = exactProbabilityDifference;
          return {
            key,
            display: a.display || b.display || key,
            score,
            byLayer: layers.map((_, index) => a.byLayer[index] - b.byLayer[index]),
            detail: tracked
              ? `tracked Δp ${signedPercent(score)} · J ${formatProbability(a.maxProbability)} · L ${formatProbability(b.maxProbability)}`
              : `${score >= 0 ? "Jacobian" : "Logit"} ${signed(score)} · J${a.occurrences}:L${b.occurrences}`,
            layerDetails: tracked
              ? layers.map(
                  (layer, index) =>
                    `Layer ${layer} · Δp ${signedPercent(a.byLayer[index] - b.byLayer[index])} · J ${rankLabel(a.ranks[index])} · L ${rankLabel(b.ranks[index])}`,
                )
              : null,
            isTracked: tracked,
            differenceUnavailable: false,
          };
        })
        .filter((row) => Math.abs(row.score) > 0.00001 || row.isTracked)
        .sort((a, b) => Math.abs(b.score) - Math.abs(a.score) || a.display.localeCompare(b.display));
    }

    return [...aggregateType(state.mode, tokens, layers).entries()]
      .map(([key, entry]) => materializeEntry(key, entry, layers))
      .sort((a, b) => b.score - a.score || a.display.localeCompare(b.display));
  }

  function materializeEntry(key, entry, layers) {
    if (!entry) {
      return {
        key,
        display: "",
        score: 0,
        byLayer: layers.map(() => 0),
        detail: "not retained",
        layerDetails: null,
        isTracked: false,
        hasTrackedRow: false,
        probabilityCounts: layers.map(() => 0),
        ranks: layers.map(() => null),
        maxProbability: 0,
        occurrences: 0,
      };
    }
    const tracked = Boolean(entry.hasTrackedRow);
    if (!tracked) {
      return {
        key,
        display: entry.display,
        score: entry.total,
        byLayer: entry.byLayer,
        detail: `${entry.occurrences} hit${entry.occurrences === 1 ? "" : "s"} · score ${entry.total.toFixed(2)}`,
        layerDetails: null,
        isTracked: false,
        hasTrackedRow: false,
        probabilityCounts: layers.map(() => 0),
        ranks: layers.map(() => null),
        maxProbability: 0,
        occurrences: entry.occurrences,
      };
    }
    const byLayer = layers.map((_, index) =>
      entry.trackedCounts[index] ? entry.trackedProbs[index] / entry.trackedCounts[index] : 0,
    );
    const ranks = entry.trackedRanks;
    const probabilityCounts = entry.trackedCounts;
    const bestRank = ranks.filter(Number.isInteger).reduce((best, rank) => Math.min(best, rank), Number.POSITIVE_INFINITY);
    const maxProbability = probabilityCounts.some((count) => count > 0) ? Math.max(...byLayer) : null;
    return {
      key,
      display: entry.display,
      score: byLayer.reduce((total, probability) => total + probability, 0),
      byLayer,
      detail: `tracked · max p ${formatProbability(maxProbability)} · best ${rankLabel(bestRank)}`,
      layerDetails: layers.map(
        (layer, index) =>
          `Layer ${layer} · p ${entry.trackedCounts[index] ? formatProbability(byLayer[index]) : "n/a"} · ${rankLabel(ranks[index])}`,
      ),
      isTracked: true,
      hasTrackedRow: true,
      probabilityCounts,
      ranks,
      maxProbability,
      occurrences: entry.occurrences,
    };
  }

  function aggregateType(type, tokens, layers) {
    const aggregate = new Map();
    for (const token of tokens) {
      const result = resultFor(token, type);
      if (!result) continue;
      layers.forEach((layer, visibleIndex) => {
        const sourceIndex = layerIndex(type, layer);
        if (sourceIndex < 0) return;
        const predictions = result.top_tokens[sourceIndex] || [];
        predictions.slice(0, state.topN).forEach((prediction, rank) => {
          const key = tokenKey(prediction);
          if (!key) return;
          const rankWeight = (state.topN - rank) / state.topN;
          if (!aggregate.has(key)) {
            aggregate.set(key, {
              display: prediction,
              total: 0,
              occurrences: 0,
              byLayer: layers.map(() => 0),
            });
          }
          const entry = aggregate.get(key);
          entry.total += rankWeight;
          entry.occurrences += 1;
          entry.byLayer[visibleIndex] += rankWeight;
        });
      });
      for (const locked of state.lockedTokens) {
        const tracked = trackedEntry(result, locked.key);
        if (!tracked) continue;
        if (!aggregate.has(locked.key)) {
          aggregate.set(locked.key, {
            display: tracked.token || locked.display,
            total: 0,
            occurrences: 0,
            byLayer: layers.map(() => 0),
          });
        }
        const entry = aggregate.get(locked.key);
        entry.hasTrackedRow = true;
        entry.trackedProbs ||= layers.map(() => 0);
        entry.trackedCounts ||= layers.map(() => 0);
        entry.trackedRanks ||= layers.map(() => null);
        layers.forEach((layer, visibleIndex) => {
          const sourceIndex = layerIndex(type, layer);
          if (sourceIndex < 0) return;
          const probability = tracked.probs[sourceIndex];
          const rank = tracked.ranks[sourceIndex];
          if (probability !== null) {
            entry.trackedProbs[visibleIndex] += probability;
            entry.trackedCounts[visibleIndex] += 1;
          }
          if (rank !== null) {
            entry.trackedRanks[visibleIndex] = Math.min(entry.trackedRanks[visibleIndex] || rank, rank);
          }
        });
      }
    }
    return aggregate;
  }

  function trackedEntry(result, key) {
    if (!result.tracked) return null;
    if (Array.isArray(result.tracked)) {
      return result.tracked.find((entry) => tokenKey(entry.token || "") === key) || null;
    }
    for (const [mapKey, entry] of Object.entries(result.tracked)) {
      const label = entry.token === undefined ? mapKey : entry.token;
      if (tokenKey(label) === key) return entry.token === undefined ? { ...entry, token: mapKey } : entry;
    }
    return null;
  }

  function prioritizeLockedRows(rows) {
    const locked = new Map(state.lockedTokens.map((token, index) => [token.key, index]));
    return [...rows].sort((a, b) => {
      const aIndex = locked.has(a.key) ? locked.get(a.key) : Number.POSITIVE_INFINITY;
      const bIndex = locked.has(b.key) ? locked.get(b.key) : Number.POSITIVE_INFINITY;
      return aIndex - bIndex;
    });
  }

  function formatProbability(value) {
    if (!Number.isFinite(value)) return "n/a";
    const percent = value * 100;
    const decimals = percent > 0 && percent < 0.1 ? 3 : 2;
    return `${percent.toFixed(decimals)}%`;
  }

  function signedPercent(value) {
    if (!Number.isFinite(value)) return "n/a";
    return `${value >= 0 ? "+" : "−"}${formatProbability(Math.abs(value))}`;
  }

  function rankLabel(value) {
    return Number.isInteger(value) && value >= 1 ? `rank #${formatNumber(value)}` : "rank n/a";
  }

  function renderLockedTokens() {
    el["locked-tokens"].replaceChildren();
    state.lockedTokens.forEach((token, index) => {
      const chip = document.createElement("span");
      chip.className = "locked-token";

      chip.style.setProperty("--lock-color", LOCK_COLORS[index]);
      const text = document.createElement("span");
      text.textContent = displayToken(token.display);
      const remove = document.createElement("button");
      remove.type = "button";
      remove.textContent = "×";
      remove.setAttribute("aria-label", `Unlock ${spokenToken(token.display)}`);
      remove.addEventListener("click", () => {
        state.lockedTokens.splice(index, 1);
        renderLockedTokens();
        renderSidebar();
        updateTranscriptStates();
      });
      chip.append(text, remove);
      el["locked-tokens"].append(chip);
    });
  }

  function toggleTokenLock(row) {
    const index = state.lockedTokens.findIndex((token) => token.key === row.key);
    if (index >= 0) {
      state.lockedTokens.splice(index, 1);
      showToast(`Unlocked ${spokenToken(row.display)}.`);
    } else {
      if (state.lockedTokens.length === MAX_LOCKS) {
        const [removed] = state.lockedTokens.splice(0, 1);
        showToast(`Replaced ${spokenToken(removed.display)} with ${spokenToken(row.display)}.`);
      } else {
        showToast(`Locked ${spokenToken(row.display)}.`);
      }
      state.lockedTokens.push({ key: row.key, display: row.display });
    }
    state.previewToken = null;
    renderLockedTokens();
    renderSidebar();
    updateTranscriptStates();
  }

  function highlightedTokens() {
    const locked = state.lockedTokens.map((token, index) => ({ ...token, color: LOCK_COLORS[index] }));
    if (state.previewToken && !locked.some((token) => token.key === state.previewToken.key)) {
      locked.push({ ...state.previewToken, color: LOCK_COLORS[Math.min(locked.length, LOCK_COLORS.length - 1)] });
    }
    return locked;
  }

  function tokenContainsPrediction(sourceToken, targetKey) {
    const types = state.mode === DIFF ? [JACOBIAN, LOGIT] : [state.mode];
    for (const type of types) {
      const result = resultFor(sourceToken, type);
      if (!result) continue;
      for (const layer of visibleLayers()) {
        const index = layerIndex(type, layer);
        if (index < 0) continue;
        if ((result.top_tokens[index] || []).slice(0, state.topN).some((token) => tokenKey(token) === targetKey)) return true;
      }
    }
    return false;
  }

  function renderReadout() {
    const scope = focusTokens();
    const layers = visibleLayers();
    el["layer-readout"].replaceChildren();
    if (state.hoveredPosition !== null) {
      const token = tokenAt(state.hoveredPosition);
      el["readout-position"].textContent = `P${state.hoveredPosition} · ${token ? displayToken(token.token) : ""}`;
    } else if (state.selectedPositions.size === 1) {
      const [position] = state.selectedPositions;
      const token = tokenAt(position);
      el["readout-position"].textContent = `P${position} · ${token ? displayToken(token.token) : ""}`;
    } else if (state.selectedPositions.size > 1) {
      el["readout-position"].textContent = `${state.selectedPositions.size} positions`;
    } else {
      el["readout-position"].textContent = "All positions";
    }

    for (const layer of layers) {
      const rows = prioritizeLockedRows(aggregateRows(scope, [layer])).slice(0, state.topN);
      const column = document.createElement("article");
      column.className = "layer-column";
      const heading = document.createElement("h3");
      heading.textContent = `Layer ${layer}`;
      const descriptor = document.createElement("span");
      descriptor.textContent = state.mode === DIFF ? "Δ rank" : "rank score";
      heading.append(descriptor);
      const stack = document.createElement("div");
      stack.className = "prediction-stack";
      const max = Math.max(0.0001, ...rows.map((row) => Math.abs(row.score)));
      for (const row of rows) {
        const prediction = document.createElement("div");
        prediction.className = `prediction${row.score < 0 ? " negative" : ""}${row.differenceUnavailable ? " unavailable" : ""}`;
        prediction.title = row.layerDetails?.[0] || `${spokenToken(row.display)} · ${state.mode === DIFF ? signed(row.score) : row.score.toFixed(2)}`;
        const name = document.createElement("span");
        name.className = "prediction-token";
        name.textContent = displayToken(row.display);
        const meter = document.createElement("span");
        meter.className = "prediction-meter";
        const fill = document.createElement("i");
        fill.style.setProperty(
          "--value",
          row.differenceUnavailable ? "0%" : `${Math.max(4, (Math.abs(row.score) / max) * 100)}%`,
        );
        meter.append(fill);
        prediction.append(name, meter);
        stack.append(prediction);
      }
      column.append(heading, stack);
      el["layer-readout"].append(column);
    }
  }

  function resultFor(token, type) {
    return token.results.find((result) => result.type === type);
  }

  function layerIndex(type, layer) {
    return (state.fixture.meta.layers_by_type[type] || []).indexOf(layer);
  }

  function tokenAt(position) {
    return state.fixture.tokens.find((token) => token.position === position);
  }

  function tokenKey(token) {
    return String(token).trim().toLocaleLowerCase();
  }

  function displayToken(token) {
    const value = String(token);
    if (value === "\n") return "↵";
    if (/^\s+$/.test(value)) return value.replace(/\n/g, "↵").replace(/ /g, "·");
    const lineSafe = value.replace(/\n/g, "↵");
    return /^\s/.test(value) ? `▁${lineSafe.trimStart()}` : lineSafe;
  }

  function spokenToken(token) {
    const value = String(token);
    if (value === "\n") return "newline";
    return value.replace(/\n/g, " newline ").trim() || "whitespace";
  }

  function signed(value) {
    return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(2)}`;
  }

  function formatNumber(value) {
    return new Intl.NumberFormat("en", { notation: value >= 100000 ? "compact" : "standard" }).format(value);
  }

  function showTooltip(target, text) {
    el.tooltip.textContent = text;
    el.tooltip.hidden = false;
    positionTooltip(target);
  }

  function positionTooltip(target) {
    if (el.tooltip.hidden) return;
    const rect = target.getBoundingClientRect();
    const tooltipRect = el.tooltip.getBoundingClientRect();
    const left = Math.min(window.innerWidth - tooltipRect.width - 8, Math.max(8, rect.left + rect.width / 2 - tooltipRect.width / 2));
    const top = rect.top > tooltipRect.height + 12 ? rect.top - tooltipRect.height - 8 : rect.bottom + 8;
    el.tooltip.style.left = `${left}px`;
    el.tooltip.style.top = `${top}px`;
  }

  function hideTooltip() {
    el.tooltip.hidden = true;
  }

  function showToast(message) {
    window.clearTimeout(toastTimer);
    el.toast.textContent = message;
    el.toast.classList.add("show");
    toastTimer = window.setTimeout(() => el.toast.classList.remove("show"), 2600);
  }
})();
