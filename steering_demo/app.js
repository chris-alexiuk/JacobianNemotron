(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  if (root && root.document) {
    root.addEventListener("DOMContentLoaded", api.mount);
  }
})(typeof window !== "undefined" ? window : null, function () {
  "use strict";

  const DISCLOSURE = "100-prompt accepted pilot — exploratory, non-final";
  const READOUT_LAYER_OPTIONS = { min: 0, max: 51, limit: 52 };
  const INTERVENTION_LAYER_OPTIONS = { min: 0, max: 50, limit: 51 };
  const DEFAULT_READOUT_LAYERS = Array.from({ length: 38 }, (_, index) => index + 13);
  const MODE_NOTES = {
    steer: "Strength applies.",
    ablate: "Strength ignored.",
    swap: "Activation-direction swap; prompt tokens remain unchanged.",
  };

  function assertInteger(value, field) {
    if (!Number.isInteger(value)) {
      throw new TypeError(`${field} must remain an integer`);
    }
    return value;
  }

  function normalizeToken(raw) {
    if (!raw || typeof raw !== "object") {
      throw new TypeError("token must be an object");
    }
    return {
      position: assertInteger(raw.position, "token.position"),
      id: assertInteger(raw.id, "token.id"),
      text: typeof raw.text === "string" ? raw.text : "",
      is_generated: raw.is_generated === true,
      is_bos: raw.is_bos === true,
    };
  }

  function normalizeReadoutEntry(raw) {
    if (!raw || typeof raw !== "object") {
      throw new TypeError("readout entry must be an object");
    }
    return {
      id: assertInteger(raw.id, "readout.id"),
      text: typeof raw.text === "string" ? raw.text : "",
      probability: Number.isFinite(raw.probability) ? raw.probability : null,
      logit: Number.isFinite(raw.logit) ? raw.logit : null,
    };
  }

  function conditionViewModel(condition) {
    if (!condition || typeof condition !== "object") {
      throw new TypeError("condition must be an object");
    }
    return {
      name: typeof condition.name === "string" ? condition.name : "condition",
      completion: typeof condition.completion === "string" ? condition.completion : "",
      elapsed_seconds: Number.isFinite(condition.elapsed_seconds) ? condition.elapsed_seconds : null,
      tokens: Array.isArray(condition.tokens) ? condition.tokens.map(normalizeToken) : [],
      readouts: condition.readouts && typeof condition.readouts === "object" ? condition.readouts : {},
    };
  }

  function readoutEntries(condition, lensType, layer, position) {
    const byLens = condition && condition.readouts && condition.readouts[lensType];
    const byLayer = byLens && (byLens[String(layer)] || byLens[layer]);
    const row = Array.isArray(byLayer) ? byLayer[position] : null;
    return Array.isArray(row) ? row.map(normalizeReadoutEntry) : [];
  }

  function selectedPositions(condition, selection) {
    const tokens = condition && Array.isArray(condition.tokens) ? condition.tokens : [];
    const mode = selection && typeof selection.mode === "string" ? selection.mode : "all";
    const available = new Set(tokens.map((token) => token.position));
    if (mode === "prompt") {
      return tokens.filter((token) => !token.is_generated).map((token) => token.position);
    }
    if (mode === "generated") {
      return tokens.filter((token) => token.is_generated).map((token) => token.position);
    }
    if (mode === "custom") {
      const custom = selection && Array.isArray(selection.custom) ? selection.custom : [];
      return Array.from(new Set(custom.filter((position) => Number.isInteger(position) && available.has(position))))
        .sort((left, right) => left - right);
    }
    return tokens.map((token) => token.position);
  }

  function betterOccurrence(candidate, current) {
    if (!current) return true;
    const candidateProbability = Number.isFinite(candidate.entry.probability) ? candidate.entry.probability : -Infinity;
    const currentProbability = Number.isFinite(current.entry.probability) ? current.entry.probability : -Infinity;
    return candidateProbability > currentProbability
      || (candidateProbability === currentProbability && candidate.rank < current.rank)
      || (candidateProbability === currentProbability && candidate.rank === current.rank && candidate.position < current.position)
      || (candidateProbability === currentProbability && candidate.rank === current.rank && candidate.position === current.position && candidate.entryIndex < current.entryIndex);
  }

  function aggregateReadoutRows(condition, lensType, layers, positions) {
    const orderedLayers = Array.from(new Set(Array.isArray(layers) ? layers.filter(Number.isInteger) : []));
    const requestedPositions = Number.isInteger(positions) ? [positions] : positions;
    const orderedPositions = Array.from(new Set(Array.isArray(requestedPositions)
      ? requestedPositions.filter(Number.isInteger)
      : []));
    const rows = new Map();
    for (const position of orderedPositions) {
      for (const layer of orderedLayers) {
        const entries = readoutEntries(condition, lensType, layer, position);
        entries.forEach((entry, entryIndex) => {
          let row = rows.get(entry.id);
          if (!row) {
            row = {
              id: entry.id,
              text: entry.text,
              occurrenceCount: 0,
              probabilitySum: 0,
              peakProbability: null,
              positionCounts: new Map(),
              cellsByLayer: new Map(),
            };
            rows.set(entry.id, row);
          }
          const probability = Number.isFinite(entry.probability) ? entry.probability : null;
          const occurrence = { position, layer, entryIndex, rank: entryIndex + 1, entry };
          let cell = row.cellsByLayer.get(layer);
          if (!cell) {
            cell = {
              layer,
              count: 0,
              probabilitySum: 0,
              peakProbability: null,
              bestOccurrence: null,
            };
            row.cellsByLayer.set(layer, cell);
          }
          row.occurrenceCount += 1;
          row.positionCounts.set(position, (row.positionCounts.get(position) || 0) + 1);
          cell.count += 1;
          if (probability !== null) {
            row.probabilitySum += probability;
            row.peakProbability = row.peakProbability === null ? probability : Math.max(row.peakProbability, probability);
            cell.probabilitySum += probability;
            cell.peakProbability = cell.peakProbability === null ? probability : Math.max(cell.peakProbability, probability);
          }
          if (betterOccurrence(occurrence, cell.bestOccurrence)) cell.bestOccurrence = occurrence;
        });
      }
    }
    return Array.from(rows.values())
      .map((row) => {
        const cells = orderedLayers.map((layer) => row.cellsByLayer.get(layer) || {
          layer,
          count: 0,
          probabilitySum: 0,
          peakProbability: null,
          bestOccurrence: null,
        });
        const writableCells = cells.filter((cell) => cell.count > 0 && cell.layer <= 50);
        const sourceCells = writableCells.length ? writableCells : cells.filter((cell) => cell.count > 0);
        const anchorCell = sourceCells.sort((left, right) => right.count - left.count || left.layer - right.layer)[0] || null;
        const anchor = anchorCell ? anchorCell.bestOccurrence : null;
        return {
          id: row.id,
          text: anchor ? anchor.entry.text : row.text,
          occurrenceCount: row.occurrenceCount,
          positionCount: row.positionCounts.size,
          layerCount: cells.filter((cell) => cell.count > 0).length,
          probabilitySum: row.probabilitySum,
          peakProbability: row.peakProbability,
          positionCounts: row.positionCounts,
          cells,
          anchor,
        };
      })
      .sort((left, right) => right.occurrenceCount - left.occurrenceCount || left.id - right.id);
  }

  function visibleWhitespace(value) {
    const text = typeof value === "string" ? value : "";
    if (text === "") return "∅";
    return text
      .replace(/ /g, "␠")
      .replace(/\t/g, "⇥")
      .replace(/\r/g, "↵")
      .replace(/\n/g, "↵")
      .replace(/[\u00a0\u202f]/g, "⍽")
      .replace(/[\u1680\u2000-\u200a\u205f\u3000]/g, "␠")
      .replace(/[\u2028\u2029]/g, "↵")
      .replace(/\u200b/g, "⟨ZWSP⟩")
      .replace(/\u200c/g, "⟨ZWNJ⟩")
      .replace(/\u200d/g, "⟨ZWJ⟩")
      .replace(/\ufeff/g, "⟨BOM⟩");
  }

  function parseLayerList(value, options) {
    const settings = Object.assign({}, READOUT_LAYER_OPTIONS, options || {});
    if (typeof value !== "string" || value.trim() === "") {
      throw new Error("Select at least one layer.");
    }
    const layers = [];
    for (const piece of value.split(",")) {
      const item = piece.trim();
      if (!item) throw new Error("Layer list contains an empty item.");
      const range = item.match(/^(\d+)\s*-\s*(\d+)$/);
      if (range) {
        const start = Number(range[1]);
        const end = Number(range[2]);
        if (start > end) throw new Error(`Layer range ${item} is reversed.`);
        for (let layer = start; layer <= end; layer += 1) layers.push(layer);
      } else if (/^\d+$/.test(item)) {
        layers.push(Number(item));
      } else {
        throw new Error(`Invalid layer item: ${item}.`);
      }
    }
    const unique = Array.from(new Set(layers)).sort((a, b) => a - b);
    if (unique.some((layer) => layer < settings.min || layer > settings.max)) {
      throw new Error(`Layers must be in ${settings.min}–${settings.max}.`);
    }
    if (unique.length > settings.limit) {
      throw new Error(`Select no more than ${settings.limit} layers.`);
    }
    return unique;
  }

  function formatLayers(layers) {
    if (!Array.isArray(layers) || layers.length === 0) return "";
    const ordered = Array.from(new Set(layers)).sort((a, b) => a - b);
    const parts = [];
    let start = ordered[0];
    let previous = ordered[0];
    for (let index = 1; index <= ordered.length; index += 1) {
      const current = ordered[index];
      if (current === previous + 1) {
        previous = current;
        continue;
      }
      parts.push(start === previous ? String(start) : `${start}-${previous}`);
      start = current;
      previous = current;
    }
    return parts.join(",");
  }

  function progressPercent(status, activeKind) {
    if (!status) return null;
    if (status.phase === "complete") return 100;
    if (["cancelled", "error", "idle"].includes(status.phase)) return 0;
    if (status.status !== "running") return null;
    if (["starting", "tokenize"].includes(status.phase)) {
      return status.phase === "starting" ? 2 : 4;
    }

    const paired = activeKind === "intervention";
    const intervened = paired && status.condition === "intervened";
    const start = paired ? (intervened ? 51 : 5) : 5;
    const end = paired ? (intervened ? 98 : 49) : 98;
    const span = end - start;
    const fraction = Number.isFinite(status.current) && Number.isFinite(status.total) && status.total > 0
      ? Math.max(0, Math.min(1, status.current / status.total))
      : 0;

    if (status.phase === "directions") return start;
    if (status.phase === "generate") return start + span * (0.05 + 0.35 * fraction);
    if (status.phase === "capture") return start + span * 0.42;
    if (status.phase === "readout") return start + span * (0.45 + 0.55 * fraction);
    return null;
  }

  function buildInterventionPayload(baseRequest, selected, controls) {
    if (!baseRequest || !selected) throw new Error("A baseline readout token is required.");
    const layers = parseLayerList(controls.layers, INTERVENTION_LAYER_OPTIONS);
    const strength = Number(controls.strength);
    if (!Number.isFinite(strength) || strength < -2 || strength > 2) {
      throw new Error("Strength must be finite and in −2…2.");
    }
    const intervention = {
      mode: controls.mode,
      lens_type: selected.lensType,
      layers,
      source_token_ids: [assertInteger(selected.entry.id, "source token ID")],
      strength,
      apply_to_generated: controls.applyToGenerated === true,
    };
    const payload = Object.assign({}, baseRequest, { intervention });
    if (controls.mode === "swap") {
      if (!controls.source || controls.source.is_single_token !== true) {
        throw new Error("Swap source must tokenize to exactly one token.");
      }
      if (!controls.target || controls.target.is_single_token !== true) {
        throw new Error("Swap target must tokenize to exactly one token.");
      }
      const sourceId = controls.source.token_ids && controls.source.token_ids[0];
      const targetId = controls.target.token_ids && controls.target.token_ids[0];
      intervention.source_token_ids = [assertInteger(sourceId, "source token ID")];
      intervention.target_token_id = assertInteger(targetId, "target token ID");
      payload.source_token_texts = [controls.source.text];
      payload.target_token_text = controls.target.text;
    }
    return payload;
  }

  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function exactTitle(value) {
    return escapeHtml(JSON.stringify(typeof value === "string" ? value : ""));
  }

  function finiteInteger(value, field, min, max) {
    const number = Number(value);
    if (!Number.isInteger(number) || number < min || number > max) {
      throw new Error(`${field} must be an integer in ${min}–${max}.`);
    }
    return number;
  }

  function mount() {
    const doc = document;
    const element = (id) => doc.getElementById(id);
    const state = {
      composerMode: "chat",
      info: null,
      health: null,
      baselineRequest: null,
      lastRequest: null,
      result: null,
      selected: null,
      sourceValidation: null,
      targetValidation: null,
      sourceValidationSequence: 0,
      targetValidationSequence: 0,
      positionSelections: {
        clean: { mode: "all", custom: [], anchor: null },
        intervened: { mode: "all", custom: [], anchor: null },
      },
      tabs: { clean: "jacobian", intervened: "jacobian" },
      inspectorCondition: "clean",
      inspectorRows: [],
      inspectorContext: null,
      sourceTrigger: null,
      interventionLayersDirty: false,
      activeRequestId: null,
      activeKind: null,
      polling: false,
      cancelRequested: false,
      toastTimer: null,
    };

    function requestId() {
      const random = typeof crypto !== "undefined" && crypto.getRandomValues
        ? Array.from(crypto.getRandomValues(new Uint32Array(2)), (part) => part.toString(36)).join("")
        : Math.random().toString(36).slice(2);
      return `web_${Date.now().toString(36)}_${random}`.slice(0, 76);
    }

    async function apiFetch(path, options) {
      const settings = Object.assign({ method: "GET", requestId: null, body: null }, options || {});
      const headers = { Accept: "application/json" };
      if (settings.body !== null) headers["Content-Type"] = "application/json";
      headers["X-Request-ID"] = settings.requestId || requestId();
      let response;
      try {
        response = await fetch(path, {
          method: settings.method,
          headers,
          body: settings.body === null ? undefined : JSON.stringify(settings.body),
          cache: "no-store",
        });
      } catch (_error) {
        throw new Error("Service is unreachable. Check the server and LAN connection.");
      }
      let data = null;
      try {
        data = await response.json();
      } catch (_error) {
        data = {};
      }
      if (!response.ok) {
        const failure = new Error(typeof data.error === "string" ? data.error : `Request failed (${response.status}).`);
        failure.status = response.status;
        failure.details = data.details;
        throw failure;
      }
      return data;
    }

    function showToast(message) {
      const toast = element("toast");
      toast.textContent = message;
      toast.hidden = false;
      clearTimeout(state.toastTimer);
      state.toastTimer = setTimeout(() => { toast.hidden = true; }, 3600);
    }

    function showError(targetId, error) {
      const target = element(targetId);
      target.textContent = error instanceof Error ? error.message : String(error);
      target.hidden = false;
    }

    function clearError(targetId) {
      element(targetId).hidden = true;
      element(targetId).textContent = "";
    }

    function renderHealth(health, failed) {
      const dot = element("statusDot");
      dot.className = "status-dot";
      if (failed) {
        dot.classList.add("is-error");
        element("serviceStatus").textContent = "Service offline";
        return;
      }
      const busy = health && health.busy;
      dot.classList.add(busy ? "is-busy" : "is-ready");
      element("serviceStatus").textContent = busy ? "Model busy" : "Model ready";
    }

    async function refreshService() {
      const dot = element("statusDot");
      dot.className = "status-dot is-checking";
      element("serviceStatus").textContent = "Checking service";
      const [healthResult, infoResult] = await Promise.allSettled([
        apiFetch("/health"),
        apiFetch("/api/info"),
      ]);
      if (healthResult.status === "fulfilled") {
        state.health = healthResult.value;
        renderHealth(state.health, false);
      } else {
        renderHealth(null, true);
      }
      if (infoResult.status === "fulfilled") {
        state.info = infoResult.value;
        renderProvenance();
        const modelId = state.info.model && state.info.model.id;
        if (modelId) element("footerModel").textContent = modelId;
      }
    }

    function setComposerMode(mode) {
      state.composerMode = mode;
      for (const name of ["raw", "chat"]) {
        const active = name === mode;
        element(`${name}Tab`).classList.toggle("is-active", active);
        element(`${name}Tab`).setAttribute("aria-selected", String(active));
        element(`${name}Tab`).tabIndex = active ? 0 : -1;
        element(`${name}Composer`).hidden = !active;
      }
    }

    function messageRow(role, content) {
      const row = doc.createElement("div");
      row.className = "message-row";
      row.innerHTML = `
        <select aria-label="Message role">
          <option value="system"${role === "system" ? " selected" : ""}>System</option>
          <option value="user"${role === "user" ? " selected" : ""}>User</option>
          <option value="assistant"${role === "assistant" ? " selected" : ""}>Assistant</option>
        </select>
        <textarea rows="2" aria-label="Message content" placeholder="Message content"></textarea>
        <button class="icon-button remove-message" type="button" title="Remove message" aria-label="Remove message">
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13M10 11v5M14 11v5" /></svg>
        </button>`;
      row.querySelector("textarea").value = content || "";
      row.querySelector(".remove-message").addEventListener("click", () => {
        if (element("messageList").children.length > 1) {
          const focusTarget = row.previousElementSibling || row.nextElementSibling;
          row.remove();
          if (focusTarget) focusTarget.querySelector("textarea").focus();
        } else {
          row.querySelector("textarea").value = "";
          row.querySelector("textarea").focus();
        }
      });
      return row;
    }

    function addMessage(role, content) {
      const row = messageRow(role || "user", content || "");
      element("messageList").appendChild(row);
      return row;
    }

    function currentRequest() {
      const layers = parseLayerList(element("readoutLayers").value, READOUT_LAYER_OPTIONS);
      const common = {
        layers,
        top_k: finiteInteger(element("topK").value, "Top tokens", 1, 16),
        max_new_tokens: finiteInteger(element("maxNewTokens").value, "New tokens", 1, 32),
        sampling: element("samplingToggle").checked,
        temperature: Number(element("temperature").value),
        top_p: Number(element("topP").value),
        seed: finiteInteger(element("seed").value, "Seed", 0, Number.MAX_SAFE_INTEGER),
      };
      if (!Number.isFinite(common.temperature) || common.temperature < 0.01 || common.temperature > 10) {
        throw new Error("Temperature must be in 0.01–10.");
      }
      if (!Number.isFinite(common.top_p) || common.top_p < 0.01 || common.top_p > 1) {
        throw new Error("Top p must be in 0.01–1.");
      }
      if (state.composerMode === "raw") {
        const prompt = element("rawPrompt").value;
        if (prompt.length === 0) throw new Error("Prompt must not be empty.");
        common.prompt = prompt;
      } else {
        const messages = Array.from(element("messageList").children).map((row) => ({
          role: row.querySelector("select").value,
          content: row.querySelector("textarea").value,
        }));
        if (messages.some((message) => message.content.length === 0)) {
          throw new Error("Every chat message must have content.");
        }
        common.messages = messages;
        common.enable_thinking = false;
      }
      return common;
    }

    function phaseLabel(status) {
      if (!status) return "Waiting for server";
      if (status.status === "busy") return "Model busy";
      if (status.status === "cancelling") return "Cancelling request";
      const condition = status.condition ? `${status.condition} · ` : "";
      if (status.phase === "generate") return `${condition}generating ${status.current || 0}/${status.total || "?"}`;
      if (status.phase === "readout") return `${condition}readouts ${status.current || 0}/${status.total || "?"}`;
      const labels = {
        starting: "Starting request",
        tokenize: "Tokenizing prompt",
        directions: `${condition}building directions`,
        capture: `${condition}capturing residuals`,
        complete: "Complete",
        cancelled: "Cancelled",
        error: "Request failed",
      };
      return labels[status.phase] || status.phase || "Running";
    }

    function updateProgress(status) {
      const track = element("progressTrack");
      const percent = progressPercent(status, state.activeKind);
      element("progressLabel").textContent = phaseLabel(status);
      track.classList.toggle("is-indeterminate", percent === null && status && status.status === "running");
      const value = percent === null ? 0 : Math.round(percent);
      track.setAttribute("aria-valuenow", String(value));
      element("progressFill").style.width = `${value}%`;
    }

    function setBusy(busy) {
      element("generateButton").disabled = busy;
      element("runInterventionButton").disabled = busy;
      element("stopButton").hidden = !busy;
    }

    async function pollStatus(id) {
      state.polling = true;
      while (state.activeRequestId === id) {
        await new Promise((resolve) => setTimeout(resolve, 700));
        if (state.activeRequestId !== id) break;
        try {
          const status = await apiFetch(`/api/status/${encodeURIComponent(id)}`, { requestId: id });
          updateProgress(status);
        } catch (_error) {
          // The primary request reports actionable failures; polling remains best effort.
        }
      }
      state.polling = false;
    }

    async function runRequest(path, body, kind) {
      if (state.activeRequestId) return;
      const id = requestId();
      state.activeRequestId = id;
      state.activeKind = kind;
      state.cancelRequested = false;
      setBusy(true);
      updateProgress({ status: "running", phase: "starting" });
      pollStatus(id);
      try {
        const result = await apiFetch(path, { method: "POST", body, requestId: id });
        state.result = result;
        state.lastRequest = body;
        if (kind === "baseline") {
          state.baselineRequest = body;
          state.selected = null;
          state.inspectorCondition = "clean";
          element("interventionPanel").hidden = true;
        } else if (result.intervened) {
          state.inspectorCondition = "intervened";
        }
        state.positionSelections.clean = { mode: "all", custom: [], anchor: null };
        state.positionSelections.intervened = { mode: "all", custom: [], anchor: null };
        element("exportButton").disabled = false;
        renderResults();
        renderProvenance();
        if (kind === "intervention" && result.intervened) {
          const completion = doc.querySelector('article[data-condition="intervened"] .completion');
          if (completion) completion.focus({ preventScroll: true });
          element("runScroll").scrollTo({ top: 0, behavior: "auto" });
        }
        updateProgress({ status: "complete", phase: "complete" });
        showToast(kind === "baseline" ? "Baseline complete." : "A/B intervention complete.");
      } catch (error) {
        showError(kind === "baseline" ? "formError" : "interventionError", error);
        updateProgress({ phase: state.cancelRequested ? "cancelled" : "error" });
      } finally {
        state.activeRequestId = null;
        state.activeKind = null;
        setBusy(false);
        refreshService();
      }
    }

    function renderPositionToken(token, selected, inScope, conditionName) {
      const classes = ["position-token"];
      if (selected) classes.push("is-selected");
      if (inScope) classes.push("is-in-scope");
      if (token.is_generated) classes.push("is-generated");
      if (token.is_bos) classes.push("is-bos");
      const flags = [token.is_generated ? "generated" : "prompt", token.is_bos ? "BOS" : ""].filter(Boolean).join(", ");
      return `<button class="${classes.join(" ")}" type="button" data-action="position" data-condition="${conditionName}" data-position="${token.position}" data-sequence-position="${token.position}" aria-pressed="${selected}" title="Exact decoded text: ${exactTitle(token.text)}" aria-label="Position ${token.position}, token ID ${token.id}, ${escapeHtml(flags)}, ${inScope ? "included in" : "excluded from"} the readout count">
        <strong>${escapeHtml(visibleWhitespace(token.text))}</strong>
        <small>#${token.position} · id ${token.id}${token.is_bos ? " · BOS" : ""}</small>
      </button>`;
    }

    function promptThread() {
      const request = state.baselineRequest || state.lastRequest;
      if (!request) return "";
      const messages = Array.isArray(request.messages)
        ? request.messages
        : [{ role: "raw", content: request.prompt || "" }];
      return messages.map((message) => {
        const role = typeof message.role === "string" ? message.role : "message";
        return `<div class="prompt-message is-${escapeHtml(role)}">
          <span>${escapeHtml(role)}</span>
          <p>${escapeHtml(message.content || "")}</p>
        </div>`;
      }).join("");
    }

    function renderConversationCondition(rawCondition, name, letter) {
      const condition = conditionViewModel(rawCondition);
      return `<article class="condition-panel" data-condition="${name}">
        <div class="condition-header">
          <div class="condition-title"><span class="condition-letter">${letter}</span><h3>${name === "clean" ? "Clean" : "Intervened"}</h3></div>
          <span class="condition-meta">${condition.tokens.length} tokens${condition.elapsed_seconds === null ? "" : ` · ${condition.elapsed_seconds.toFixed(2)}s`}</span>
        </div>
        <p class="result-disclosure">${DISCLOSURE}</p>
        <pre class="completion" tabindex="0" aria-label="${name === "clean" ? "Clean" : "Intervened"} completion">${escapeHtml(condition.completion)}</pre>
      </article>`;
    }

    function matrixTemplate(layers) {
      return `minmax(168px, 1.25fr) 54px repeat(${layers.length}, minmax(5px, 1fr))`;
    }

    function renderMatrixRow(row, conditionName, lensType, layers, rowIndex, selectedPositionCount) {
      const anchor = row.anchor;
      const selected = state.selected
        && state.selected.conditionName === conditionName
        && state.selected.lensType === lensType
        && state.selected.layer === (anchor && anchor.layer)
        && state.selected.entry.id === row.id;
      const anchorProbability = !anchor || !Number.isFinite(anchor.entry.probability)
        ? "—"
        : `${(anchor.entry.probability * 100).toFixed(anchor.entry.probability >= 0.01 ? 1 : 2)}%`;
      const countLabel = `${row.occurrenceCount} top-k ${row.occurrenceCount === 1 ? "hit" : "hits"} across ${row.positionCount} of ${selectedPositionCount} positions and ${row.layerCount} of ${layers.length} layers`;
      let label;
      if (anchor && anchor.layer <= 50) {
        label = `<button class="matrix-token top-token${selected ? " is-selected" : ""}" type="button" data-action="source" data-condition="${conditionName}" data-lens="${lensType}" data-layer="${anchor.layer}" data-position="${anchor.position}" data-entry="${anchor.entryIndex}" data-token-id="${row.id}" data-row-index="${rowIndex}" aria-pressed="${selected}" aria-label="${escapeHtml(visibleWhitespace(row.text))}, token ID ${row.id}; ${countLabel}; use position ${anchor.position}, layer ${anchor.layer} as the intervention source" title="Exact decoded text: ${exactTitle(row.text)}; ${countLabel}; representative source p ${anchorProbability} at position ${anchor.position}, layer ${anchor.layer}">
          <strong>${escapeHtml(visibleWhitespace(row.text))}</strong><small>id ${row.id} · ${row.positionCount} pos · source L${anchor.layer}</small>
        </button>`;
      } else {
        label = `<span class="matrix-token is-read-only" title="${countLabel}; read-only final block; exact decoded text: ${exactTitle(row.text)}"><strong>${escapeHtml(visibleWhitespace(row.text))}</strong><small>id ${row.id} · ${row.positionCount} pos · read only</small></span>`;
      }
      const busiestLayerCount = Math.max(1, ...row.cells.map((cell) => cell.count));
      const cells = row.cells.map((cell, index) => {
        const layer = layers[index];
        if (!cell.count) {
          return `<span class="heat-cell is-empty" title="Layer ${layer}; not returned in top-k at any selected position" aria-hidden="true"></span>`;
        }
        const best = cell.bestOccurrence;
        const heat = cell.count / busiestLayerCount;
        const probability = best && Number.isFinite(best.entry.probability) ? `${(best.entry.probability * 100).toFixed(2)}%` : "n/a";
        const logit = best && Number.isFinite(best.entry.logit) ? best.entry.logit.toFixed(3) : "n/a";
        return `<span class="heat-cell" style="--heat:${heat.toFixed(3)}" title="Layer ${layer}; ${cell.count} of ${selectedPositionCount} positions; best rank ${best ? best.rank : "n/a"} at position ${best ? best.position : "n/a"}; p ${probability}; logit ${logit}" aria-hidden="true"></span>`;
      }).join("");
      return `<div class="matrix-row${selected ? " is-selected" : ""}" data-row-index="${rowIndex}" data-token-id="${row.id}" style="--layer-count:${layers.length};grid-template-columns:${matrixTemplate(layers)}">
        ${label}<span class="count-cell" title="${countLabel}">${row.occurrenceCount}</span>${cells}
      </div>`;
    }

    function renderInspector(options) {
      if (!state.result || !state.result.clean) return;
      const settings = Object.assign({ focusPosition: null, focusSelector: null, stripScrollLeft: null }, options || {});
      const name = state.inspectorCondition === "intervened" && state.result.intervened
        ? "intervened"
        : "clean";
      state.inspectorCondition = name;
      const condition = conditionViewModel(state.result[name]);
      const selection = state.positionSelections[name];
      if (selection.mode === "custom" && selectedPositions(condition, selection).length === 0) {
        state.positionSelections[name] = { mode: "all", custom: [], anchor: null };
      }
      const activeSelection = state.positionSelections[name];
      const positionsInScope = selectedPositions(condition, activeSelection);
      const positionSet = new Set(positionsInScope);
      const customSet = new Set(activeSelection.mode === "custom" ? activeSelection.custom : []);
      let lensType = state.tabs[name];
      const available = condition.readouts && condition.readouts[lensType];
      if (!available || Object.keys(available).length === 0) lensType = "logit";
      state.tabs[name] = lensType;
      const layers = condition.readouts && Array.isArray(condition.readouts.layers)
        ? condition.readouts.layers.filter(Number.isInteger)
        : DEFAULT_READOUT_LAYERS;
      const rows = aggregateReadoutRows(condition, lensType, layers, positionsInScope).slice(0, 128);
      state.inspectorRows = rows;
      state.inspectorContext = { conditionName: name, lensType, layers, positions: positionsInScope };
      const conditionTabs = state.result.intervened
        ? `<div class="condition-tabs" role="tablist" aria-label="Result condition">
            <button type="button" class="condition-tab${name === "clean" ? " is-active" : ""}" data-action="condition-tab" data-condition="clean" role="tab" aria-selected="${name === "clean"}" tabindex="${name === "clean" ? "0" : "-1"}"><span>A</span> Clean</button>
            <button type="button" class="condition-tab${name === "intervened" ? " is-active" : ""}" data-action="condition-tab" data-condition="intervened" role="tab" aria-selected="${name === "intervened"}" tabindex="${name === "intervened" ? "0" : "-1"}"><span>B</span> Intervened</button>
          </div>`
        : `<div class="condition-tabs is-single"><span class="condition-tab is-active"><span>A</span> Clean</span></div>`;
      const header = `<div class="inspector-controls">
        ${conditionTabs}
        <div class="readout-tabs" role="tablist" aria-label="Readout type">
          <button class="readout-tab${lensType === "jacobian" ? " is-active" : ""}" type="button" data-action="lens-tab" data-condition="${name}" data-lens="jacobian" role="tab" aria-selected="${lensType === "jacobian"}" tabindex="${lensType === "jacobian" ? "0" : "-1"}">Jacobian</button>
          <button class="readout-tab${lensType === "logit" ? " is-active" : ""}" type="button" data-action="lens-tab" data-condition="${name}" data-lens="logit" role="tab" aria-selected="${lensType === "logit"}" tabindex="${lensType === "logit" ? "0" : "-1"}">Logit</button>
        </div>
      </div>`;
      const scopeLabel = activeSelection.mode === "custom"
        ? `${positionsInScope.length} ${positionsInScope.length === 1 ? "position" : "positions"} selected`
        : `${positionsInScope.length} ${activeSelection.mode === "all" ? "total" : activeSelection.mode} ${positionsInScope.length === 1 ? "position" : "positions"}`;
      const positions = `<div class="position-header">
          <label class="position-scope-control"><span>Readout positions</span>
            <select data-action="position-scope" data-condition="${name}" aria-label="Readout position scope">
              <option value="all"${activeSelection.mode === "all" ? " selected" : ""}>All positions</option>
              <option value="prompt"${activeSelection.mode === "prompt" ? " selected" : ""}>Prompt only</option>
              <option value="generated"${activeSelection.mode === "generated" ? " selected" : ""}>Generated only</option>
              ${activeSelection.mode === "custom" ? `<option value="custom" selected>${positionsInScope.length} selected</option>` : ""}
            </select>
          </label>
          <output class="position-scope-summary" aria-live="polite">${scopeLabel}</output>
        </div>
        <div class="token-strip inspector-token-strip" aria-label="Sequence positions included in readout counts">${condition.tokens.map((token) => renderPositionToken(token, customSet.has(token.position), positionSet.has(token.position), name)).join("")}</div>`;
      const denseColumns = layers.length > 20;
      const columns = layers.map((layer, index) => {
        const finalIndex = layers.length - 1;
        const nearFinalLayer = index < finalIndex && layers[finalIndex] - layer < 5;
        const showLabel = !denseColumns || index === 0 || index === finalIndex || (layer % 5 === 0 && !nearFinalLayer);
        return `<span title="Layer ${layer}${layer === 51 ? "; read only" : ""}">${showLabel ? layer : ""}</span>`;
      }).join("");
      const matrixRows = rows.length
        ? rows.map((row, rowIndex) => renderMatrixRow(row, name, lensType, layers, rowIndex, positionsInScope.length)).join("")
        : `<div class="matrix-empty">No ${lensType === "jacobian" ? "J-lens" : "logit-lens"} readouts across this position scope.</div>`;
      element("inspectorBody").innerHTML = `${header}<div class="position-panel">${positions}</div>
        <div class="matrix-summary"><div><span>Token</span><small>${rows.length} candidates</small></div><span>Count</span><div><span>Count by layer</span><small>${positionsInScope.length} positions · L${layers[0] ?? "—"}–${layers[layers.length - 1] ?? "—"}</small></div></div>
        <div class="matrix-scroll" tabindex="0" aria-label="${name} ${lensType} token by layer readout matrix">
          <div class="matrix-head" style="--layer-count:${layers.length};grid-template-columns:${matrixTemplate(layers)}"><span>Decoded token</span><span>Count</span>${columns}</div>${matrixRows}
        </div>`;
      const strip = element("inspectorBody").querySelector(".inspector-token-strip");
      if (strip && Number.isFinite(settings.stripScrollLeft)) strip.scrollLeft = settings.stripScrollLeft;
      let focusTarget = null;
      if (Number.isInteger(settings.focusPosition)) {
        focusTarget = element("inspectorBody").querySelector(`button[data-action="position"][data-condition="${name}"][data-position="${settings.focusPosition}"]`);
      } else if (settings.focusSelector) {
        focusTarget = element("inspectorBody").querySelector(settings.focusSelector);
      }
      if (focusTarget) focusTarget.focus({ preventScroll: true });
      restoreContributorHighlight();
    }

    function clearContributorHighlight() {
      doc.querySelectorAll(".position-token.is-contributor").forEach((token) => {
        token.classList.remove("is-contributor");
        token.style.removeProperty("--contribution-opacity");
        token.removeAttribute("data-contribution-count");
      });
      element("inspectorBody").querySelectorAll(".matrix-row.is-previewed").forEach((row) => row.classList.remove("is-previewed"));
    }

    function highlightContributorRow(rowIndex) {
      clearContributorHighlight();
      const row = state.inspectorRows[rowIndex];
      const context = state.inspectorContext;
      if (!row || !context || !(row.positionCounts instanceof Map)) return;
      const maximum = Math.max(1, ...row.positionCounts.values());
      for (const [position, count] of row.positionCounts) {
        const token = element("inspectorBody").querySelector(`.position-token[data-condition="${context.conditionName}"][data-position="${position}"]`);
        if (!token) continue;
        const normalized = count / maximum;
        token.classList.add("is-contributor");
        token.style.setProperty("--contribution-opacity", `${Math.round(16 + normalized * 54)}%`);
        token.dataset.contributionCount = String(count);
      }
      const matrixRow = element("inspectorBody").querySelector(`.matrix-row[data-row-index="${rowIndex}"]`);
      if (matrixRow) matrixRow.classList.add("is-previewed");
    }

    function restoreContributorHighlight() {
      if (!state.selected || !state.inspectorContext
          || state.selected.conditionName !== state.inspectorContext.conditionName
          || state.selected.lensType !== state.inspectorContext.lensType) {
        clearContributorHighlight();
        return;
      }
      const rowIndex = state.inspectorRows.findIndex((row) => row.id === state.selected.entry.id);
      if (rowIndex >= 0) highlightContributorRow(rowIndex);
      else clearContributorHighlight();
    }

    function renderResults() {
      if (!state.result || !state.result.clean) return;
      const panels = [renderConversationCondition(state.result.clean, "clean", "A")];
      if (state.result.intervened) panels.push(renderConversationCondition(state.result.intervened, "intervened", "B"));
      else panels.push(`<article class="condition-panel is-placeholder"><div><strong>B · Intervened</strong><p>No intervention condition in this run.</p></div></article>`);
      element("comparisonGrid").innerHTML = `<div class="prompt-thread">${promptThread()}</div><div class="response-grid">${panels.join("")}</div>`;
      element("runIdentity").textContent = state.result.run_id ? `run ${state.result.run_id}` : "Completed run";
      renderInspector();
    }

    function lookupSelected(button) {
      const conditionName = button.dataset.condition;
      const rawCondition = state.result && state.result[conditionName];
      if (!rawCondition) throw new Error("Selected condition is unavailable.");
      const condition = conditionViewModel(rawCondition);
      const layer = Number(button.dataset.layer);
      const position = Number(button.dataset.position);
      const lensType = button.dataset.lens;
      const index = Number(button.dataset.entry);
      const entries = readoutEntries(condition, lensType, layer, position);
      if (!entries[index]) throw new Error("Selected readout token is unavailable.");
      const tokenId = Number(button.dataset.tokenId);
      if (!Number.isInteger(tokenId) || entries[index].id !== tokenId) {
        throw new Error("Selected readout identity is stale. Refresh the position scope and try again.");
      }
      const visibleLayers = state.inspectorContext
        && state.inspectorContext.conditionName === conditionName
        && state.inspectorContext.lensType === lensType
        ? state.inspectorContext.layers.filter((value) => value <= 50)
        : [layer];
      const aggregate = state.inspectorRows[Number(button.dataset.rowIndex)] || null;
      return { conditionName, layer, position, lensType, entry: entries[index], visibleLayers, aggregate };
    }

    function openIntervention(selected, trigger) {
      if (!Number.isInteger(selected.layer) || selected.layer < 0 || selected.layer > 50) {
        throw new Error("Block 51 is read-only and cannot be used for interventions.");
      }
      state.selected = selected;
      state.sourceTrigger = trigger || null;
      state.interventionLayersDirty = false;
      element("inspectorBody").querySelectorAll(".top-token.is-selected").forEach((item) => {
        item.classList.remove("is-selected");
        item.setAttribute("aria-pressed", "false");
      });
      element("inspectorBody").querySelectorAll(".matrix-row.is-selected").forEach((item) => item.classList.remove("is-selected"));
      if (trigger) {
        trigger.classList.add("is-selected");
        trigger.setAttribute("aria-pressed", "true");
      }
      state.sourceValidation = {
        text: selected.entry.text,
        token_ids: [selected.entry.id],
        pieces: [{ id: selected.entry.id, text: selected.entry.text }],
        is_single_token: true,
      };
      state.targetValidation = null;
      element("sourceTokenText").value = selected.entry.text;
      element("targetTokenText").value = "";
      renderTokenValidation("source", state.sourceValidation);
      renderTokenValidation("target", null);
      const label = escapeHtml(visibleWhitespace(selected.entry.text));
      const aggregateMeta = selected.aggregate
        ? `${selected.aggregate.occurrenceCount} hits across ${selected.aggregate.positionCount} positions · `
        : "";
      element("sourceSummary").innerHTML = `<strong>${label}</strong><span>id ${selected.entry.id}</span><span class="source-meta">${aggregateMeta}${selected.lensType} · source layer ${selected.layer} · ${selected.conditionName} position ${selected.position}</span>`;
      const steerRadio = doc.querySelector('input[name="interventionMode"][value="steer"]');
      steerRadio.checked = true;
      element("interventionLayers").value = String(selected.layer);
      element("strengthRange").value = "-0.1";
      element("strengthNumber").value = "-0.1";
      element("generatedToggle").checked = false;
      element("interventionPanel").hidden = false;
      clearError("interventionError");
      updateModeControls();
      element("interventionPanel").scrollIntoView({ behavior: "smooth", block: "nearest" });
      element("closeInterventionButton").focus({ preventScroll: true });
    }

    function closeIntervention() {
      element("interventionPanel").hidden = true;
      if (state.sourceTrigger && state.sourceTrigger.isConnected) {
        state.sourceTrigger.focus({ preventScroll: true });
      }
    }

    function currentMode() {
      const checked = doc.querySelector('input[name="interventionMode"]:checked');
      return checked ? checked.value : "steer";
    }

    function updateLayerWarning() {
      const warning = element("layerWarning");
      try {
        const layers = parseLayerList(element("interventionLayers").value, INTERVENTION_LAYER_OPTIONS);
        const late = layers.filter((layer) => layer >= 41);
        if (late.length >= 4 || layers.length >= 12) {
          warning.textContent = "Broad late-layer steering may behave as direct output manipulation; interpret it cautiously.";
          warning.hidden = false;
        } else {
          warning.hidden = true;
        }
      } catch (_error) {
        warning.hidden = true;
      }
    }

    function updateModeControls() {
      const mode = currentMode();
      if (state.selected && !state.interventionLayersDirty) {
        const defaultLayers = mode === "swap" && Array.isArray(state.selected.visibleLayers) && state.selected.visibleLayers.length
          ? state.selected.visibleLayers
          : [state.selected.layer];
        element("interventionLayers").value = formatLayers(defaultLayers);
      }
      const strengthDisabled = mode !== "steer";
      element("strengthRange").disabled = strengthDisabled;
      element("strengthNumber").disabled = strengthDisabled;
      element("strengthControl").style.opacity = strengthDisabled ? "0.55" : "1";
      element("swapTarget").hidden = mode !== "swap";
      element("modeNote").textContent = MODE_NOTES[mode];
      updateLayerWarning();
    }

    function renderTokenValidation(kind, result, error) {
      const output = element(`${kind}Tokenization`);
      const label = kind === "source" ? "Source" : "Target";
      output.className = "tokenization-result";
      if (error) {
        output.classList.add("is-invalid");
        output.textContent = error.message || String(error);
        return;
      }
      if (!result) {
        output.textContent = `${label} not validated`;
        return;
      }
      const pieces = Array.isArray(result.pieces) ? result.pieces : [];
      const pieceHtml = pieces.map((piece) => {
        const id = Number.isInteger(piece.id) ? piece.id : "invalid";
        return `<span class="piece-chip" title="Exact decoded text: ${exactTitle(piece.text)}"><strong>${escapeHtml(visibleWhitespace(piece.text))}</strong><small>id ${id}</small></span>`;
      }).join("");
      output.classList.add(result.is_single_token ? "is-valid" : "is-invalid");
      output.innerHTML = `${pieceHtml}<span>${result.is_single_token ? "Valid single token" : `${pieces.length} token pieces; exactly one required`}</span>`;
    }

    let sourceDebounce = null;
    let targetDebounce = null;
    async function validateSwapToken(kind) {
      const validationKey = `${kind}Validation`;
      const sequenceKey = `${kind}ValidationSequence`;
      const text = element(`${kind}TokenText`).value;
      const sequence = ++state[sequenceKey];
      state[validationKey] = null;
      if (text.length === 0) {
        renderTokenValidation(kind, null);
        return null;
      }
      element(`${kind}Tokenization`).className = "tokenization-result";
      element(`${kind}Tokenization`).textContent = "Tokenizing…";
      try {
        const result = await apiFetch("/api/tokenize", { method: "POST", body: { text }, requestId: requestId() });
        if (sequence !== state[sequenceKey]) return null;
        result.text = text;
        if (result.is_single_token && (!Array.isArray(result.token_ids) || !Number.isInteger(result.token_ids[0]))) {
          throw new Error("Tokenizer returned an invalid numeric token ID.");
        }
        state[validationKey] = result;
        renderTokenValidation(kind, result);
        return result;
      } catch (error) {
        if (sequence !== state[sequenceKey]) return null;
        renderTokenValidation(kind, null, error);
        return null;
      }
    }

    function controlsValue() {
      return {
        mode: currentMode(),
        layers: element("interventionLayers").value,
        strength: element("strengthNumber").value,
        applyToGenerated: element("generatedToggle").checked,
        source: state.sourceValidation,
        target: state.targetValidation,
      };
    }

    function provenanceItems() {
      const info = state.info || {};
      const run = state.result && state.result.provenance ? state.result.provenance : {};
      const model = info.model || {};
      const lens = info.lens || {};
      const layerPolicy = info.layer_policy || {};
      const readoutPolicy = layerPolicy.readout || {};
      const interventionPolicy = layerPolicy.intervention || {};
      const readoutPolicyLabel = Array.isArray(readoutPolicy.default)
        ? `${readoutPolicy.min}–${readoutPolicy.max} · default ${formatLayers(readoutPolicy.default)} · up to ${readoutPolicy.max_selected}`
        : null;
      const interventionPolicyLabel = Number.isInteger(interventionPolicy.max_selected)
        ? `${interventionPolicy.min}–${interventionPolicy.max} · up to ${interventionPolicy.max_selected} · block 51 read-only`
        : null;
      const values = [
        ["Disclosure", DISCLOSURE],
        ["Model", run.model_id || model.id],
        ["Revision", run.model_revision || model.revision],
        ["Model dtype", model.dtype],
        ["Mamba backend", run.mamba_backend || model.mamba_backend],
        ["Lens SHA-256", run.lens_sha256 || lens.sha256],
        ["Acceptance", run.lens_acceptance_tier || (lens.acceptance && lens.acceptance.tier)],
        ["Fit prompts", run.lens_prompt_count || lens.prompt_count],
        ["Fit source SHA-256", run.fit_source_sha256 || lens.fit_source_sha256],
        ["Live source SHA-256", run.live_application_source_sha256 || info.live_application_source_sha256],
        ["Prompt format", run.prompt_format],
        ["Chat thinking", run.chat_template ? (run.chat_template.enable_thinking ? "enabled" : "disabled") : null],
        ["Readout layers", readoutPolicyLabel],
        ["Intervention layers", interventionPolicyLabel],
        ["Neuronpedia commit", run.neuronpedia_reference_commit || info.neuronpedia_reference_commit],
        ["Prompt token SHA-256", run.formatted_prompt_token_sha256],
        ["Run timestamp", run.timestamp],
        ["Run elapsed", Number.isFinite(run.elapsed_seconds) ? `${run.elapsed_seconds.toFixed(2)} s` : null],
      ];
      return values.filter((item) => item[1] !== undefined && item[1] !== null && item[1] !== "");
    }

    function renderProvenance() {
      const items = provenanceItems();
      element("provenanceGrid").innerHTML = items.length
        ? items.map(([label, value]) => `<div class="provenance-item"><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")
        : "<div class=\"provenance-item\"><dt>Service</dt><dd>Waiting for service identity.</dd></div>";
    }

    function exportResult() {
      if (!state.result) return;
      const bundle = {
        schema_version: "nemotron-steering-run-v1",
        exported_at: new Date().toISOString(),
        disclosure: DISCLOSURE,
        request: state.lastRequest,
        result: state.result,
        service_info: state.info,
      };
      const blob = new Blob([`${JSON.stringify(bundle, null, 2)}\n`], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = doc.createElement("a");
      const runId = state.result.run_id || "run";
      link.href = url;
      link.download = `nano-steering-${runId}.json`;
      doc.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    }

    function moveTabFocus(event) {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      const tablist = event.target.closest('[role="tablist"]');
      if (!tablist) return;
      const tabs = Array.from(tablist.querySelectorAll('button[role="tab"]:not(:disabled)'));
      const index = tabs.indexOf(event.target);
      if (index < 0 || tabs.length < 2) return;
      event.preventDefault();
      let next = index;
      if (event.key === "Home") next = 0;
      else if (event.key === "End") next = tabs.length - 1;
      else next = (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      tabs[next].focus();
      tabs[next].click();
    }

    function handleResultAction(event) {
      const button = event.target.closest("button[data-action]");
      if (!button) return;
      if (button.dataset.action === "position") {
        const conditionName = button.dataset.condition;
        const position = Number(button.dataset.position);
        const selection = state.positionSelections[conditionName];
        const rawCondition = state.result && state.result[conditionName];
        if (!rawCondition || !Number.isInteger(position)) return;
        const condition = conditionViewModel(rawCondition);
        let custom;
        if (event.shiftKey && Number.isInteger(selection.anchor)) {
          const start = Math.min(selection.anchor, position);
          const end = Math.max(selection.anchor, position);
          const range = condition.tokens
            .map((token) => token.position)
            .filter((value) => value >= start && value <= end);
          custom = selection.mode === "custom"
            ? Array.from(new Set([...selection.custom, ...range])).sort((left, right) => left - right)
            : range;
        } else if (selection.mode === "custom") {
          const values = new Set(selection.custom);
          if (values.has(position)) values.delete(position);
          else values.add(position);
          custom = Array.from(values).sort((left, right) => left - right);
        } else {
          custom = [position];
        }
        state.positionSelections[conditionName] = custom.length
          ? { mode: "custom", custom, anchor: position }
          : { mode: "all", custom: [], anchor: position };
        const strip = button.closest(".inspector-token-strip");
        renderInspector({ focusPosition: position, stripScrollLeft: strip ? strip.scrollLeft : null });
      } else if (button.dataset.action === "lens-tab") {
        state.tabs[button.dataset.condition] = button.dataset.lens;
        renderInspector({ focusSelector: `button[data-action="lens-tab"][data-lens="${button.dataset.lens}"]` });
      } else if (button.dataset.action === "condition-tab") {
        state.inspectorCondition = button.dataset.condition;
        renderInspector({ focusSelector: `button[data-action="condition-tab"][data-condition="${button.dataset.condition}"]` });
      } else if (button.dataset.action === "source") {
        try {
          openIntervention(lookupSelected(button), button);
          highlightContributorRow(Number(button.dataset.rowIndex));
          const row = button.closest(".matrix-row");
          if (row) row.classList.add("is-selected");
        }
        catch (error) { showToast(error.message); }
      }
    }

    function handleInspectorChange(event) {
      const select = event.target.closest('select[data-action="position-scope"]');
      if (!select) return;
      const conditionName = select.dataset.condition;
      const mode = select.value;
      if (!state.positionSelections[conditionName] || !["all", "prompt", "generated"].includes(mode)) return;
      state.positionSelections[conditionName] = { mode, custom: [], anchor: null };
      renderInspector({ focusSelector: `select[data-action="position-scope"][data-condition="${conditionName}"]` });
    }

    function previewMatrixRow(event) {
      const row = event.target.closest(".matrix-row[data-row-index]");
      if (row) highlightContributorRow(Number(row.dataset.rowIndex));
    }

    function leaveMatrixRow(event) {
      const row = event.target.closest(".matrix-row[data-row-index]");
      if (!row || (event.relatedTarget && row.contains(event.relatedTarget))) return;
      restoreContributorHighlight();
    }

    doc.querySelectorAll("[data-composer-mode]").forEach((button) => {
      button.addEventListener("click", () => setComposerMode(button.dataset.composerMode));
    });
    doc.querySelector('[aria-label="Prompt format"]').addEventListener("keydown", moveTabFocus);
    element("inspectorBody").addEventListener("keydown", moveTabFocus);
    element("addMessageButton").addEventListener("click", () => {
      addMessage("user", "").querySelector("textarea").focus();
    });
    element("samplingToggle").addEventListener("change", () => {
      element("samplingFields").hidden = !element("samplingToggle").checked;
    });
    element("promptForm").addEventListener("submit", (event) => {
      event.preventDefault();
      clearError("formError");
      let request;
      try {
        request = currentRequest();
      } catch (error) {
        showError("formError", error);
        return;
      }
      runRequest("/api/baseline", request, "baseline");
    });
    element("stopButton").addEventListener("click", async () => {
      if (!state.activeRequestId || state.cancelRequested) return;
      state.cancelRequested = true;
      element("stopButton").disabled = true;
      updateProgress({ status: "cancelling", phase: "cancelling" });
      try {
        const response = await apiFetch(`/api/cancel/${encodeURIComponent(state.activeRequestId)}`, { method: "POST", requestId: state.activeRequestId });
        showToast(response.accepted ? "Cancellation requested." : "Request is no longer active.");
      } catch (error) {
        showToast(error.message);
      } finally {
        element("stopButton").disabled = false;
      }
    });
    element("inspectorBody").addEventListener("click", handleResultAction);
    element("inspectorBody").addEventListener("change", handleInspectorChange);
    element("inspectorBody").addEventListener("pointerover", previewMatrixRow);
    element("inspectorBody").addEventListener("pointerout", leaveMatrixRow);
    element("inspectorBody").addEventListener("focusin", previewMatrixRow);
    element("inspectorBody").addEventListener("focusout", leaveMatrixRow);
    doc.querySelectorAll('input[name="interventionMode"]').forEach((radio) => radio.addEventListener("change", updateModeControls));
    element("interventionLayers").addEventListener("input", () => {
      state.interventionLayersDirty = true;
      updateLayerWarning();
    });
    element("strengthRange").addEventListener("input", () => { element("strengthNumber").value = element("strengthRange").value; });
    element("strengthNumber").addEventListener("input", () => {
      const number = Number(element("strengthNumber").value);
      if (Number.isFinite(number)) element("strengthRange").value = String(Math.max(-2, Math.min(2, number)));
    });
    element("sourceTokenText").addEventListener("input", () => {
      clearTimeout(sourceDebounce);
      state.sourceValidation = null;
      sourceDebounce = setTimeout(() => validateSwapToken("source"), 350);
    });
    element("targetTokenText").addEventListener("input", () => {
      clearTimeout(targetDebounce);
      state.targetValidation = null;
      targetDebounce = setTimeout(() => validateSwapToken("target"), 350);
    });
    element("runInterventionButton").addEventListener("click", async () => {
      clearError("interventionError");
      const runButton = element("runInterventionButton");
      runButton.disabled = true;
      let payload;
      try {
        if (currentMode() === "swap") {
          clearTimeout(sourceDebounce);
          clearTimeout(targetDebounce);
          await Promise.all([
            validateSwapToken("source"),
            validateSwapToken("target"),
          ]);
        }
        payload = buildInterventionPayload(state.baselineRequest, state.selected, controlsValue());
      } catch (error) {
        showError("interventionError", error);
        runButton.disabled = false;
        return;
      }
      await runRequest("/api/intervene", payload, "intervention");
    });
    element("closeInterventionButton").addEventListener("click", closeIntervention);
    doc.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && !element("interventionPanel").hidden) {
        event.preventDefault();
        closeIntervention();
      }
    });
    element("exportButton").addEventListener("click", exportResult);
    element("refreshStatusButton").addEventListener("click", refreshService);

    addMessage("user", "");
    renderProvenance();
    refreshService();
  }

  return {
    DISCLOSURE,
    normalizeToken,
    normalizeReadoutEntry,
    conditionViewModel,
    readoutEntries,
    selectedPositions,
    aggregateReadoutRows,
    visibleWhitespace,
    parseLayerList,
    formatLayers,
    buildInterventionPayload,
    progressPercent,
    mount,
  };
});
