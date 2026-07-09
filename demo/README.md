# Nemotron lens fixture explorer

This directory is a no-build, static explorer for Jacobian-lens and ordinary
logit-lens fixtures. It is the interaction scaffold for the requested
Neuronpedia-style demo; it does not load or contact Nemotron.

## Important data warning

<code>fixtures/nemotron-3-nano.sample.json</code> is hand-authored,
<code>illustrative</code> geography data. It exists so the UI can be exercised
without a 30B-model GPU. It is **not** a Nemotron inference, fitted-lens
artifact, paper result, or directed-modulation result.

A real fixture must be generated from a validated lens and have
<code>_fixture.mode = "recorded"</code> plus checkpoint/lens provenance. See
[the demo acceptance specification](../docs/demo.md).

## Run locally

From the repository root:

~~~bash
nemotron-jlens serve-demo --directory demo --host 127.0.0.1 --port 8000
~~~

Then open <http://127.0.0.1:8000>. Use **Import fixture** to load a recorded
JSON file. **Export fixture** round-trips the currently loaded data.

An accepted 100-prompt recording may be bundled under
<code>fixtures/nemotron-3-nano.pilot-recorded.json</code> and opened explicitly
at <http://127.0.0.1:8000/?fixture=pilot>. This query is an opt-in: the plain
page still loads only a canonical accepted paper-scale recording or the
illustrative sample, never the pilot.

The bundled illustrative sample remains the default until
<code>fixtures/nemotron-3-nano.recorded.json</code> exists and carries the exact
exported paper-scale acceptance decision described below. An under-scale
recording such as <code>fixtures/nemotron-3-nano.smoke-recorded.json</code>
must be imported intentionally; the header then shows its fitting prompt count
and the notice marks it as a pipeline/interface check rather than the final
1,000-prompt result. Keep smoke recordings under that distinct filename and do
not overwrite <code>nemotron-3-nano.sample.json</code>.

Serving over HTTP is recommended; opening <code>index.html</code> directly may
prevent the browser from fetching the bundled sample.

## Produce a recorded fixture

After fitting, merging, and validating the full lens:

~~~bash
nemotron-jlens export-fixture artifacts/nemotron-1000.pt \
  demo/fixtures/nemotron-3-nano.recorded.json \
  --example modulation-topic \
  --top-n 8 --layer-stride 1 \
  --tracked-token ' fish' \
  --tracked-token ' whale' \
  --tracked-token ' coral' \
  --tracked-token ' lobster' \
  --device-map auto
~~~

This is a GPU command. The explorer itself remains static and can be served on
any machine after export.

## Interactions

- Hover or keyboard-focus a transcript token to inspect one position.
- Click a transcript token to pin it; Shift-click adds/removes positions.
- Switch among Jacobian, logit, and rank-score difference modes.
- Restrict visible sampled layers with the two range controls.
- Change how many recorded top-k candidates contribute to the view.
- Hover a sidebar token to preview its transcript map.
- Click up to four sidebar tokens to lock color-coded highlights.
- Press Escape to clear hover, then selected positions, then locked tokens.

Difference mode compares aggregate top-k rank scores for ordinary rows. When
both result types provide a tracked token, it compares the recorded
probabilities for that token. Neither display is a focus-versus-control
comparison or a causal effect.

## Fixture shape

The page expects:

- <code>meta.kind = "meta"</code>;
- numeric layer lists under
  <code>meta.layers_by_type.JACOBIAN_LENS</code> and
  <code>meta.layers_by_type.LOGIT_LENS</code>;
- a nonempty <code>tokens</code> array with unique integer positions; and
- for every token and lens type, <code>top_tokens</code> and
  <code>top_probs</code> matrices with one row per declared layer.

Optional <code>_fixture.transcript</code> entries define labeled inclusive
position ranges. Without them, the page segments tokens using
<code>is_generated</code>. Optional <code>done</code> metadata is shown in the
run summary when available.

Each result matrix has shape `[declared layers][retained predictions]`.
`top_probs` must match the corresponding `top_tokens` row lengths, and every
probability is in `[0, 1]`.

### Saved opening state

Optional `_fixture.ui` values recreate a share-like opening state:

~~~json
{
  "selected_positions": [42, 43, 44],
  "locked_tokens": [" fish", " whale", " coral", " lobster"]
}
~~~

Selected positions must exist in `tokens`. At most four nonempty strings may be
locked. Export writes the current selection and locks back to this object. The
bundled illustrative fixture deliberately starts with both arrays empty.

### Full-vocabulary tracked tokens

Top-k rows cannot show a locked token after it falls below the retained list. A
recorded exporter therefore attaches a `tracked` row list to each result:

~~~json
{
  "type": "JACOBIAN_LENS",
  "top_tokens": [[" letter"], [" drawer"]],
  "top_probs": [[0.09], [0.14]],
  "tracked": [
    {
      "token": " fish",
      "id": 12345,
      "ranks": [123, 17],
      "probs": [0.00031, 0.012]
    }
  ]
}
~~~

`token` preserves the tokenizer's display form and `id` is its vocabulary ID.
`ranks` and `probs` align with that result type's declared layer list; ranks are
one-based and missing values are `null`. Locked rows use these actual
measurements when present. The viewer also accepts the earlier token-keyed map
shape for compatibility. Fixtures without `tracked` remain supported and show
only retained top-k rank scores.

Recorded fixtures should put the immutable model/tokenizer revisions, lens and
corpus hashes, exporter commit, prompt construction, evaluation settings, and
timestamp under `_fixture.provenance`. The viewer displays supplied provenance;
it does not certify it.

### Acceptance and default selection

The exporter copies the validated lens acceptance decision into
<code>_fixture.acceptance</code>:

~~~json
{
  "schema": "nemotron-jlens-acceptance/v1",
  "tier": "paper-scale",
  "status": "accepted",
  "is_final": true,
  "exportable": true,
  "checks": {
    "pinned_model": true,
    "pinned_dataset": true,
    "known_manifest": true,
    "complete_prompt_set": true,
    "full_layer_coverage": true,
    "fixed_fit_settings": true,
    "pinned_transformers": true,
    "fp32_storage": true
  },
  "reasons": []
}
~~~

The final recording is eligible to replace the illustrative startup fixture
only when that exact schema declares <code>tier = "paper-scale"</code>,
<code>status = "accepted"</code>, <code>is_final = true</code>, exactly 1,000
fitting prompts, and every required check is true. The browser also requires
the canonical corpus manifest, pinned WikiText dataset identity, source layers
0–50, target layer 51, FP32 lens storage, all fixed fit settings, and the exact
<code>modulation-topic</code> focus example. The top-level acceptance decision
must be deeply identical to <code>_fixture.provenance.lens.acceptance</code>.
Smoke exports use
<code>status = "non-final"</code>; a canonical pilot may be
<code>status = "accepted"</code>, but both declare
<code>is_final = false</code> and remain importable but opt-in. Missing, unknown,
inconsistent, noncanonical, or hand-edited final decisions fail closed and
never become the default. The page never probes the smoke filename.

For every valid recorded import, a visible provenance panel shows the complete
model revision, lens and corpus-manifest SHA-256 declarations, fitting prompt
count, target/source layers, acceptance tier/status, reasons, and checks. Its
verification-boundary warning is intentional: browser validation cannot prove
that the displayed hashes match remote artifacts or that the scientific run
occurred.

## Directed-modulation acceptance

The genuine acceptance fixture uses this exact user instruction:

> Write "She carefully placed the letter back inside the wooden drawer."
> Concentrate on ocean creatures while you write the sentence. Don't write
> anything else.

It evaluates the assistant sequence `She carefully placed the letter back
inside the wooden drawer.`. Response positions must be selected according to
Nemotron's own tokenizer, and `fish`, `whale`, `coral`, and `lobster` should be
locked with full-vocabulary trajectories. Acceptance also requires a matched
"do not think about ocean creatures" control and complete provenance. A
successfully rendered fixture is not evidence for directed modulation unless
its recorded values and control support that conclusion.

The complete contract, provenance requirements, exact prompt, controls, and
acceptance tests are in [docs/demo.md](../docs/demo.md).

## Files

| File | Role |
|---|---|
| <code>index.html</code> | Accessible page structure |
| <code>styles.css</code> | Responsive layout and visual states |
| <code>provenance.js</code> | Recorded provenance and final-default policy |
| <code>app.js</code> | Fixture validation, interaction state, and rendering |
| <code>tests/provenance.test.js</code> | Deterministic acceptance-policy checks |
| <code>fixtures/nemotron-3-nano.sample.json</code> | Illustrative UI-only fixture |
| <code>fixtures/nemotron-3-nano.smoke-recorded.json</code> | Optional real, under-scale pipeline recording; not a final result |
| <code>fixtures/modulation-topic-neutral.smoke-recorded.json</code> | Real under-scale no-side-task control |
| <code>fixtures/modulation-topic-suppress.smoke-recorded.json</code> | Real under-scale suppression control |
| <code>fixtures/modulation-topic-mention-control.smoke-recorded.json</code> | Real under-scale mention-only control |

No build step, package install, API key, model weights, cookies, or telemetry
are required to view an existing fixture.
