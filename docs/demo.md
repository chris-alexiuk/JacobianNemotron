# Directed-modulation demo specification

## Target

The interaction reference is the shared Neuronpedia Jacobian-lens view:

<https://www.neuronpedia.org/qwen3.6-27b/jlens?shareId=cmr1hlmkj0001pt2x8udm0029>

That page is a Qwen result. This project reproduces the **experiment and
interaction pattern** for pinned Nemotron 3 Nano; it must not copy Qwen values
or assume the same token indices, layer count, or outcome.

The deliverable has two views:

- <code>demo/</code> is a local, fixture-driven transcript/readout explorer
  modeled on the requested interaction.
- <code>nemotron-jlens render</code> produces Anthropic's released static
  layer-by-position viewer from the same lens and prompt.

The custom page deliberately has no model backend, telemetry, or external
state. A GPU export records data once; the browser reads JSON. This makes a
demo shareable and auditable without hiding inference behind a live service.

## Exact primary condition

User message:

> Write "She carefully placed the letter back inside the wooden drawer."
> Concentrate on ocean creatures while you write the sentence. Don't write
> anything else.

Teacher-forced assistant response:

> She carefully placed the letter back inside the wooden drawer.

The bundled upstream example slug is <code>modulation-topic</code>. The
assistant response is part of the resolved chat prompt so every condition can
be compared over identical visible text.

Export from a validated full lens:

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

The four tracked displays match the reference share, but they are hypotheses,
not a cherry-picked success criterion. Store their exact Nemotron token IDs
and tokenizer decompositions. If a display is not one token under the pinned
tokenizer, report that fact and track the actual pieces instead of pretending
that Qwen's tokenization transfers.

## Required controls

Use the same chat template, exact assistant response, lens, source layers,
readout settings, and display span for:

1. **Focus** (<code>modulation-topic</code>): the exact primary prompt above.
2. **No-side-task** (<code>modulation-topic-neutral</code>):

   > Write "She carefully placed the letter back inside the wooden drawer."
   > Don't write anything else.

3. **Suppression** (<code>modulation-topic-suppress</code>):

   > Write "She carefully placed the letter back inside the wooden drawer."
   > Don't think about ocean creatures. Don't write anything else.

4. **Mention-only, recommended**
   (<code>modulation-topic-mention-control</code>):

   > Write "She carefully placed the letter back inside the wooden drawer."
   > This prompt contains a reference to ocean creatures. Don't write anything
   > else.

Export the controls with the same display arguments:

~~~bash
for SLUG in \
  modulation-topic-neutral \
  modulation-topic-suppress \
  modulation-topic-mention-control
do
  nemotron-jlens export-fixture artifacts/nemotron-1000.pt \
    "demo/fixtures/$SLUG.recorded.json" \
    --example "$SLUG" --top-n 8 --layer-stride 1 \
    --tracked-token ' fish' --tracked-token ' whale' \
    --tracked-token ' coral' --tracked-token ' lobster' \
    --device-map auto
done
~~~

Build the CPU-only comparison artifact after all fixtures are recorded:

~~~bash
nemotron-jlens compare-fixtures \
  demo/fixtures/nemotron-3-nano.recorded.json \
  --control neutral=demo/fixtures/modulation-topic-neutral.recorded.json \
  --control suppression=demo/fixtures/modulation-topic-suppress.recorded.json \
  --control mention=demo/fixtures/modulation-topic-mention-control.recorded.json \
  --output artifacts/directed-modulation-comparison.json
~~~

This command reads only exact full-vocabulary `tracked` ranks and
probabilities at `is_generated` assistant positions. It rejects illustrative
fixtures and mismatched model revisions, lens checksums, or layer sets. Positive
probability and reciprocal-rank deltas favor focus; positive median-rank
improvement means the control median rank was worse than focus.

Archive every resolved prompt string and token ID sequence. The Qwen share
selects positions 42–52, but those indices are not valid acceptance criteria
for Nemotron. The correct span is the assistant response as tokenized by the
pinned Nemotron tokenizer.

The demo may show one condition at a time, but the result bundle must at least
include focus, neutral, and suppression fixtures or an analysis artifact that
compares them. At minimum,
report the following per tracked token and condition:

- best and median rank over the assistant span by layer;
- full-softmax probability at the best layer/position;
- focus-minus-no-side-task and focus-minus-suppression changes;
- the best layer and response-relative position; and
- Jacobian-lens versus ordinary logit-lens behavior.

Do not call the result positive merely because one ocean word appears once in
top-k. Look for a coherent signal across multiple pre-registered terms or
semantically relevant terms, a meaningful control contrast, and stability
under reasonable layer/position summaries. Report null and contrary results.

## Recorded fixture contract

The explorer accepts a JSON object with four top-level sections:

~~~json
{
  "_fixture": {
    "schema": "nemotron-jlens-fixture/v1",
    "mode": "recorded",
    "title": "Directed modulation",
    "description": "Recorded from the pinned Nemotron model",
    "acceptance": {
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
    },
    "transcript": [
      {"label": "Prompt", "start": 0, "end": 41, "generated": false},
      {"label": "Assistant", "start": 42, "end": 52, "generated": true}
    ]
  },
  "meta": {
    "kind": "meta",
    "model": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "types": ["JACOBIAN_LENS", "LOGIT_LENS"],
    "layers_by_type": {
      "JACOBIAN_LENS": [0],
      "LOGIT_LENS": [0]
    },
    "top_n": 8
  },
  "tokens": [
    {
      "kind": "token",
      "position": 0,
      "token": "Write",
      "id": 123,
      "is_generated": false,
      "results": [
        {
          "type": "JACOBIAN_LENS",
          "top_tokens": [[" token"]],
          "top_probs": [[0.01]]
        },
        {
          "type": "LOGIT_LENS",
          "top_tokens": [[" token"]],
          "top_probs": [[0.01]]
        }
      ]
    }
  ],
  "done": {"kind": "done"}
}
~~~

The numbers above illustrate shape only; they are not Nemotron observations.
The actual transcript boundaries and layer arrays come from the exporter.

Contract invariants:

- <code>meta.kind</code> is <code>meta</code>.
- Both lens types have nonempty numeric layer arrays.
- Token positions are unique integers.
- Every token has one result for each lens type.
- Each result has one top-token and probability row per corresponding layer.
- Token and probability rows align by rank.
- <code>_fixture.mode</code> is <code>recorded</code> only for actual model
  output; synthetic or hand-authored data use <code>illustrative</code>.
- Recorded artifacts carry model revision, lens SHA-256, upstream code
  revision, prompt text/hash, tokenizer identity, fitting prompt count, source
  layer set, and export parameters. Consumers should not trust a
  <code>recorded</code> label without that provenance.
- The exporter copies its validated, schema-versioned acceptance decision into
  <code>_fixture.acceptance</code>; this decision is artifact metadata, not a
  browser inference from prompt count.

A genuine model replay may be recorded without being paper-scale. The explorer
shows the fitted prompt count from <code>_fixture.provenance.lens.n_prompts</code>;
recordings below the fixed 1,000-prompt target are visibly marked under-scale
and are suitable only for pipeline/interface validation. Smoke acceptance
metadata uses <code>status = "non-final"</code>; a canonical pilot can be
<code>status = "accepted"</code>, but both use
<code>is_final = false</code>. Store an eight-prompt smoke export as
<code>demo/fixtures/nemotron-3-nano.smoke-recorded.json</code>, import it
intentionally, and leave the bundled illustrative sample unchanged. Neither a
valid recorded label nor successful rendering establishes the directed-
modulation result.

The explorer checks <code>fixtures/nemotron-3-nano.recorded.json</code> at
startup, but selects it only when all of the following declarations agree:

- acceptance schema <code>nemotron-jlens-acceptance/v1</code>;
- tier <code>paper-scale</code>, status <code>accepted</code>,
  <code>is_final = true</code>, and <code>exportable = true</code>;
- exactly 1,000 fitting prompts from the canonical manifest;
- true <code>pinned_model</code>, <code>pinned_dataset</code>,
  <code>known_manifest</code>, <code>complete_prompt_set</code>,
  <code>full_layer_coverage</code>, <code>fixed_fit_settings</code>,
  <code>pinned_transformers</code>, and
  <code>fp32_storage</code> checks; and
- the pinned model/tokenizer/provenance validation already required of every
  recorded fixture.

If the file is absent or any condition fails, the illustrative sample remains
the default. The smoke filename is never auto-loaded. Unknown or missing
acceptance metadata remains non-final and opt-in.

Top-k probabilities are full-vocabulary softmax probabilities, not a softmax
renormalized over the displayed tokens.

## Explorer behavior

The page currently implements:

- Jacobian, ordinary logit, and difference tabs;
- a sampled-layer start/end range and top-k control;
- prompt/assistant transcript segmentation;
- hover or keyboard focus to scope readouts to one transcript position;
- click to pin one position and Shift-click to compare positions;
- per-layer readout columns;
- an aggregate token sidebar over the active positions and layers;
- hover preview and up to four color-coded locked token highlights;
- local JSON import and export; and
- responsive desktop/mobile layout with no network dependency.

For ordinary top-k rows, “Difference” is an aggregate **rank-score** contrast
between Jacobian and logit readouts. For explicitly tracked tokens it uses the
recorded full-vocabulary probability difference. The UI must distinguish these
semantics: neither value is a focus-versus-control comparison or a causal
effect.

The browser does not fit a lens, run Nemotron, generate text, or validate the
cryptographic provenance. Those responsibilities belong to the CLI pipeline
and artifact review.

## Visual and interaction acceptance

A recorded demo is ready only when all of the following pass:

- The header identifies Nemotron 3 Nano and labels data as recorded.
- The fixture exposes exact checkpoint revision and final lens checksum.
- Recorded fixtures expose a provenance panel with the model revision, lens and
  corpus-manifest SHA-256 values, fitting prompt count, target/source layers,
  acceptance tier/status, and any exported reasons/checks.
- The panel explicitly says its declarations are browser-unverified.
- Prompt and teacher-forced response are visibly distinct.
- Hovering a transcript token updates the layer readout without changing the
  selected condition.
- Clicking and Shift-clicking positions produces stable selection behavior.
- Jacobian and logit tabs show the same visible layer set; difference mode
  states its rank-score semantics.
- Layer-range and top-k controls update the transcript highlights, layer
  columns, and sidebar consistently.
- Four tokens can be locked, color identity remains stable, and unlocking
  removes the highlight.
- Ocean-creature terms can be inspected even if their observed result is null.
- The layout remains usable at narrow viewport widths.
- Import rejects malformed fixture shapes with a visible error.
- Export round-trips the loaded fixture without silently changing values.
- The page works from the documented local HTTP server.

## Scientific acceptance

The final report must answer, with the recorded fixtures rather than a
screenshot:

1. Did the primary focus condition produce an ocean-creature representation
   in Jacobian space during the fixed assistant response?
2. Was it stronger or more coherent than the ordinary logit lens?
3. Was it stronger than both the no-side-task and suppression controls?
4. At which Nemotron layers and response-relative positions did it occur?
5. Did results persist with the 1,000-prompt lens after the pilot?

Possible dispositions are **positive**, **null**, **mixed**, or
**infrastructure failure**. “Infrastructure failure” is reserved for failed
preflight, fitting, checksums, or export—not for a model that simply lacks the
expected signal.

## Serving

Custom explorer:

~~~bash
nemotron-jlens serve-demo --directory demo --host 127.0.0.1 --port 8000
~~~

Official rendered page:

~~~bash
nemotron-jlens serve-demo \
  --directory artifacts/static-demo \
  --host 127.0.0.1 --port 8001
~~~

Use an authenticated static host if either output leaves the local machine.
The fixture can expose complete prompts and internal model readouts even
though it contains no model weights.
