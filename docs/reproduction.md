# Nemotron 3 Nano Jacobian-lens reproduction protocol

## Objective

Re-run the public Jacobian-lens method from Anthropic's
[*Verbalizable Representations Form a Global Workspace in Language
Models*](https://transformer-circuits.pub/2026/workspace/index.html) on a
pinned Nemotron 3 Nano checkpoint, preserve enough provenance to audit the
result, and turn the result into an interactive directed-modulation demo.

This protocol makes a narrower claim than “the paper has been reproduced.” It
separates:

1. **Core method reproduction:** fit and apply the released average-Jacobian
   readout, and render its layer-by-position output. This repository has an
   end-to-end implementation for Nemotron.
2. **Empirical replication:** run the paper's released prompt sets on a new,
   hybrid Mamba/attention/MoE model and report the measurements. The public
   prompt data are present, but a Nemotron GPU run is still required.
3. **Unreleased causal work:** the paper's steering, activation-ablation, and
   representation-swap harnesses are not in the public
   [Anthropic repository](https://github.com/anthropics/jacobian-lens).
   Reconstructing them would be an independent implementation and must be
   reported as such.

There is currently **no GPU-produced Nemotron lens in this checkout** and no
claim that the directed-modulation result is positive. The sample web fixture
is marked <code>illustrative</code>; it tests the interface, not the
hypothesis.

## Method being transferred

For source residual layer \(l\), the lens fits an average linear transport into
the target residual basis:

\[
J_l = \mathbb{E}_{x,t,t'\geq t}
\left[\frac{\partial h_{T,t'}}{\partial h_{l,t}}\right].
\]

The transported activation is normalized as a target-layer residual and
decoded through the model's own unembedding:

\[
\operatorname{readout}_l(h_{l,t}) =
W_U\,\operatorname{norm}_T(J_l h_{l,t}).
\]

The public fitter estimates output dimensions in batches using vector-Jacobian
products. For this adaptation, defaults are 1,000 pretraining-like sequences,
128 tokens per sequence, the first 16 source positions skipped, and an output
dimension batch of 8. The target is block output 51 (CLI
<code>--target-layer -1</code>), with source block outputs 0–50. That
final-block choice follows the public fitter's open-model default; it should
not be presented as numerically identical to every model-specific
configuration in the paper.

## Pinned inputs

| Input | Pin |
|---|---|
| Model | <code>nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16</code> |
| Model revision | <code>cbd3fa9f933d55ef16a84236559f4ee2a0526848</code> |
| Model architecture | 52 blocks, residual width 2,688, vocabulary 131,072 |
| Upstream Jacobian-lens code | <code>581d398613e5602a5af361e1c34d3a92ea82ba8e</code> |
| Fitting dataset | <code>Salesforce/wikitext</code>, <code>wikitext-103-raw-v1</code>, train split |
| Dataset revision | <code>b08601e04326c79dfdd32d625aee71d232d685c3</code> |
| Corpus construction | <code>ordered-concatenate-and-chunk-v1</code> |
| Local adaptation source | Per-artifact <code>adaptation_source_sha256</code> |

The corpus builder materializes the pinned dataset revision in the local
Hugging Face cache as disk-backed Arrow data, which is memory-mapped by
default, and then visits rows in dataset order. The non-streaming load is
intentional: it avoids streaming-iterator teardown failures seen in some
container runtimes without requiring the dataset to remain resident in RAM.
The builder removes empty/header rows, concatenates text, emits deterministic
character chunks, and writes per-prompt and whole-manifest SHA-256 hashes.
This is a pretraining-like corpus, not a claim to recover Nemotron's private
training distribution. The exact manifest becomes part of every lens shard's
metadata.

<code>adaptation_source_sha256</code> fingerprints the local
<code>jlens</code> and <code>nemotron_jlens</code> scientific source bundle.
It is recorded alongside commit/model/data pins, carried into downstream
provenance, and enforced across shard merge. It identifies local uncommitted
scientific changes that a commit hash alone would miss.

The loader rejects architecture drift from the pinned model card values and
uses the explicit Hugging Face layout:

~~~text
backbone.layers      residual blocks
backbone.norm_f      final normalization
backbone.embeddings  token embedding
lm_head              unembedding
~~~

## Reproduction levels

Use three deliberately different runs:

| Run | Prompts | Source layers | Purpose |
|---|---:|---|---|
| Smoke | 4–8 | <code>0,25,50</code> | Exercise loading, checkpoints, merge, and UI only |
| Pilot | 100 | <code>all</code> | Check whether the lens is usable and inspect the target example |
| Full | 1,000 | <code>all</code> | Match the public fitter's paper-scale default |

The upstream authors report that roughly 100 prompts can be usable, but this
is not a guarantee for Nemotron. A smoke lens is never evidence for the
global-workspace claim. Do not merge runs with different corpora, source
layers, target layers, sequence lengths, or skip counts; the merger rejects
those metadata mismatches.

## Required gates

### 1. Software and provenance

Before installing this project, activate a GPU environment with
CUDA-compatible <code>mamba-ssm</code> and
<code>causal-conv1d</code>. The pinned remote model imports both even when the
naive Torch fallback is selected. Prefer the model card's NVIDIA NeMo
25.11.01 environment; otherwise provide builds matched to the actual
Torch/CUDA stack. The model card reports Transformers 4.57.3 as its tested
Hugging Face reference. Do not assume an arbitrary dependency resolution is
equivalent: record it and run preflight.

Verify the imports, install the project, capture the environment, and print
the immutable model specification before allocating a job:

~~~bash
python -c "import mamba_ssm, causal_conv1d"
pip install -e '.[dev]'
python -c "import transformers; assert transformers.__version__ == '4.57.3'"
nemotron-jlens info > artifacts/info.json
~~~

The project dependency is pinned to the checkpoint vendor's tested
Transformers 4.57.3, so installing it does not silently replace the working
NeMo runtime with a newer major release.

Retain <code>artifacts/info.json</code>, the corpus JSONL and its
<code>.meta.json</code>, every lens <code>.meta.json</code>, job logs,
CUDA/driver and dependency versions, and the final demo fixture.

### 2. Native-logit and full-stack VJP preflight

~~~bash
nemotron-jlens preflight --device-map auto
~~~

This must pass both checks:

- explicit <code>norm_f</code> + <code>lm_head</code> logits match the native
  Hugging Face logits within the configured tolerance; and
- one finite, nonzero VJP runs from source block 0 through target block 51.

The second check is intentionally end to end: it catches an unsupported fused
Mamba backward before a long fit. If and only if that failure implicates the
fused Mamba kernel, retry with:

~~~bash
nemotron-jlens preflight --device-map auto --disable-mamba-kernels
~~~

The fallback patches every config-declared
<code>NemotronHMamba2Mixer</code> onto that remote mixer's own naive
<code>torch_forward</code> and replaces its fused gated group RMSNorm with a
pure-Torch grouped RMSNorm. It counts patched layers and fails closed if the
pinned remote layout, fallback method, norm interface, or count has drifted.
For this pin, a fallback preflight must report
<code>mamba_backend=torch</code> and
<code>patched_mamba_layers=23</code>.

The naive path is slower and can use more memory. Its preflight on the exact
fit topology is mandatory even if the fused preflight fit comfortably. Carry
the same flag into every fit shard and downstream model-loading export/render
command. Do not proceed on a zero, non-finite, or truncated gradient, and do
not combine artifacts from different Mamba backends.

### 3. Deterministic fitting and merge

Prepare either the pilot or full corpus, then fit disjoint round-robin prompt
shards. A representative full run is:

~~~bash
nemotron-jlens prepare-corpus \
  artifacts/corpora/wikitext-1000.jsonl --n-prompts 1000

nemotron-jlens fit artifacts/corpora/wikitext-1000.jsonl \
  artifacts/shards/full-0.pt \
  --shard-index 0 --num-shards 8 \
  --source-layers all --target-layer -1 \
  --max-seq-len 128 --skip-first 16 --dim-batch 8
~~~

Run shard indices 0–7 with identical scientific and loader arguments, then:

~~~bash
nemotron-jlens merge artifacts/nemotron-1000.pt \
  artifacts/shards/full-0.pt artifacts/shards/full-1.pt \
  artifacts/shards/full-2.pt artifacts/shards/full-3.pt \
  artifacts/shards/full-4.pt artifacts/shards/full-5.pt \
  artifacts/shards/full-6.pt artifacts/shards/full-7.pt

nemotron-jlens validate artifacts/nemotron-1000.pt \
  > artifacts/nemotron-1000.validation.json
~~~

Validation checks prompt count, layer set, matrix shape, finite entries,
nonzero norms, and the artifact checksum recorded in the sidecar. Merge
accepts only a complete partition of direct shard artifacts: exactly one of
every declared shard index, never a pre-merged input or subset. It verifies
round-robin prompt-index ownership, rejects duplicate indices/hashes, requires
the combined prompt indices to be contiguous, and streams one lens at a time.
It also enforces model/data/fit settings, Mamba backend and patched count, and
<code>adaptation_source_sha256</code>.

### 4. Produce both demo artifacts

Export a real Neuronpedia-style fixture from the validated lens:

~~~bash
nemotron-jlens export-fixture artifacts/nemotron-1000.pt \
  demo/fixtures/nemotron-3-nano.recorded.json \
  --example modulation-topic --top-n 8 --layer-stride 1 \
  --tracked-token ' fish' --tracked-token ' whale' \
  --tracked-token ' coral' --tracked-token ' lobster'
~~~

Also render the official upstream layer-by-position viewer:

~~~bash
nemotron-jlens render artifacts/nemotron-1000.pt \
  artifacts/static-demo --example modulation-topic \
  --top-n 8 --layer-stride 1
~~~

The two outputs answer different needs: the upstream renderer is the closest
methodological comparison to the paper, while the fixture drives the requested
transcript/readout explorer. Both must name the exact lens checksum and model
revision before they are considered recorded evidence.

## Directed-modulation acceptance test

The primary user message is exactly:

> Write "She carefully placed the letter back inside the wooden drawer."
> Concentrate on ocean creatures while you write the sentence. Don't write
> anything else.

The teacher-forced assistant text is exactly:

> She carefully placed the letter back inside the wooden drawer.

Teacher forcing keeps the visible sentence fixed while the internal readout is
compared across conditions. Do not copy Qwen's numeric token positions into a
Nemotron report: derive the assistant span with Nemotron's pinned tokenizer.

At minimum record three conditions with identical formatting and assistant
text:

1. **Focus:** the exact ocean-creatures prompt above.
2. **No-side-task control:** request only the fixed sentence and “Don't write
   anything else.”
3. **Suppression control:** ask the model not to think about ocean creatures
   while writing the same sentence. Store the exact wording in the result.

Inspect both Jacobian-lens and ordinary logit-lens readouts, all fitted layers,
and the entire assistant span. Pre-register <code> fish</code>,
<code> whale</code>, <code> coral</code>, and <code> lobster</code> as tracked
displays, while preserving the exact Nemotron token IDs and tokenizer
decomposition in the artifact. Useful summaries are rank, probability, best
layer, best response position, and focus-minus-control change for each tracked
token.

A successful *execution* means all conditions and artifacts are recorded and
auditable. A successful *positive result* requires the focus condition to show
a coherent, reproducible ocean-creature signal relative to both controls. If
it does not, report the null or contrary result; do not tune the prompt,
tracked words, layer window, or token span after looking at the answer without
labeling that analysis exploratory.

## Evidence checklist

- [ ] Pinned model and dataset revisions appear in all sidecars.
- [ ] Native-logit parity and source-0-to-target-51 VJP preflight pass.
- [ ] Corpus manifest and every prompt hash verify.
- [ ] All expected disjoint shards finish and their metadata agree.
- [ ] Merged lens has 1,000 prompts, layers 0–50, finite 2,688×2,688 matrices.
- [ ] Final lens checksum is recorded in validation and both demo outputs.
- [ ] Focus, no-side-task, and suppression prompts are archived verbatim.
- [ ] Readout uses Nemotron tokenizer positions rather than Qwen positions.
- [ ] The explorer exposes Jacobian, logit, and difference views.
- [ ] Any result is labeled positive, null, or exploratory from actual data.

See [the compute runbook](compute-runbook.md) for operational commands and
[the demo specification](demo.md) for UI and scientific acceptance details.
