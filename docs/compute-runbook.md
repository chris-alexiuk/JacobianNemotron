# Compute runbook

This is the operational path from an empty artifact directory to a validated
Nemotron Jacobian lens and two demo outputs. Read the
[reproduction protocol](reproduction.md) first; it defines what can and cannot
be claimed from the run.

## Capacity planning

The pinned checkpoint has 31.6B total parameters but only about 3.6B active
parameters per token. Sparse activation reduces compute; it does **not** remove
the need to store the full checkpoint.

For the default target layer, there are 51 dense 2,688×2,688 Jacobian matrices:

| Item | Approximate size |
|---|---:|
| Dense lens at 16 bits | 0.686 GiB (about 703 MiB) |
| Dense lens at 32 bits | 1.373 GiB |
| Fit steady core: FP32 running sum + current per-prompt scratch | 2.745 GiB host RAM |
| Fit-finalization core peak: sum + last scratch + new FP32 mean | 4.118 GiB host RAM |
| Streaming-merge core peak: FP32 accumulator + one loaded input | 2.745 GiB host RAM |
| FP16 serialization copy during save | up to 0.686 GiB additional |
| BF16 model weights alone | 58.9 GiB (63.2 GB decimal) |

These are core tensor-payload estimates, not process-RSS limits. The fitter's
usual CPU state is the 2.745 GiB running sum plus prompt scratch. At
finalization, the last scratch remains live while the 1.373 GiB mean is
constructed, producing the roughly 4.118 GiB core peak. The merger streams one
input at a time, so its accumulation core is roughly 2.745 GiB; saving can
materialize another roughly 0.686 GiB FP16 serialization copy. Allocator
retention can make these phases overlap in observed RSS, so provision
additional headroom.

All estimates exclude Python objects, validation temporaries, checkpoint I/O
buffers, tokenizer and model loading, CUDA kernels, MoE routing, activation
graphs, allocator fragmentation, and Hugging Face device-map overhead. They
are not GPU-memory promises.

With the default dimension batch of 8, each prompt requires
<code>ceil(2688 / 8) = 336</code> backward sweeps. Therefore:

| Run | Prompts | Backward sweeps |
|---|---:|---:|
| Smoke | 8 | 2,688 |
| Pilot | 100 | 33,600 |
| Full | 1,000 | 336,000 |
| One of 8 full shards | 125 | 42,000 |

Start planning with **two 80 GiB GPUs per fitter process** plus ample system
RAM and fast local model storage. A single 80 GiB GPU may or may not fit a
specific runtime; only the preflight on the intended topology answers that.
The recommendation is conservative planning guidance, not a measured benchmark
from this checkout. Record actual peak allocation and elapsed time from the
preflight before estimating the queue request.

## 1. Set up one immutable environment

Start from a GPU environment that already provides CUDA-compatible
<code>mamba-ssm</code> and <code>causal-conv1d</code> builds matched to its
Torch/CUDA stack. The pinned remote modeling module imports both packages
during model import, so they are required even when
<code>--disable-mamba-kernels</code> later selects the naive Torch execution
path. Prefer the model card's NVIDIA NeMo 25.11.01 environment. If using a
different image, resolve the CUDA/Torch-specific packages before installing
this project; this runbook intentionally does not invent universal wheel
versions.

~~~bash
# Run inside the pre-provisioned GPU environment.
python -c "import mamba_ssm, causal_conv1d"
pip install -e '.[dev]'
python -c "import transformers; assert transformers.__version__ == '4.57.3'"

mkdir -p artifacts/corpora artifacts/shards artifacts/logs
nemotron-jlens info | tee artifacts/info.json
~~~

The project dependency pins the model card's tested Transformers 4.57.3 rather
than upgrading the NeMo image to an unrelated major release. The assertion
above fails immediately if an installer leaves a conflicting version in place.
Record the actual Transformers, Torch, CUDA, <code>mamba-ssm</code>, and
<code>causal-conv1d</code versions used, and require preflight to pass on that
exact environment.

For a cluster, put the Hugging Face cache on fast storage with enough capacity
for the complete checkpoint. Set <code>HF_HOME</code> consistently, accept any
applicable model terms, and authenticate before the scheduled job. Freeze the
Python environment after preflight; do not upgrade Torch, Transformers, CUDA,
or the remote model code between shards.

Each new shard records <code>adaptation_source_sha256</code>, a content hash of
the local <code>jlens</code> and <code>nemotron_jlens</code> scientific source
bundle. Keep the working tree identical across workers. The hash complements
the upstream commit/model revision pins, propagates into merged/demo
provenance, and is a merge invariant: shards produced by different adaptation
source cannot be combined.

The loader pins:

~~~text
model     nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
revision  cbd3fa9f933d55ef16a84236559f4ee2a0526848
dtype     bfloat16
~~~

## 2. Run preflight on the actual fit topology

~~~bash
nemotron-jlens preflight \
  --device-map auto \
  | tee artifacts/logs/preflight.json
~~~

Do not submit the full array unless <code>logit_parity.ok</code> and
<code>vjp.ok</code> are both true. Check:

- source layer is 0 and target layer is 51;
- gradient norm is finite and nonzero;
- source/target devices are expected;
- native/adapted logit maximum error is within tolerance; and
- CUDA peak allocation leaves operational headroom.

If the VJP fails specifically in a fused Mamba backward kernel:

~~~bash
nemotron-jlens preflight \
  --device-map auto \
  --disable-mamba-kernels \
  | tee artifacts/logs/preflight-torch-mamba.json
~~~

This is a fail-closed implementation patch, not merely a configuration flag.
The loader reads the pinned remote config's Mamba block count, finds every
<code>NemotronHMamba2Mixer</code>, replaces that mixer's effective forward
method with its own naive <code>torch_forward</code>, and replaces its fused
gated group RMSNorm with a differentiable pure-Torch grouped RMSNorm. It
refuses to load if the config lacks the block layout, a mixer lacks the known
fallback/norm interface, or the patched count differs from the config-declared
count. For the pinned architecture, the reported
<code>patched_mamba_layers</code> should be 23 and
<code>mamba_backend</code> should be <code>torch</code>.

The naive path can be substantially slower and more memory hungry than fused
kernels. A successful fallback preflight on the exact fit topology is
mandatory; never infer that it will fit from the fused-path result. If it
passes, add <code>--disable-mamba-kernels</code> to every fit shard and to
model-loading export/render commands. Do not mix backends across shards. The
backend and patched-layer count are recorded and enforced during merge.

If fallback memory fails, try <code>--dim-batch 4</code> in fitting only after
the fallback preflight succeeds; this roughly doubles the number of backward
sweeps. Keep the chosen dimension batch constant across a reported run.

<code>--compile-blocks</code> is an opt-in experiment and requires
<code>--device-map cuda</code>. Do not introduce compilation in the full run
unless the complete preflight and a smoke fit pass with that exact setting.

## 3. Materialize and freeze the corpus

Pilot:

~~~bash
nemotron-jlens prepare-corpus \
  artifacts/corpora/wikitext-100.jsonl \
  --n-prompts 100
~~~

Full:

~~~bash
nemotron-jlens prepare-corpus \
  artifacts/corpora/wikitext-1000.jsonl \
  --n-prompts 1000
~~~

Each command creates both a JSONL and a <code>.meta.json</code> sidecar. Copy
both files unchanged to every worker. The loader verifies the manifest hash,
prompt hashes, order, and count before loading the model. Avoid
<code>--force</code> after any shard has started; a regenerated corpus defines
a different run even if its filename is unchanged.

Corpus preparation first materializes the pinned dataset revision in the
Hugging Face cache as disk-backed, memory-mapped Arrow data. This deliberate
non-streaming path avoids streaming-iterator teardown failures in container
runtimes without retaining the full dataset in RAM; provision cache disk space
on the corpus-preparation worker.

## 4. Smoke-test checkpoint and merge behavior

Create a small independent corpus:

~~~bash
nemotron-jlens prepare-corpus \
  artifacts/corpora/wikitext-8.jsonl \
  --n-prompts 8
~~~

Fit two disjoint shards using sparse source layers:

~~~bash
nemotron-jlens fit artifacts/corpora/wikitext-8.jsonl \
  artifacts/shards/smoke-0.pt \
  --shard-index 0 --num-shards 2 \
  --source-layers 0,25,50 \
  --target-layer -1 --max-seq-len 128 --skip-first 16 \
  --dim-batch 8 --checkpoint-every 2 --device-map auto \
  | tee artifacts/logs/smoke-0.json

nemotron-jlens fit artifacts/corpora/wikitext-8.jsonl \
  artifacts/shards/smoke-1.pt \
  --shard-index 1 --num-shards 2 \
  --source-layers 0,25,50 \
  --target-layer -1 --max-seq-len 128 --skip-first 16 \
  --dim-batch 8 --checkpoint-every 2 --device-map auto \
  | tee artifacts/logs/smoke-1.json

nemotron-jlens merge artifacts/smoke.pt \
  artifacts/shards/smoke-0.pt artifacts/shards/smoke-1.pt
nemotron-jlens validate artifacts/smoke.pt \
  | tee artifacts/logs/smoke-validation.json
~~~

A smoke run tests the machinery only. Its values must not appear as the final
scientific result.

## 5. Pilot before full scale

The pilot should use all source layers because the requested demo needs a
layer trajectory:

~~~bash
nemotron-jlens fit artifacts/corpora/wikitext-100.jsonl \
  artifacts/pilot.pt \
  --source-layers all --target-layer -1 \
  --max-seq-len 128 --skip-first 16 --dim-batch 8 \
  --checkpoint-every 10 --device-map auto \
  | tee artifacts/logs/pilot.json

nemotron-jlens validate artifacts/pilot.pt \
  | tee artifacts/logs/pilot-validation.json
~~~

Render the directed-modulation page from the pilot and inspect whether the
artifact is structurally useful before spending on 1,000 prompts:

~~~bash
nemotron-jlens render artifacts/pilot.pt artifacts/pilot-static \
  --example modulation-topic --top-n 8 --layer-stride 1 \
  --device-map auto
~~~

The upstream observation that around 100 prompts can be usable is a planning
hint, not a pass criterion. Record pilot behavior even if it is noisy.

## 6. Run the full fit as disjoint jobs

For eight workers, each worker runs one value of <code>SHARD_INDEX</code> from
0 through 7:

~~~bash
SHARD_INDEX=0
NUM_SHARDS=8

nemotron-jlens fit artifacts/corpora/wikitext-1000.jsonl \
  "artifacts/shards/full-$SHARD_INDEX.pt" \
  --shard-index "$SHARD_INDEX" --num-shards "$NUM_SHARDS" \
  --source-layers all --target-layer -1 \
  --max-seq-len 128 --skip-first 16 --dim-batch 8 \
  --checkpoint-every 10 --device-map auto \
  | tee "artifacts/logs/full-$SHARD_INDEX.json"
~~~

This is process-level parallelism: each worker loads a complete logical model
and writes a different artifact. Do not launch two processes against the same
output or checkpoint path.

The fitter resumes from
<code>full-N.pt.checkpoint.pt</code>. By default that checkpoint is removed
after a successful final save. Use <code>--keep-checkpoint</code> when cluster
policy or debugging requires it. A retry must use the identical corpus and
scientific settings. Save scheduler logs separately from the CLI JSON output.

## 7. Audit shards and merge once

Confirm that all eight lens files and all eight metadata sidecars exist. Then:

~~~bash
nemotron-jlens merge artifacts/nemotron-1000.pt \
  artifacts/shards/full-0.pt artifacts/shards/full-1.pt \
  artifacts/shards/full-2.pt artifacts/shards/full-3.pt \
  artifacts/shards/full-4.pt artifacts/shards/full-5.pt \
  artifacts/shards/full-6.pt artifacts/shards/full-7.pt \
  | tee artifacts/logs/full-merge.json

nemotron-jlens validate artifacts/nemotron-1000.pt \
  | tee artifacts/nemotron-1000.validation.json
~~~

The local merger computes a prompt-count-weighted average while loading and
releasing one input lens at a time. It accepts only a **complete direct-shard
partition**: every input must have <code>kind=shard</code>, all must declare
the same <code>num_shards</code>, and the input set must contain each
<code>shard_index</code> from 0 through <code>num_shards - 1</code> exactly
once. It rejects pre-merged inputs, missing indices, and duplicate indices.

For current fitter artifacts, every shard also carries corpus
<code>prompt_indices</code>. Merge verifies that each index belongs to its
round-robin shard, no index or prompt hash is duplicated, and the combined
indices form the complete contiguous range from zero through the merged prompt
count minus one. Scientific invariants include model/revision, dataset
manifest, layer/fit settings, Mamba backend and patched count, and
<code>adaptation_source_sha256</code>. Never work around a mismatch or an
incomplete partition by editing sidecars.

Archive this atomic set:

~~~text
nemotron-1000.pt
nemotron-1000.pt.meta.json
nemotron-1000.validation.json
wikitext-1000.jsonl
wikitext-1000.jsonl.meta.json
all shard sidecars and logs
~~~

## 8. Export the requested explorer fixture

The export runs a real teacher-forced Nemotron forward pass and derives both
Jacobian-lens and logit-lens top-k values:

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

Export the neutral, suppression, and mention-only controls with identical
display settings:

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

Create the auditable focus-versus-control summary on CPU:

~~~bash
nemotron-jlens compare-fixtures \
  demo/fixtures/nemotron-3-nano.recorded.json \
  --control neutral=demo/fixtures/modulation-topic-neutral.recorded.json \
  --control suppression=demo/fixtures/modulation-topic-suppress.recorded.json \
  --control mention=demo/fixtures/modulation-topic-mention-control.recorded.json \
  --output artifacts/directed-modulation-comparison.json
~~~

The comparator stops on non-recorded fixtures or incompatible model revision,
lens SHA-256, or layer sets. It does not infer missing tracked measurements
from displayed top-k rows.

If the complete chat template is too long for an exploratory export, use
<code>--last-n-tokens</code> only as an explicitly labeled display reduction;
do not silently change the fitted lens or acceptance span. Keep the recorded
fixture beside the lens checksum from which it was derived.

Serve the custom explorer:

~~~bash
nemotron-jlens serve-demo --directory demo --host 127.0.0.1 --port 8000
~~~

Open <http://127.0.0.1:8000>, choose **Import fixture**, and load
<code>nemotron-3-nano.recorded.json</code>. The default sample remains
illustrative until intentionally replaced.

## 9. Render the official upstream viewer

~~~bash
nemotron-jlens render artifacts/nemotron-1000.pt \
  artifacts/static-demo \
  --example modulation-topic \
  --top-n 8 --layer-stride 1 \
  --max-seq-len 512 --max-tracked 512 \
  --device-map auto

nemotron-jlens serve-demo \
  --directory artifacts/static-demo \
  --host 127.0.0.1 --port 8001
~~~

The renderer writes <code>index.html</code>, compressed payload data, and
<code>build.json</code>. Preserve the entire directory. This page is the
closest visual comparison to Anthropic's released implementation; the custom
explorer is the closest interaction comparison to the requested Neuronpedia
view.

## Failure policy

- **Model/config mismatch:** stop. Verify model ID and exact revision; do not
  relax architecture validation.
- **Logit-parity failure:** stop. The residual layout or target normalization
  is wrong.
- **Fused-Mamba backward failure:** rerun preflight with the documented torch
  fallback. Confirm the expected patched count, then keep that backend for
  every fit shard and downstream model-loading step.
- **Torch-Mamba fallback drift:** stop if the remote config/mixer layout,
  per-mixer <code>torch_forward</code>, grouped RMSNorm interface, or patched
  count differs. Do not weaken the fail-closed checks.
- **Torch-Mamba fallback OOM/timeout:** request more capacity or explicitly
  revise the plan after measuring preflight; the naive path is expected to be
  slower and more memory hungry.
- **CUDA OOM:** first reduce dimension batch and repeat smoke/pilot; otherwise
  add capacity. Record the changed setting.
- **NaN/Inf or zero matrix:** reject the shard; retain logs and investigate.
- **Corpus or artifact checksum mismatch:** restore the immutable source;
  never recompute a hash into the sidecar.
- **Merge overlap/mismatch:** identify the duplicated or drifted job and rerun
  it. Do not force a merge.
- **Incomplete direct-shard partition:** supply exactly one direct shard for
  every declared index; do not merge a subset, add a pre-merged artifact, or
  edit shard coordinates.
- **No directed-modulation signal:** report a null result. It is not an
  infrastructure failure if preflight, fit, validation, and controls passed.
