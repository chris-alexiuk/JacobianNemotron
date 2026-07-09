# Finish the Nemotron 3 Nano Jacobian lens

This is the handoff procedure for replacing the accepted 100-prompt pilot with
the canonical 1,000-prompt, all-layer lens. It is written for a bare-metal host
with two or three 80 GiB H100 GPUs. Run every command from the repository root.

Do not extend or merge the pilot. The final lens must be fitted from the
canonical 1,000-prompt corpus as eight new, disjoint shards.

## 1. Freeze the scientific identity

These values are acceptance inputs, not suggestions:

| Item | Required value |
|---|---|
| Model | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| Model revision | `cbd3fa9f933d55ef16a84236559f4ee2a0526848` |
| WikiText revision | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| Corpus manifest SHA-256 | `c75fc7ee5d92335f0620a08c6f87d210fc3b3fe4f3d6bcfae7d0cd864a88d63e` |
| Fit-source SHA-256 | `7d4f586302eec297d714a1b790327d19f56932faae4cf9552cd344ac66766fad` |
| Upstream J-lens commit | `581d398613e5602a5af361e1c34d3a92ea82ba8e` |
| NeMo image | `nvcr.io/nvidia/nemo:25.11.01@sha256:0adfd600a7e5d62bb71c751fa3fb712cf3ced38e9b6e4718ed609eae404e223b` |
| Transformers | `4.57.3` |
| Model dtype | BF16 |
| Stored lens dtype | FP32 |
| Source layers | `0..50` |
| Target layer | `51` (`--target-layer -1`) |
| Prompts | `1000` |
| Sequence length | `128` |
| Skipped positions | first `16` |
| Dimension batch | `8` |
| Shards | `8` round-robin shards |
| Mamba backend | `fused-or-auto`, zero patched layers |

Do not edit Python under `jlens/` or `nemotron_jlens/` after this point. Those
directories define the fit-source hash embedded in every shard. Do not upgrade
Torch, Transformers, CUDA, `mamba-ssm`, `causal-conv1d`, or remote model code
between preflight, shards, export, and render.

Check the source and host before allocating the long run:

```bash
set -euo pipefail

test "$(uv run nemotron-jlens info | jq -r .adaptation_source_sha256)" = \
  7d4f586302eec297d714a1b790327d19f56932faae4cf9552cd344ac66766fad

uv run pytest -q
uv run ruff check .
node --test demo/tests/*.test.js steering_demo/tests/*.test.js
nvidia-smi --query-gpu=index,name,memory.total,uuid \
  --format=csv,noheader
```

Stop if the source hash differs. Restore the exact fitting source instead of
changing the expected hash.

## 2. Start the pinned bare-metal container

Set the host paths once. `HF_CACHE` must have room for the complete model
snapshot and WikiText Arrow cache.

```bash
export REPO="$(pwd)"
export HF_CACHE="${HOME}/.cache/huggingface/hub"
export HF_DATASETS_CACHE="${HOME}/.cache/huggingface/datasets"
export NEMO_IMAGE='nvcr.io/nvidia/nemo:25.11.01@sha256:0adfd600a7e5d62bb71c751fa3fb712cf3ced38e9b6e4718ed609eae404e223b'

mkdir -p "$HF_CACHE" "$HF_DATASETS_CACHE" \
  artifacts/corpora artifacts/shards artifacts/logs artifacts/archive
docker pull "$NEMO_IMAGE"

docker run --rm -it --name nemotron-jlens-full --init \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp -e PYTHONPATH=/workspace \
  -e HF_HUB_CACHE=/hf -e HF_DATASETS_CACHE=/hf-datasets \
  -e TOKENIZERS_PARALLELISM=false \
  -p 8000:8000 -p 8001:8001 \
  -v "$REPO:/workspace" -v "$HF_CACHE:/hf" \
  -v "$HF_DATASETS_CACHE:/hf-datasets" \
  -w /workspace --entrypoint bash "$NEMO_IMAGE"
```

All remaining fitting commands in this guide run inside that container. Install
the repository without changing the pinned scientific runtime, then materialize
the exact model snapshot:

```bash
set -euo pipefail
export PATH="/tmp/.local/bin:$PATH"
python -c 'import mamba_ssm, causal_conv1d'
python -m pip install -e '.[dev]'
python -c 'import transformers; assert transformers.__version__ == "4.57.3"'

python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download(
    repo_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    revision="cbd3fa9f933d55ef16a84236559f4ee2a0526848",
    cache_dir="/hf",
))
PY

nemotron-jlens info | tee artifacts/info-full.json
test "$(jq -r .adaptation_source_sha256 artifacts/info-full.json)" = \
  7d4f586302eec297d714a1b790327d19f56932faae4cf9552cd344ac66766fad
```

If the host already supplies this exact environment without Docker, the same
commands may run directly. Record the full package, CUDA, driver, and GPU
identity; do not treat a merely similar environment as equivalent.

## 3. Preflight every fitter GPU

Each fitter process must see exactly one physical GPU. With
`CUDA_VISIBLE_DEVICES`, that device is remapped to `cuda:0` inside the process.
Use `--device-map cuda` so insufficient HBM fails instead of silently offloading
part of the model to CPU.

For a two-GPU host set `GPU_IDS=(0 1)`. For a three-GPU host set
`GPU_IDS=(0 1 2)`.

```bash
GPU_IDS=(0 1)  # Change to: GPU_IDS=(0 1 2) for three H100s.

for gpu in "${GPU_IDS[@]}"; do
  CUDA_VISIBLE_DEVICES="$gpu" nemotron-jlens preflight \
    --device-map cuda --cache-dir /hf \
    > "artifacts/logs/full-preflight-gpu${gpu}.json" \
    2> "artifacts/logs/full-preflight-gpu${gpu}.stderr.log"

  jq -e '
    .model_id == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16" and
    .revision == "cbd3fa9f933d55ef16a84236559f4ee2a0526848" and
    .adaptation_source_sha256 == "7d4f586302eec297d714a1b790327d19f56932faae4cf9552cd344ac66766fad" and
    .mamba_backend == "fused-or-auto" and
    .patched_mamba_layers == 0 and
    .logit_parity.ok == true and
    .vjp.ok == true and
    (.vjp.gradient_norm > 0)
  ' "artifacts/logs/full-preflight-gpu${gpu}.json" >/dev/null
done
```

The measured H200 reference peaked near 59.3 GiB for preflight and 64.0 GiB
for an all-layer one-prompt fit. An 80 GiB H100 is plausible but must pass on
the actual host. Do not run two fitters on one GPU.

The CLI has a documented Torch Mamba fallback. That path is useful for
diagnosis, but the current paper-scale acceptance profile requires
`fused-or-auto` and zero patched layers. If fused VJP fails, stop. Do not run
the final array with `--disable-mamba-kernels` and call it accepted without a
separately reviewed protocol and acceptance-code change.

## 4. Run an all-layer smoke fit

This catches peak-memory and checkpoint failures that the one-VJP preflight
cannot. It is disposable and must never be merged into the full lens.

```bash
nemotron-jlens prepare-corpus \
  artifacts/corpora/wikitext-1-smoke.jsonl --n-prompts 1

CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" nemotron-jlens fit \
  artifacts/corpora/wikitext-1-smoke.jsonl \
  artifacts/shards/full-smoke.pt \
  --source-layers all --target-layer -1 \
  --max-seq-len 128 --skip-first 16 --dim-batch 8 \
  --checkpoint-every 1 --device-map cuda --cache-dir /hf \
  > artifacts/logs/full-smoke.json \
  2> artifacts/logs/full-smoke.stderr.log

nemotron-jlens validate artifacts/shards/full-smoke.pt \
  > artifacts/logs/full-smoke.validation.json
jq -e '.ok == true and .n_prompts == 1 and (.source_layers | length) == 51' \
  artifacts/logs/full-smoke.validation.json >/dev/null
```

Confirm the process did not OOM, the sidecar reports `fused-or-auto`, every
matrix is finite FP32 with shape 2688x2688, and the source hash is unchanged.

## 5. Create and freeze the canonical corpus

Create this once. Never use `--force` after any full shard starts.

```bash
nemotron-jlens prepare-corpus \
  artifacts/corpora/wikitext-1000.jsonl --n-prompts 1000 \
  | tee artifacts/logs/full-corpus.json

jq -e '
  .n_prompts == 1000 and
  .dataset.id == "Salesforce/wikitext" and
  .dataset.revision == "b08601e04326c79dfdd32d625aee71d232d685c3" and
  .manifest_sha256 == "c75fc7ee5d92335f0620a08c6f87d210fc3b3fe4f3d6bcfae7d0cd864a88d63e"
' artifacts/corpora/wikitext-1000.jsonl.meta.json >/dev/null

test "$(wc -l < artifacts/corpora/wikitext-1000.jsonl)" -eq 1000
cp -a artifacts/corpora/wikitext-1000.jsonl \
  artifacts/corpora/wikitext-1000.jsonl.meta.json artifacts/archive/
```

Copy both corpus files unchanged to another machine only if fitting spans
hosts. Every host must use the same repository source and scientific runtime.

## 6. Fit eight resumable shards

The complete fit is 336,000 backward sweeps. The measured H200 all-layer smoke
was roughly 5.6 minutes per prompt, so budget about 109 aggregate GPU-hours
before retries. Two comparable GPUs imply roughly 55 wall-clock hours; three
imply roughly 37 to 41 hours. H100 timing is only an estimate until the smoke
fit finishes.

Use one lane per GPU. The assignment below runs four 125-prompt shards per GPU
on two GPUs, or three/three/two shards on three GPUs. Each failed command can be
rerun unchanged; it resumes from `full-N.pt.checkpoint.pt`.

```bash
set -u
NUM_SHARDS=8

fit_lane() {
  local gpu="$1"
  shift
  local shard
  for shard in "$@"; do
    echo "Starting shard ${shard} on physical GPU ${gpu}" >&2
    CUDA_VISIBLE_DEVICES="$gpu" nemotron-jlens fit \
      artifacts/corpora/wikitext-1000.jsonl \
      "artifacts/shards/full-${shard}.pt" \
      --shard-index "$shard" --num-shards "$NUM_SHARDS" \
      --source-layers all --target-layer -1 \
      --max-seq-len 128 --skip-first 16 --dim-batch 8 \
      --checkpoint-every 10 --keep-checkpoint \
      --device-map cuda --cache-dir /hf \
      > "artifacts/logs/full-${shard}.json" \
      2> "artifacts/logs/full-${shard}.stderr.log" || return 1
  done
}

pids=()
if [ "${#GPU_IDS[@]}" -eq 2 ]; then
  fit_lane "${GPU_IDS[0]}" 0 2 4 6 & pids+=("$!")
  fit_lane "${GPU_IDS[1]}" 1 3 5 7 & pids+=("$!")
elif [ "${#GPU_IDS[@]}" -eq 3 ]; then
  fit_lane "${GPU_IDS[0]}" 0 3 6 & pids+=("$!")
  fit_lane "${GPU_IDS[1]}" 1 4 7 & pids+=("$!")
  fit_lane "${GPU_IDS[2]}" 2 5   & pids+=("$!")
else
  echo 'GPU_IDS must contain exactly two or three GPU indices' >&2
  return 2 2>/dev/null || exit 2
fi

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
test "$rc" -eq 0
```

Monitor without modifying outputs:

```bash
watch -n 30 'nvidia-smi; ls -lh artifacts/shards/full-*.pt* 2>/dev/null'
```

Do not launch two commands with the same shard index or output path. Do not
delete a checkpoint after a failure. Resume only with the identical corpus,
source, runtime, backend, and fit arguments.

## 7. Audit every shard before merge

Run validation serially to keep host memory predictable:

```bash
for shard in {0..7}; do
  test -s "artifacts/shards/full-${shard}.pt"
  test -s "artifacts/shards/full-${shard}.pt.meta.json"
  nemotron-jlens validate "artifacts/shards/full-${shard}.pt" \
    > "artifacts/shards/full-${shard}.validation.json"
done
```

Then enforce the cross-shard contract:

```bash
python - <<'PY'
import json
from pathlib import Path

SOURCE = "7d4f586302eec297d714a1b790327d19f56932faae4cf9552cd344ac66766fad"
CORPUS = "c75fc7ee5d92335f0620a08c6f87d210fc3b3fe4f3d6bcfae7d0cd864a88d63e"
MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
REVISION = "cbd3fa9f933d55ef16a84236559f4ee2a0526848"

runtime = None
seen = set()
for shard in range(8):
    path = Path(f"artifacts/shards/full-{shard}.pt.meta.json")
    m = json.loads(path.read_text())
    assert m["kind"] == "shard"
    assert m["shard_index"] == shard and m["num_shards"] == 8
    assert m["n_prompts"] == 125
    assert m["prompt_indices"] == list(range(shard, 1000, 8))
    assert m["source_layers"] == list(range(51)) and m["target_layer"] == 51
    assert m["max_seq_len"] == 128 and m["skip_first"] == 16
    assert m["dim_batch"] == 8 and m["dtype"] == "bfloat16"
    assert m["storage_dtype"] == "float32"
    assert m["mamba_backend"] == "fused-or-auto"
    assert m["patched_mamba_layers"] == 0
    assert m["model_id"] == MODEL and m["model_revision"] == REVISION
    assert m["corpus_manifest_sha256"] == CORPUS
    assert m["adaptation_source_sha256"] == SOURCE
    assert m["validation"]["ok"] is True
    assert m["acceptance"]["is_final"] is False
    runtime = runtime or m["runtime"]
    assert m["runtime"] == runtime
    assert not seen.intersection(m["prompt_indices"])
    seen.update(m["prompt_indices"])

assert seen == set(range(1000))
print("all eight shards form one valid 1,000-prompt partition")
PY
```

The direct shards are non-final by themselves. That is expected. Stop on any
mismatch; rerun the bad shard instead of editing its sidecar.

## 8. Merge once and accept the final lens

Merge on a host with ample RAM and local disk. The merger streams inputs but
still needs several GiB of working memory.

```bash
nemotron-jlens merge artifacts/nemotron-1000.pt \
  artifacts/shards/full-0.pt artifacts/shards/full-1.pt \
  artifacts/shards/full-2.pt artifacts/shards/full-3.pt \
  artifacts/shards/full-4.pt artifacts/shards/full-5.pt \
  artifacts/shards/full-6.pt artifacts/shards/full-7.pt \
  | tee artifacts/logs/full-merge.json

nemotron-jlens validate artifacts/nemotron-1000.pt \
  > artifacts/nemotron-1000.validation.json

jq -e '
  .ok == true and .n_prompts == 1000 and .dtype == "float32" and
  .source_layers == [range(0; 51)] and
  .metadata.kind == "merged" and
  .metadata.n_prompts == 1000 and
  .metadata.storage_dtype == "float32" and
  .metadata.corpus_manifest_sha256 == "c75fc7ee5d92335f0620a08c6f87d210fc3b3fe4f3d6bcfae7d0cd864a88d63e" and
  .metadata.adaptation_source_sha256 == "7d4f586302eec297d714a1b790327d19f56932faae4cf9552cd344ac66766fad" and
  .metadata.acceptance.tier == "paper-scale" and
  .metadata.acceptance.status == "accepted" and
  .metadata.acceptance.is_final == true and
  .metadata.acceptance.exportable == true and
  ([.metadata.acceptance.checks[]] | all)
' artifacts/nemotron-1000.validation.json >/dev/null

export FULL_LENS_SHA="$(sha256sum artifacts/nemotron-1000.pt | awk '{print $1}')"
test "$FULL_LENS_SHA" = \
  "$(jq -r .artifact_sha256 artifacts/nemotron-1000.pt.meta.json)"
printf '%s  %s\n' "$FULL_LENS_SHA" artifacts/nemotron-1000.pt \
  | tee artifacts/nemotron-1000.sha256
```

Archive the lens, sidecar, validation, corpus and sidecar, all eight shard
sidecars, all result/stderr logs, `artifacts/info-full.json`, and the final
checksum. Keep the pilot files unchanged under their existing names.

## 9. Generate the real fixtures and browser renders

Model-loading export commands must use the same backend and exact model
runtime as fitting. One H100 is enough; run exports sequentially.

```bash
GPU="${GPU_IDS[0]}"
COMMON=(--top-n 8 --layer-stride 1 \
  --tracked-token ' fish' --tracked-token ' whale' \
  --tracked-token ' coral' --tracked-token ' lobster' \
  --device-map cuda --cache-dir /hf)

CUDA_VISIBLE_DEVICES="$GPU" nemotron-jlens export-fixture \
  artifacts/nemotron-1000.pt \
  demo/fixtures/nemotron-3-nano.recorded.json \
  --example modulation-topic "${COMMON[@]}"

for slug in modulation-topic-neutral modulation-topic-suppress \
  modulation-topic-mention-control; do
  CUDA_VISIBLE_DEVICES="$GPU" nemotron-jlens export-fixture \
    artifacts/nemotron-1000.pt "demo/fixtures/${slug}.recorded.json" \
    --example "$slug" "${COMMON[@]}"
done

nemotron-jlens compare-fixtures \
  demo/fixtures/nemotron-3-nano.recorded.json \
  --control neutral=demo/fixtures/modulation-topic-neutral.recorded.json \
  --control suppression=demo/fixtures/modulation-topic-suppress.recorded.json \
  --control mention=demo/fixtures/modulation-topic-mention-control.recorded.json \
  --output artifacts/directed-modulation-comparison.json

CUDA_VISIBLE_DEVICES="$GPU" nemotron-jlens render \
  artifacts/nemotron-1000.pt artifacts/static-demo \
  --example modulation-topic --top-n 8 --layer-stride 1 \
  --max-seq-len 512 --max-tracked 512 \
  --device-map cuda --cache-dir /hf
```

Fail if any fixture names another lens SHA, model revision, layer set, or
acceptance tier. A null or mixed focus/control result is a valid scientific
outcome; do not tune prompts or tracked tokens after inspecting it.

Serve and inspect both outputs:

```bash
nemotron-jlens serve-demo --directory demo --host 0.0.0.0 --port 8000
# From a second host shell, start the other server in the same container:
docker exec -d nemotron-jlens-full python -m nemotron_jlens.cli \
  serve-demo --directory artifacts/static-demo \
  --host 0.0.0.0 --port 8001
```

Open `http://127.0.0.1:8000`, import
`demo/fixtures/nemotron-3-nano.recorded.json`, and verify the custom explorer.
Open `http://127.0.0.1:8001` and verify the official layer-by-position render.
Record desktop and mobile browser screenshots and confirm that the displayed
lens checksum equals `$FULL_LENS_SHA`.

## 10. Promote the live app to the full lens

Do this only after Sections 1 through 9 pass. The current server deliberately
rejects anything except the pilot, so promotion is an explicit code change.

1. Preserve the pilot artifact and its sidecar. Make
   `artifacts/nemotron-1000.pt` the new server default; never overwrite the old
   bytes under the pilot filename.
2. In `nemotron_steering/constants.py`, replace `LENS_SHA256` with
   `$FULL_LENS_SHA`. Replace the pilot-specific disclosure constant with a
   neutral full-lens constant such as `LENS_DISCLOSURE = "1,000-prompt
   paper-scale accepted lens"`, then update all imports.
3. In `nemotron_steering/provenance.py`, require `n_prompts == 1000`, the
   canonical corpus manifest, FP32 storage, `fused-or-auto`, zero patched
   layers, and acceptance tuple `(paper-scale, accepted, true)`.
4. Change the default lens path and help text in
   `nemotron_steering/server.py`.
5. Replace pilot-only text in `steering_demo/index.html`,
   `steering_demo/app.js`, `nemotron_mood/backend.py`, service responses,
   smoke output, and the steering tests. Do not leave a result labeled pilot.
6. Update the provenance tests to validate the new immutable lens while adding
   a separate test that the old pilot artifact and SHA remain unchanged.
7. Start the service once with the final lens. The first mood request computes
   a new neutral-reference calibration whose identity includes the new lens
   SHA. Record that new calibration ID; never reuse the pilot calibration.
8. Record the new `application_source_sha256`. It is expected to change when
   `nemotron_steering/`, `nemotron_mood/`, or `steering_demo/` changes. The
   fit-source SHA must remain exactly `7d4f5863...`.

Run the complete test and real-browser gates after promotion:

```bash
python -m pytest -q
python -m ruff check .

python -m nemotron_steering.server \
  --lens artifacts/nemotron-1000.pt \
  --host 0.0.0.0 --port 8000 \
  --device-map auto --cache-dir /hf
```

With that service reachable on host port 8000, run the real browser harness
from a host shell that has Firefox and `geckodriver` installed:

```bash
mkdir -p artifacts
geckodriver --host 127.0.0.1 --port 4444 \
  > artifacts/geckodriver.log 2>&1 &
WEBDRIVER_PID=$!
trap 'kill "$WEBDRIVER_PID" 2>/dev/null || true' EXIT

python3 tests/run_steering_browser_smoke.py \
  --webdriver http://127.0.0.1:4444 \
  --url http://127.0.0.1:8000/ \
  --output artifacts/nano-steering-browser.json
```

Then require all of the following on the real Nano model:

- `/health` is ready and `/api/info` reports the exact full lens SHA, 1,000
  prompts, paper-scale acceptance, model revision, runtime, fit-source hash,
  and new live-application hash.
- Baseline generation, zero-strength parity, positive and negative steer,
  ablation, and one-token source-to-target swap all complete.
- A second clean baseline exactly matches the first; hooks return to zero and
  memory does not grow monotonically.
- The mood endpoint reports the new calibration ID and no leaked hooks.
- The browser works from another trusted-LAN machine for both desktop and
  mobile layouts, with all layers available and no pilot disclosure.

Do not expose port 8000 to the public internet. Use a trusted LAN, firewall, or
Tailscale ACL.

## 11. Final handoff bundle

The lens is complete only when this atomic bundle is archived:

- `artifacts/nemotron-1000.pt`
- `artifacts/nemotron-1000.pt.meta.json`
- `artifacts/nemotron-1000.validation.json`
- `artifacts/nemotron-1000.sha256`
- canonical corpus JSONL and sidecar
- all eight direct-shard sidecars, validation files, and logs
- preflight and smoke outputs for every fitter GPU type/topology
- environment identity and exact NeMo image digest
- four recorded fixtures and their focus/control comparison
- the complete `artifacts/static-demo/` directory
- promoted live-app source hash, fresh mood calibration identity, tests, real
  Nano smoke report, and browser screenshots

Do not declare completion from shard existence alone. The decisive gate is a
merged artifact classified `paper-scale / accepted / is_final=true`, followed
by genuine model exports and the real browser acceptance run.
