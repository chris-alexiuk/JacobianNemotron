"""Command-line entry point for the Nemotron 3 Nano reproduction."""

from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import jlens
from nemotron_jlens.artifacts import (
    merge_shards,
    metadata_path,
    read_metadata,
    validate_artifact,
    validate_lens,
)
from nemotron_jlens.comparison import compare_fixtures
from nemotron_jlens.config import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_REVISION,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_DIM_BATCH,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_N_PROMPTS,
    DEFAULT_SKIP_FIRST,
    DEFAULT_TEXT_FIELD,
    NEMOTRON,
    UPSTREAM_JLENS_COMMIT,
)
from nemotron_jlens.corpus import prepare_corpus
from nemotron_jlens.fixture import DEFAULT_TRACKED_TOKENS, export_fixture
from nemotron_jlens.loading import load_nemotron
from nemotron_jlens.pipeline import fit_shard, render_static_demo
from nemotron_jlens.preflight import check_native_logit_parity, run_vjp_preflight
from nemotron_jlens.provenance import adaptation_source_sha256


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Accelerate device map; use 'auto' for multi-GPU or 'cuda' for one GPU",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--compile-blocks", action="store_true")
    parser.add_argument(
        "--disable-mamba-kernels",
        action="store_true",
        help="Use the differentiable torch fallback when fused Mamba backward fails",
    )


def _control_spec(value: str) -> tuple[str, str]:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path:
        raise argparse.ArgumentTypeError("control must have the form LABEL=PATH")
    if label.strip() == "focus":
        raise argparse.ArgumentTypeError("the control label 'focus' is reserved")
    return label.strip(), path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nemotron-jlens",
        description="Reproduce the Jacobian-lens workspace analysis on Nemotron 3 Nano.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="print pinned revisions and resource estimates")
    info.add_argument("--dim-batch", type=int, default=DEFAULT_DIM_BATCH)

    corpus = sub.add_parser(
        "prepare-corpus", help="materialize the exact fitting prompts"
    )
    corpus.add_argument("output")
    corpus.add_argument("--n-prompts", type=int, default=DEFAULT_N_PROMPTS)
    corpus.add_argument("--dataset", default=DEFAULT_DATASET_ID)
    corpus.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    corpus.add_argument("--dataset-revision", default=DEFAULT_DATASET_REVISION)
    corpus.add_argument("--split", default=DEFAULT_DATASET_SPLIT)
    corpus.add_argument("--text-field", default=DEFAULT_TEXT_FIELD)
    corpus.add_argument("--max-chars", type=int, default=2000)
    corpus.add_argument("--min-chars", type=int, default=600)
    corpus.add_argument("--force", action="store_true")

    preflight = sub.add_parser("preflight", help="verify logits and one full-stack VJP")
    _add_runtime_args(preflight)
    preflight.add_argument("--source-layer", type=int, default=0)
    preflight.add_argument("--target-layer", type=int, default=-1)
    preflight.add_argument("--max-seq-len", type=int, default=32)
    preflight.add_argument(
        "--prompt",
        default=(
            "The quick brown fox jumps over the lazy dog while a careful scientist "
            "checks whether every internal computation remains differentiable."
        ),
    )

    fit = sub.add_parser("fit", help="fit one resumable prompt shard")
    fit.add_argument("corpus")
    fit.add_argument("output")
    fit.add_argument("--shard-index", type=int, default=0)
    fit.add_argument("--num-shards", type=int, default=1)
    fit.add_argument("--source-layers", default="all")
    fit.add_argument("--target-layer", type=int, default=-1)
    fit.add_argument("--dim-batch", type=int, default=DEFAULT_DIM_BATCH)
    fit.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    fit.add_argument("--skip-first", type=int, default=DEFAULT_SKIP_FIRST)
    fit.add_argument("--checkpoint-every", type=int, default=10)
    fit.add_argument("--keep-checkpoint", action="store_true")
    _add_runtime_args(fit)

    merge = sub.add_parser("merge", help="merge disjoint fitted prompt shards")
    merge.add_argument("output")
    merge.add_argument("inputs", nargs="+")

    validate = sub.add_parser("validate", help="validate a saved lens and checksum")
    validate.add_argument("lens")

    render = sub.add_parser("render", help="render the official interactive slice page")
    render.add_argument("lens")
    render.add_argument("output_dir")
    render.add_argument("--example", default="modulation-topic")
    render.add_argument("--prompt", default=None)
    render.add_argument("--prompt-file", default=None)
    render.add_argument("--top-n", type=int, default=8)
    render.add_argument("--layer-stride", type=int, default=1)
    render.add_argument("--last-n-tokens", type=int, default=None)
    render.add_argument("--max-seq-len", type=int, default=512)
    render.add_argument("--max-tracked", type=int, default=512)
    _add_runtime_args(render)

    fixture = sub.add_parser(
        "export-fixture",
        help="export genuine Jacobian/logit measurements for the local demo",
    )
    fixture.add_argument("lens")
    fixture.add_argument("output")
    fixture.add_argument("--example", default="modulation-topic")
    fixture.add_argument("--top-n", type=int, default=8)
    fixture.add_argument("--layer-stride", type=int, default=1)
    fixture.add_argument("--last-n-tokens", type=int, default=None)
    fixture.add_argument("--max-seq-len", type=int, default=512)
    fixture.add_argument(
        "--tracked-token",
        action="append",
        dest="tracked_tokens",
        default=None,
        help=(
            "single vocabulary token to keep exact ranks for (repeatable; "
            "defaults to the four ocean-creature tokens)"
        ),
    )
    _add_runtime_args(fixture)

    compare = sub.add_parser(
        "compare-fixtures",
        help="compare recorded focus/control fixtures using exact tracked rows",
    )
    compare.add_argument("focus")
    compare.add_argument(
        "--control",
        action="append",
        type=_control_spec,
        required=True,
        metavar="LABEL=PATH",
        help="labeled recorded control fixture (repeatable)",
    )
    compare.add_argument("--output", default=None)

    serve = sub.add_parser(
        "serve-demo", help="serve the fixture or a rendered static demo"
    )
    serve.add_argument("--directory", default="demo")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def _run(args: argparse.Namespace) -> Any:
    if args.command == "info":
        return {
            "model": NEMOTRON.to_dict(),
            "upstream_jlens_commit": UPSTREAM_JLENS_COMMIT,
            "adaptation_source_sha256": adaptation_source_sha256(),
            "defaults": {
                "n_prompts": DEFAULT_N_PROMPTS,
                "max_seq_len": DEFAULT_MAX_SEQ_LEN,
                "skip_first": DEFAULT_SKIP_FIRST,
                "dim_batch": args.dim_batch,
            },
            "resources": {
                "fp16_lens_gib": NEMOTRON.artifact_size_gib(),
                "fp32_lens_gib": NEMOTRON.artifact_size_gib(bytes_per_element=4),
                "fit_host_ram_gib": NEMOTRON.fit_host_ram_gib(),
                "fit_finalize_host_ram_gib": NEMOTRON.fit_finalize_host_ram_gib(),
                "streaming_merge_host_ram_gib": (
                    NEMOTRON.streaming_merge_host_ram_gib()
                ),
                "bf16_weight_gib": NEMOTRON.weight_size_gib(),
                "backward_passes_per_prompt": NEMOTRON.backward_passes_per_prompt(
                    args.dim_batch
                ),
            },
        }
    if args.command == "prepare-corpus":
        dataset_config = (
            None
            if args.dataset_config.lower() in ("none", "null", "")
            else args.dataset_config
        )
        return prepare_corpus(
            args.output,
            n_prompts=args.n_prompts,
            dataset_id=args.dataset,
            dataset_config=dataset_config,
            dataset_revision=args.dataset_revision,
            split=args.split,
            text_field=args.text_field,
            max_chars=args.max_chars,
            min_chars=args.min_chars,
            force=args.force,
        )
    if args.command == "preflight":
        loaded = load_nemotron(
            dtype=args.dtype,
            device_map=args.device_map,
            compile_blocks=args.compile_blocks,
            disable_mamba_kernels=args.disable_mamba_kernels,
            cache_dir=args.cache_dir,
        )
        return {
            "model_id": loaded.model_id,
            "revision": loaded.revision,
            "mamba_backend": loaded.mamba_backend,
            "patched_mamba_layers": loaded.patched_mamba_layers,
            "adaptation_source_sha256": adaptation_source_sha256(),
            "logit_parity": check_native_logit_parity(
                loaded.hf_model,
                loaded.lens_model,
                args.prompt,
                max_seq_len=args.max_seq_len,
            ),
            "vjp": run_vjp_preflight(
                loaded.lens_model,
                args.prompt,
                source_layer=args.source_layer,
                target_layer=args.target_layer,
                max_seq_len=args.max_seq_len,
            ),
        }
    if args.command == "fit":
        return fit_shard(
            corpus_path=args.corpus,
            output_path=args.output,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            source_layer_spec=args.source_layers,
            target_layer=args.target_layer,
            dim_batch=args.dim_batch,
            max_seq_len=args.max_seq_len,
            skip_first=args.skip_first,
            checkpoint_every=args.checkpoint_every,
            keep_checkpoint=args.keep_checkpoint,
            dtype=args.dtype,
            device_map=args.device_map,
            compile_blocks=args.compile_blocks,
            disable_mamba_kernels=args.disable_mamba_kernels,
            cache_dir=args.cache_dir,
        )
    if args.command == "merge":
        return merge_shards(args.inputs, args.output)
    if args.command == "validate":
        lens = jlens.JacobianLens.load(args.lens)
        result = validate_lens(lens)
        if not metadata_path(args.lens).exists():
            raise ValueError("Nemotron artifact validation requires a metadata sidecar")
        artifact_metadata = read_metadata(args.lens)
        validate_artifact(lens, artifact_metadata)
        if (
            artifact_metadata["model_id"] != NEMOTRON.model_id
            or artifact_metadata["model_revision"] != NEMOTRON.revision
        ):
            raise ValueError(
                "artifact metadata does not identify the pinned Nemotron checkpoint"
            )
        result["metadata"] = artifact_metadata
        return result
    if args.command == "render":
        if args.prompt is not None and args.prompt_file is not None:
            raise ValueError("pass at most one of --prompt and --prompt-file")
        prompt = args.prompt
        if args.prompt_file is not None:
            prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        return render_static_demo(
            lens_path=args.lens,
            output_dir=args.output_dir,
            example_slug=args.example,
            prompt=prompt,
            top_n=args.top_n,
            layer_stride=args.layer_stride,
            last_n_tokens=args.last_n_tokens,
            max_seq_len=args.max_seq_len,
            max_tracked=args.max_tracked,
            dtype=args.dtype,
            device_map=args.device_map,
            compile_blocks=args.compile_blocks,
            disable_mamba_kernels=args.disable_mamba_kernels,
            cache_dir=args.cache_dir,
        )
    if args.command == "export-fixture":
        return export_fixture(
            lens_path=args.lens,
            output_path=args.output,
            example_slug=args.example,
            top_n=args.top_n,
            layer_stride=args.layer_stride,
            last_n_tokens=args.last_n_tokens,
            max_seq_len=args.max_seq_len,
            tracked_token_values=(
                DEFAULT_TRACKED_TOKENS
                if args.tracked_tokens is None
                else args.tracked_tokens
            ),
            dtype=args.dtype,
            device_map=args.device_map,
            compile_blocks=args.compile_blocks,
            disable_mamba_kernels=args.disable_mamba_kernels,
            cache_dir=args.cache_dir,
        )
    if args.command == "compare-fixtures":
        controls = dict(args.control)
        if len(controls) != len(args.control):
            raise ValueError("control labels must be unique")
        return compare_fixtures(
            args.focus,
            controls,
            output_path=args.output,
        )
    if args.command == "serve-demo":
        directory = Path(args.directory).resolve()
        handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
        server = ThreadingHTTPServer((args.host, args.port), handler)
        print(f"Serving {directory} at http://{args.host}:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return None
    raise AssertionError(f"unhandled command {args.command}")


def main() -> None:
    jlens.configure_logging()
    result = _run(build_parser().parse_args())
    if result is not None:
        _print(result)


if __name__ == "__main__":
    main()
