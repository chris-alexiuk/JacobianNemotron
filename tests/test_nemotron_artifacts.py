import copy
import hashlib
from itertools import permutations

import pytest
import torch

from jlens import JacobianLens
from nemotron_jlens.artifacts import (
    classify_reproduction,
    load_validated_artifact,
    merge_shards,
    read_metadata,
    save_validated_artifact,
    serialized_storage_dtype,
    validate_artifact,
    validate_lens,
    write_metadata,
)
from nemotron_jlens.config import (
    ACCEPTANCE_SCHEMA,
    ARTIFACT_STORAGE_DTYPE,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_REVISION,
    DEFAULT_DATASET_SPLIT,
    NEMOTRON,
    PAPER_SCALE_CORPUS_MANIFEST_SHA256,
    PAPER_SCALE_N_PROMPTS,
    PILOT_CORPUS_MANIFEST_SHA256,
    PILOT_N_PROMPTS,
    REPRODUCTION_FIT_SETTINGS,
    REQUIRED_TRANSFORMERS_VERSION,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


ADAPTATION_SOURCE_SHA = _sha256("adaptation-source")
CORPUS_MANIFEST_SHA = _sha256("corpus-manifest")
TEST_RUNTIME = {
    "python": "3.12.0",
    "torch": "2.9.0+cu130",
    "cuda_runtime": "13.0",
    "packages": {
        "transformers": REQUIRED_TRANSFORMERS_VERSION,
        "mamba-ssm": "2.2.5",
        "causal-conv1d": "1.5.3",
        "accelerate": "1.12.0",
        "datasets": "3.1.0",
        "huggingface-hub": "0.36.0",
    },
}


def _metadata(
    prompt_hashes,
    *,
    prompt_indices=None,
    shard_index=0,
    num_shards=1,
    target_layer=2,
    environment=None,
    overrides=None,
):
    if prompt_indices is None:
        prompt_indices = list(range(len(prompt_hashes)))
    runtime = copy.deepcopy(
        (overrides or {}).get("runtime", TEST_RUNTIME)
    )
    if environment is None:
        environment = {**copy.deepcopy(runtime), "gpu": "test"}
    metadata = {
        "schema_version": 1,
        "kind": "shard",
        "upstream_jlens_commit": "upstream-commit",
        "adaptation_source_sha256": ADAPTATION_SOURCE_SHA,
        "model_id": "test/model",
        "model_revision": "revision",
        "architecture": {
            "model_id": "test/model",
            "revision": "revision",
            "model_type": "tiny",
            "n_layers": 3,
            "d_model": 3,
        },
        "dataset": {
            "id": "test/data",
            "config": "test-config",
            "revision": "data-revision",
            "split": "train",
        },
        "corpus_manifest_sha256": CORPUS_MANIFEST_SHA,
        "corpus_path": "/data/corpus.jsonl",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "n_prompts": len(prompt_hashes),
        "prompt_indices": prompt_indices,
        "source_layers": [0, 1],
        "target_layer": target_layer,
        "dim_batch": 4,
        "max_seq_len": 128,
        "skip_first": 16,
        "dtype": "bfloat16",
        "storage_dtype": ARTIFACT_STORAGE_DTYPE,
        "compile_blocks": False,
        "disable_mamba_kernels": True,
        "mamba_backend": "torch",
        "patched_mamba_layers": 1,
        "device_map": "cuda",
        "runtime": runtime,
        "environment": environment,
        "prompt_hashes": prompt_hashes,
    }
    if overrides:
        metadata.update(overrides)
    metadata["acceptance"] = classify_reproduction(metadata)
    return metadata


def _write_shard(
    path,
    *,
    values,
    n_prompts,
    prompt_hashes,
    prompt_indices=None,
    shard_index=0,
    num_shards=1,
    target_layer=2,
    environment=None,
    overrides=None,
):
    lens = JacobianLens(
        jacobians={
            0: torch.eye(3) * values[0],
            1: torch.eye(3) * values[1],
        },
        n_prompts=n_prompts,
        d_model=3,
    )
    lens.save(str(path), dtype=torch.float32)
    write_metadata(
        path,
        _metadata(
            prompt_hashes,
            prompt_indices=prompt_indices,
            shard_index=shard_index,
            num_shards=num_shards,
            target_layer=target_layer,
            environment=environment,
            overrides=overrides,
        ),
    )
    return lens


def test_validate_lens_checks_shape_finiteness_norm_and_expectations():
    lens = JacobianLens(
        jacobians={0: torch.eye(3), 1: 2 * torch.eye(3)},
        n_prompts=4,
        d_model=3,
    )
    summary = validate_lens(lens, expected_d_model=3, expected_layers=[0, 1])
    assert summary["ok"] is True
    assert summary["n_prompts"] == 4
    assert summary["layers"]["0"]["identity_distance_over_sqrt_d"] == 0.0

    with pytest.raises(ValueError, match="d_model"):
        validate_lens(lens, expected_d_model=4)
    with pytest.raises(ValueError, match="source_layers"):
        validate_lens(lens, expected_layers=[0])
    with pytest.raises(ValueError, match="invalid shape"):
        validate_lens(
            JacobianLens(jacobians={0: torch.ones(2, 3)}, n_prompts=1, d_model=3)
        )
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_lens(
            JacobianLens(
                jacobians={0: torch.tensor([[float("nan")]])},
                n_prompts=1,
                d_model=1,
            )
        )
    with pytest.raises(ValueError, match="invalid norm"):
        validate_lens(
            JacobianLens(jacobians={0: torch.zeros(1, 1)}, n_prompts=1, d_model=1)
        )


def test_validate_artifact_cross_checks_payload_bounds_and_provenance():
    lens = JacobianLens(
        jacobians={0: torch.eye(3), 1: 2 * torch.eye(3)},
        n_prompts=2,
        d_model=3,
    )
    metadata = _metadata([_sha256("hash-0"), _sha256("hash-1")])
    summary = validate_artifact(lens, metadata, expected_kind="shard")
    assert summary["n_prompts"] == 2

    bad = copy.deepcopy(metadata)
    bad["n_prompts"] = 1
    with pytest.raises(ValueError, match="n_prompts"):
        validate_artifact(lens, bad)

    bad = copy.deepcopy(metadata)
    bad["source_layers"] = [0]
    with pytest.raises(ValueError, match="source_layers"):
        validate_artifact(lens, bad)

    bad = copy.deepcopy(metadata)
    bad["prompt_hashes"] = [_sha256("hash-0")]
    with pytest.raises(ValueError, match="wrong prompt count"):
        validate_artifact(lens, bad)

    bad = copy.deepcopy(metadata)
    bad["target_layer"] = 3
    with pytest.raises(ValueError, match="outside architecture"):
        validate_artifact(lens, bad)

    bad = copy.deepcopy(metadata)
    del bad["model_revision"]
    with pytest.raises(ValueError, match="missing provenance"):
        validate_artifact(lens, bad)

    bad = copy.deepcopy(metadata)
    bad["dataset"]["revision"] = ""
    with pytest.raises(ValueError, match="dataset is missing"):
        validate_artifact(lens, bad)

    bad = copy.deepcopy(metadata)
    bad["architecture"]["d_model"] = 4
    with pytest.raises(ValueError, match="d_model"):
        validate_artifact(lens, bad)

    bad = copy.deepcopy(metadata)
    bad["runtime"]["packages"]["transformers"] = "4.57.2"
    with pytest.raises(ValueError, match="exact version 4.57.3"):
        validate_artifact(lens, bad)

    bad = copy.deepcopy(metadata)
    del bad["runtime"]["packages"]["mamba-ssm"]
    with pytest.raises(ValueError, match="mamba-ssm"):
        validate_artifact(lens, bad)

    bad = copy.deepcopy(metadata)
    bad["environment"]["packages"]["accelerate"] = "different"
    with pytest.raises(ValueError, match="environment disagrees"):
        validate_artifact(lens, bad)


def test_metadata_round_trip_verifies_artifact_checksum(tmp_path):
    path = tmp_path / "lens.pt"
    _write_shard(
        path,
        values=(1.0, 2.0),
        n_prompts=1,
        prompt_hashes=[_sha256("hash-a")],
    )
    metadata = read_metadata(path)
    assert metadata["artifact_sha256"]
    assert metadata["created_at"]

    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="artifact checksum mismatch"):
        read_metadata(path)


def test_save_validated_artifact_preserves_fp32_and_publishes_sidecar_last(tmp_path):
    path = tmp_path / "lens.pt"
    lens = JacobianLens(
        jacobians={
            0: torch.eye(3) * 1.0001,
            1: torch.eye(3) * 2.0003,
        },
        n_prompts=1,
        d_model=3,
    )
    metadata = _metadata([_sha256("hash-a")])

    saved = save_validated_artifact(
        lens, path, metadata, expected_kind="shard"
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert {matrix.dtype for matrix in payload["J"].values()} == {torch.float32}
    torch.testing.assert_close(payload["J"][0], lens.jacobians[0], atol=0, rtol=0)
    assert saved["storage_dtype"] == "float32"
    assert saved["validation"]["dtype"] == "float32"
    assert read_metadata(path)["artifact_sha256"] == saved["artifact_sha256"]

    artifact_before = path.read_bytes()
    sidecar_before = path.with_suffix(".pt.meta.json").read_bytes()
    invalid = copy.deepcopy(metadata)
    invalid["n_prompts"] = 2
    with pytest.raises(ValueError, match="n_prompts"):
        save_validated_artifact(lens, path, invalid, expected_kind="shard")
    assert path.read_bytes() == artifact_before
    assert path.with_suffix(".pt.meta.json").read_bytes() == sidecar_before
    assert not list(tmp_path.glob(".*.tmp"))


def test_load_validated_artifact_rejects_quantized_serialized_means(tmp_path):
    path = tmp_path / "quantized.pt"
    lens = JacobianLens(
        jacobians={0: torch.eye(3), 1: 2 * torch.eye(3)},
        n_prompts=1,
        d_model=3,
    )
    lens.save(str(path), dtype=torch.float16)
    metadata = _metadata([_sha256("hash-a")])
    write_metadata(path, metadata)
    with pytest.raises(ValueError, match="unsupported serialized storage dtype"):
        read_metadata(path)
    with pytest.raises(ValueError, match="unsupported serialized storage dtype"):
        load_validated_artifact(path, metadata, expected_kind="shard")


def _canonical_profile_metadata(manifest: str, n_prompts: int) -> dict:
    metadata = {
        "model_id": NEMOTRON.model_id,
        "model_revision": NEMOTRON.revision,
        "architecture": NEMOTRON.to_dict(),
        "dataset": {
            "id": DEFAULT_DATASET_ID,
            "config": DEFAULT_DATASET_CONFIG,
            "revision": DEFAULT_DATASET_REVISION,
            "split": DEFAULT_DATASET_SPLIT,
        },
        "corpus_manifest_sha256": manifest,
        "n_prompts": n_prompts,
        "prompt_indices": list(range(n_prompts)),
        "source_layers": list(range(NEMOTRON.n_layers - 1)),
        "target_layer": NEMOTRON.n_layers - 1,
        "storage_dtype": ARTIFACT_STORAGE_DTYPE,
        "runtime": copy.deepcopy(TEST_RUNTIME),
        **REPRODUCTION_FIT_SETTINGS,
    }
    return metadata


def test_reproduction_classification_is_exact_and_reusable():
    paper = classify_reproduction(
        _canonical_profile_metadata(
            PAPER_SCALE_CORPUS_MANIFEST_SHA256, PAPER_SCALE_N_PROMPTS
        )
    )
    assert paper == {
        "schema": ACCEPTANCE_SCHEMA,
        "tier": "paper-scale",
        "status": "accepted",
        "is_final": True,
        "exportable": True,
        "checks": {
            "pinned_model": True,
            "pinned_dataset": True,
            "known_manifest": True,
            "complete_prompt_set": True,
            "full_layer_coverage": True,
            "fixed_fit_settings": True,
            "pinned_transformers": True,
            "fp32_storage": True,
        },
        "reasons": [],
    }

    pilot = classify_reproduction(
        _canonical_profile_metadata(PILOT_CORPUS_MANIFEST_SHA256, PILOT_N_PROMPTS)
    )
    assert pilot["tier"] == "pilot"
    assert pilot["status"] == "accepted"
    assert pilot["is_final"] is False
    assert pilot["reasons"]

    smoke_metadata = _canonical_profile_metadata(_sha256("smoke-corpus"), 8)
    smoke = classify_reproduction(smoke_metadata)
    assert smoke["tier"] == "smoke"
    assert smoke["status"] == "non-final"
    assert smoke["is_final"] is False
    assert smoke["exportable"] is True
    assert smoke["checks"]["known_manifest"] is False


@pytest.mark.parametrize(
    ("field", "value", "failed_check"),
    [
        ("target_layer", 50, "full_layer_coverage"),
        ("dim_batch", 16, "fixed_fit_settings"),
        ("storage_dtype", "float16", "fp32_storage"),
        ("prompt_indices", list(range(999)), "complete_prompt_set"),
        (
            "runtime",
            {
                **TEST_RUNTIME,
                "packages": {
                    **TEST_RUNTIME["packages"],
                    "transformers": "4.57.2",
                },
            },
            "pinned_transformers",
        ),
    ],
)
def test_paper_scale_acceptance_fails_closed_on_every_contract_axis(
    field, value, failed_check
):
    metadata = _canonical_profile_metadata(
        PAPER_SCALE_CORPUS_MANIFEST_SHA256, PAPER_SCALE_N_PROMPTS
    )
    metadata[field] = value
    acceptance = classify_reproduction(metadata)
    assert acceptance["tier"] == "smoke"
    assert acceptance["status"] == "non-final"
    assert acceptance["checks"][failed_check] is False


def test_validate_artifact_enforces_fp32_and_computed_acceptance():
    lens = JacobianLens(
        jacobians={0: torch.eye(3), 1: 2 * torch.eye(3)},
        n_prompts=1,
        d_model=3,
    )
    metadata = _metadata([_sha256("hash-a")])

    forged = copy.deepcopy(metadata)
    forged["acceptance"]["tier"] = "paper-scale"
    with pytest.raises(ValueError, match="acceptance classification"):
        validate_artifact(lens, forged)

    lens.jacobians[0] = lens.jacobians[0].half()
    with pytest.raises(ValueError, match="dtype"):
        validate_artifact(lens, metadata)


def test_merge_shards_is_prompt_weighted_and_preserves_provenance(tmp_path):
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    output = tmp_path / "merged.pt"
    _write_shard(
        first,
        values=(1.0, 2.0),
        n_prompts=3,
        prompt_hashes=[_sha256(f"hash-{index}") for index in (0, 2, 4)],
        prompt_indices=[0, 2, 4],
        shard_index=0,
        num_shards=2,
        environment={**copy.deepcopy(TEST_RUNTIME), "gpu": "GPU-A"},
    )
    _write_shard(
        second,
        values=(4.0, 8.0),
        n_prompts=2,
        prompt_hashes=[_sha256(f"hash-{index}") for index in (1, 3)],
        prompt_indices=[1, 3],
        shard_index=1,
        num_shards=2,
        environment={**copy.deepcopy(TEST_RUNTIME), "gpu": "GPU-B"},
    )

    metadata = merge_shards([second, first], output)
    merged = JacobianLens.load(str(output))
    assert merged.n_prompts == 5
    assert serialized_storage_dtype(output) == "float32"
    assert all(matrix.dtype == torch.float32 for matrix in merged.jacobians.values())
    torch.testing.assert_close(
        merged.jacobians[0], torch.eye(3) * 2.2, atol=0, rtol=1e-6
    )
    torch.testing.assert_close(
        merged.jacobians[1], torch.eye(3) * 4.4, atol=0, rtol=1e-6
    )
    assert metadata["kind"] == "merged"
    assert metadata["n_prompts"] == 5
    assert metadata["prompt_indices"] == list(range(5))
    assert metadata["prompt_hashes"] == [
        _sha256(f"hash-{index}") for index in range(5)
    ]
    assert metadata["upstream_jlens_commit"] == "upstream-commit"
    assert metadata["adaptation_source_sha256"] == ADAPTATION_SOURCE_SHA
    assert metadata["architecture"]["d_model"] == 3
    assert metadata["dataset"]["revision"] == "data-revision"
    assert metadata["corpus_manifest_sha256"] == CORPUS_MANIFEST_SHA
    assert metadata["corpus_path"] == "/data/corpus.jsonl"
    assert metadata["dim_batch"] == 4
    assert metadata["dtype"] == "bfloat16"
    assert metadata["storage_dtype"] == "float32"
    assert metadata["compile_blocks"] is False
    assert metadata["disable_mamba_kernels"] is True
    assert metadata["mamba_backend"] == "torch"
    assert metadata["patched_mamba_layers"] == 1
    assert metadata["runtime"] == TEST_RUNTIME
    assert [item["environment"]["gpu"] for item in metadata["input_shards"]] == [
        "GPU-A",
        "GPU-B",
    ]
    assert all(
        item["runtime"] == TEST_RUNTIME for item in metadata["input_shards"]
    )
    saved_metadata = read_metadata(output)
    assert saved_metadata["artifact_sha256"]
    assert saved_metadata["validation"]["ok"] is True
    assert saved_metadata["validation"]["dtype"] == "float32"
    assert saved_metadata["acceptance"]["tier"] == "smoke"


def test_merge_shards_is_byte_identical_for_every_input_permutation(tmp_path):
    shards = []
    values = ((1e10, 1e8), (-1e10, -1e8), (3.0, 6.0))
    for shard_index, shard_values in enumerate(values):
        path = tmp_path / f"shard-{shard_index}.pt"
        _write_shard(
            path,
            values=shard_values,
            n_prompts=1,
            prompt_hashes=[_sha256(f"hash-{shard_index}")],
            prompt_indices=[shard_index],
            shard_index=shard_index,
            num_shards=len(values),
        )
        shards.append(path)

    expected_bytes = None
    expected_checksum = None
    expected_input_shards = None
    for permutation_index, inputs in enumerate(permutations(shards)):
        output = tmp_path / f"merged-{permutation_index}.pt"
        metadata = merge_shards(inputs, output)
        artifact_bytes = output.read_bytes()
        checksum = hashlib.sha256(artifact_bytes).hexdigest()

        assert metadata["artifact_sha256"] == checksum
        assert [
            item["shard_index"] for item in metadata["input_shards"]
        ] == list(range(len(shards)))
        if expected_bytes is None:
            expected_bytes = artifact_bytes
            expected_checksum = checksum
            expected_input_shards = metadata["input_shards"]
        else:
            assert artifact_bytes == expected_bytes
            assert checksum == expected_checksum
            assert metadata["input_shards"] == expected_input_shards

    merged = JacobianLens.load(str(output))
    torch.testing.assert_close(
        merged.jacobians[0], torch.eye(3), atol=0, rtol=0
    )
    torch.testing.assert_close(
        merged.jacobians[1], torch.eye(3) * 2, atol=0, rtol=0
    )


def test_merge_shards_rejects_overlap(tmp_path):
    first = tmp_path / "first.pt"
    overlap = tmp_path / "overlap.pt"
    _write_shard(
        first,
        values=(1.0, 2.0),
        n_prompts=1,
        prompt_hashes=[_sha256("same")],
        prompt_indices=[0],
        shard_index=0,
        num_shards=2,
    )
    _write_shard(
        overlap,
        values=(3.0, 4.0),
        n_prompts=1,
        prompt_hashes=[_sha256("same")],
        prompt_indices=[1],
        shard_index=1,
        num_shards=2,
    )
    with pytest.raises(ValueError, match="prompt shards overlap"):
        merge_shards([first, overlap], tmp_path / "bad-overlap.pt")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("upstream_jlens_commit", "different-commit"),
        ("adaptation_source_sha256", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
        ("architecture", {"n_layers": 4, "d_model": 3}),
        ("dim_batch", 8),
        ("dtype", "float16"),
        ("compile_blocks", True),
        ("disable_mamba_kernels", False),
        ("mamba_backend", "fused-or-auto"),
        ("patched_mamba_layers", 0),
        (
            "runtime",
            {
                **TEST_RUNTIME,
                "packages": {
                    **TEST_RUNTIME["packages"],
                    "accelerate": "1.13.0",
                },
            },
        ),
    ],
)
def test_merge_shards_rejects_metadata_drift(tmp_path, field, value):
    first = tmp_path / "first.pt"
    drift = tmp_path / "drift.pt"
    _write_shard(
        first,
        values=(1.0, 2.0),
        n_prompts=1,
        prompt_hashes=[_sha256("hash-0")],
        prompt_indices=[0],
        shard_index=0,
        num_shards=2,
    )
    _write_shard(
        drift,
        values=(3.0, 4.0),
        n_prompts=1,
        prompt_hashes=[_sha256("hash-1")],
        prompt_indices=[1],
        shard_index=1,
        num_shards=2,
        overrides={field: value},
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        merge_shards([first, drift], tmp_path / "bad-drift.pt")


def test_merge_shards_requires_complete_unique_direct_partition(tmp_path):
    first = tmp_path / "first.pt"
    duplicate = tmp_path / "duplicate.pt"
    _write_shard(
        first,
        values=(1.0, 2.0),
        n_prompts=1,
        prompt_hashes=[_sha256("hash-0")],
        prompt_indices=[0],
        shard_index=0,
        num_shards=2,
    )
    with pytest.raises(ValueError, match="incomplete shard partition"):
        merge_shards([first], tmp_path / "incomplete.pt")

    _write_shard(
        duplicate,
        values=(3.0, 4.0),
        n_prompts=1,
        prompt_hashes=[_sha256("hash-2")],
        prompt_indices=[2],
        shard_index=0,
        num_shards=2,
    )
    with pytest.raises(ValueError, match="duplicate shard indices"):
        merge_shards([first, duplicate], tmp_path / "duplicate.pt")

    direct_metadata = read_metadata(first)
    direct_metadata["kind"] = "merged"
    write_metadata(first, direct_metadata)
    with pytest.raises(ValueError, match="direct shard artifacts"):
        merge_shards([first], tmp_path / "nested.pt")


def test_merge_shards_rejects_metadata_prompt_count_mismatch(tmp_path):
    path = tmp_path / "bad-count.pt"
    _write_shard(
        path,
        values=(1.0, 2.0),
        n_prompts=2,
        prompt_hashes=[_sha256("only-one")],
        overrides={"n_prompts": 2},
    )
    with pytest.raises(ValueError, match="wrong prompt count"):
        merge_shards([path], tmp_path / "merged.pt")
