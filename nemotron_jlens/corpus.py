"""Deterministic prompt manifests for exact, shardable Jacobian fitting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from nemotron_jlens.config import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_REVISION,
    DEFAULT_DATASET_SPLIT,
    DEFAULT_TEXT_FIELD,
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CorpusRecord:
    index: int
    text: str
    sha256: str


def _chunks(
    records: Iterable[dict],
    *,
    text_field: str,
    max_chars: int,
    min_chars: int,
) -> Iterable[str]:
    """Concatenate short dataset rows into stable, pretraining-like chunks."""
    buffer = ""
    for row in records:
        text = str(row.get(text_field, "")).strip()
        if not text or text.startswith("="):
            continue
        buffer = f"{buffer}\n{text}" if buffer else text
        while len(buffer) >= max_chars:
            chunk, buffer = buffer[:max_chars], buffer[max_chars:]
            if len(chunk.strip()) >= min_chars:
                yield chunk.strip()
    if len(buffer.strip()) >= min_chars:
        yield buffer.strip()


def prepare_corpus(
    output: str | Path,
    *,
    n_prompts: int,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_config: str | None = DEFAULT_DATASET_CONFIG,
    dataset_revision: str = DEFAULT_DATASET_REVISION,
    split: str = DEFAULT_DATASET_SPLIT,
    text_field: str = DEFAULT_TEXT_FIELD,
    max_chars: int = 2000,
    min_chars: int = 600,
    force: bool = False,
) -> dict:
    """Build a canonical manifest from a pinned, disk-backed HF dataset.

    Hugging Face streaming uses a background iterable/download path that can abort
    during interpreter teardown in some container runtimes.  Materializing the
    pinned Arrow dataset in the datasets cache is deterministic, memory-mapped by
    default, and keeps corpus construction independent of that streaming runtime.
    """
    if n_prompts <= 0:
        raise ValueError("n_prompts must be positive")
    if min_chars <= 0 or max_chars < min_chars:
        raise ValueError("need 0 < min_chars <= max_chars")

    output = Path(output)
    meta_path = Path(f"{output}.meta.json")
    if (output.exists() or meta_path.exists()) and not force:
        raise FileExistsError(f"{output} already exists; pass force=True to replace it")

    from datasets import load_dataset

    dataset = load_dataset(
        dataset_id,
        dataset_config,
        split=split,
        revision=dataset_revision,
        streaming=False,
        keep_in_memory=False,
    )
    records: list[CorpusRecord] = []
    for index, text in enumerate(
        _chunks(
            dataset,
            text_field=text_field,
            max_chars=max_chars,
            min_chars=min_chars,
        )
    ):
        records.append(CorpusRecord(index=index, text=text, sha256=sha256_text(text)))
        if len(records) == n_prompts:
            break
    if len(records) != n_prompts:
        raise RuntimeError(
            f"dataset ended after {len(records)} usable prompts; requested {n_prompts}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(f".{output.name}.tmp")
    with open(temp, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n"
            )
    temp.replace(output)

    metadata = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "id": dataset_id,
            "config": dataset_config,
            "revision": dataset_revision,
            "split": split,
            "text_field": text_field,
        },
        "construction": {
            "algorithm": "ordered-concatenate-and-chunk-v1",
            "max_chars": max_chars,
            "min_chars": min_chars,
            "header_filter": "drop rows whose stripped text starts with '='",
        },
        "n_prompts": len(records),
        "manifest_sha256": sha256_file(output),
        "prompt_hashes": [record.sha256 for record in records],
    }
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metadata


def load_corpus(path: str | Path) -> tuple[list[CorpusRecord], dict]:
    """Load a corpus and verify every prompt plus the manifest sidecar."""
    path = Path(path)
    meta_path = Path(f"{path}.meta.json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    actual_manifest_hash = sha256_file(path)
    if actual_manifest_hash != metadata["manifest_sha256"]:
        raise ValueError(
            f"corpus manifest checksum mismatch: {actual_manifest_hash} != "
            f"{metadata['manifest_sha256']}"
        )

    records: list[CorpusRecord] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            record = CorpusRecord(**raw)
            if record.index != len(records):
                raise ValueError(
                    f"non-contiguous corpus index at line {line_number}: {record.index}"
                )
            if sha256_text(record.text) != record.sha256:
                raise ValueError(f"prompt checksum mismatch at line {line_number}")
            records.append(record)
    if len(records) != metadata["n_prompts"]:
        raise ValueError("corpus prompt count disagrees with metadata")
    if [record.sha256 for record in records] != metadata["prompt_hashes"]:
        raise ValueError("corpus prompt order disagrees with metadata")
    return records, metadata


def select_shard(
    records: list[CorpusRecord], *, shard_index: int, num_shards: int
) -> list[CorpusRecord]:
    """Round-robin prompt sharding; disjoint shards merge exactly by averaging."""
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError(
            f"need num_shards > 0 and 0 <= shard_index < num_shards; got "
            f"{shard_index}/{num_shards}"
        )
    return records[shard_index::num_shards]
