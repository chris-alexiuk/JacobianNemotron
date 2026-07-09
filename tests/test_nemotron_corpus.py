import json
from dataclasses import asdict

import datasets
import pytest

from nemotron_jlens.corpus import (
    CorpusRecord,
    _chunks,
    load_corpus,
    prepare_corpus,
    select_shard,
    sha256_file,
    sha256_text,
)


def _write_corpus(tmp_path, texts):
    path = tmp_path / "corpus.jsonl"
    records = [
        CorpusRecord(index=index, text=text, sha256=sha256_text(text))
        for index, text in enumerate(texts)
    ]
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    metadata = {
        "n_prompts": len(records),
        "manifest_sha256": sha256_file(path),
        "prompt_hashes": [record.sha256 for record in records],
    }
    meta_path = tmp_path / "corpus.jsonl.meta.json"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    return path, meta_path, records


def test_chunks_filters_headers_and_concatenates_rows_deterministically():
    rows = [
        {"text": " = article header "},
        {"text": ""},
        {"text": " abc "},
        {"text": "defgh"},
        {"text": "ij"},
    ]
    assert list(_chunks(rows, text_field="text", max_chars=6, min_chars=2)) == [
        "abc\nde",
        "fgh\nij",
    ]


def test_prepare_corpus_uses_disk_backed_dataset_and_is_deterministic(
    tmp_path, monkeypatch
):
    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return [
            {"text": "abcdef"},
            {"text": "ghijkl"},
        ]

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    first_metadata = prepare_corpus(
        first,
        n_prompts=2,
        max_chars=6,
        min_chars=2,
    )
    second_metadata = prepare_corpus(
        second,
        n_prompts=2,
        max_chars=6,
        min_chars=2,
    )

    assert len(calls) == 2
    assert all(kwargs["streaming"] is False for _, kwargs in calls)
    assert all(kwargs["keep_in_memory"] is False for _, kwargs in calls)
    assert first.read_bytes() == second.read_bytes()
    assert first_metadata["manifest_sha256"] == second_metadata["manifest_sha256"]
    assert first_metadata["prompt_hashes"] == second_metadata["prompt_hashes"]
    records, metadata = load_corpus(first)
    assert [record.text for record in records] == ["abcdef", "ghijkl"]
    assert metadata == first_metadata


def test_load_corpus_verifies_manifest_prompts_and_round_robin_shards(tmp_path):
    path, _, expected = _write_corpus(tmp_path, [f"prompt {i}" for i in range(7)])
    records, metadata = load_corpus(path)
    assert records == expected
    assert metadata["n_prompts"] == 7
    assert [
        record.index for record in select_shard(records, shard_index=0, num_shards=3)
    ] == [
        0,
        3,
        6,
    ]
    assert [
        record.index for record in select_shard(records, shard_index=1, num_shards=3)
    ] == [
        1,
        4,
    ]
    assert [
        record.index for record in select_shard(records, shard_index=2, num_shards=3)
    ] == [
        2,
        5,
    ]


def test_load_corpus_rejects_manifest_tampering(tmp_path):
    path, _, _ = _write_corpus(tmp_path, ["alpha", "beta"])
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest checksum mismatch"):
        load_corpus(path)


def test_load_corpus_rejects_per_prompt_tampering_even_with_new_manifest_hash(
    tmp_path,
):
    path, meta_path, _ = _write_corpus(tmp_path, ["alpha", "beta"])
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["text"] = "altered"
    lines[0] = json.dumps(first, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["manifest_sha256"] = sha256_file(path)
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="prompt checksum mismatch at line 1"):
        load_corpus(path)


@pytest.mark.parametrize(
    ("shard_index", "num_shards"),
    [(0, 0), (-1, 2), (2, 2)],
)
def test_select_shard_rejects_invalid_coordinates(shard_index, num_shards):
    with pytest.raises(ValueError, match="num_shards > 0"):
        select_shard([], shard_index=shard_index, num_shards=num_shards)
