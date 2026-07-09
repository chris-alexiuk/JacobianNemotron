"""Run and save the real Nano acceptance sequence against a live server."""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from nemotron_steering.constants import PILOT_DISCLOSURE


class Client:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(
        self, method: str, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            data = json.dumps(body, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        headers["X-Request-ID"] = f"smoke-{uuid4().hex}"
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                detail = json.load(exc)
            except json.JSONDecodeError:
                detail = {"error": exc.reason}
            raise RuntimeError(
                f"{method} {path} failed ({exc.code}): {detail}"
            ) from exc


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def _close(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if isinstance(left, int) and isinstance(right, int):
            return left == right
        return math.isclose(float(left), float(right), rel_tol=1e-5, abs_tol=1e-6)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _close(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _close(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _operation(
    client: Client,
    common: dict[str, Any],
    *,
    name: str,
    mode: str,
    source_id: int,
    source_text: str,
    target_id: int | None = None,
    target_text: str | None = None,
    strength: float = -0.1,
) -> dict[str, Any]:
    body = {
        **common,
        "intervention": {
            "mode": mode,
            "lens_type": "jacobian",
            "layers": [26],
            "source_token_ids": [source_id],
            "strength": strength,
            "apply_to_generated": False,
        },
        "source_token_texts": [source_text],
    }
    if target_id is not None:
        body["intervention"]["target_token_id"] = target_id
        body["target_token_text"] = target_text
    started = time.monotonic()
    response = client.request("POST", "/api/intervene", body)
    return {
        "name": name,
        "wall_seconds": time.monotonic() - started,
        "response": response,
    }


def run_smoke(client: Client) -> dict[str, Any]:
    started = time.monotonic()
    health = client.request("GET", "/health")
    info = client.request("GET", "/api/info")
    source = client.request("POST", "/api/tokenize", {"text": " whale"})
    target = client.request("POST", "/api/tokenize", {"text": " fish"})
    if not source.get("is_single_token") or not target.get("is_single_token"):
        raise RuntimeError("acceptance source and target must each be one token")
    source_id = int(source["token_ids"][0])
    target_id = int(target["token_ids"][0])
    common = {
        "messages": [
            {
                "role": "user",
                "content": "Write one short sentence about marine life.",
            }
        ],
        "enable_thinking": False,
        "layers": [26],
        "top_k": 4,
        "max_new_tokens": 8,
        "sampling": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "seed": 20260707,
    }

    baseline_started = time.monotonic()
    baseline_first = client.request("POST", "/api/baseline", common)
    baseline_first_wall = time.monotonic() - baseline_started
    operations = [
        _operation(
            client,
            common,
            name="zero-strength",
            mode="steer",
            source_id=source_id,
            source_text=" whale",
            strength=0.0,
        ),
        _operation(
            client,
            common,
            name="positive-steer",
            mode="steer",
            source_id=source_id,
            source_text=" whale",
            strength=0.08,
        ),
        _operation(
            client,
            common,
            name="negative-steer",
            mode="steer",
            source_id=source_id,
            source_text=" whale",
            strength=-0.08,
        ),
        _operation(
            client,
            common,
            name="ablation",
            mode="ablate",
            source_id=source_id,
            source_text=" whale",
        ),
        _operation(
            client,
            common,
            name="swap-whale-to-fish",
            mode="swap",
            source_id=source_id,
            source_text=" whale",
            target_id=target_id,
            target_text=" fish",
        ),
    ]
    baseline_last_started = time.monotonic()
    baseline_last = client.request("POST", "/api/baseline", common)

    zero = operations[0]["response"]
    first_ids = baseline_first["clean"]["completion_token_ids"]
    last_ids = baseline_last["clean"]["completion_token_ids"]
    checks = {
        "ready_and_loaded": health.get("status") == "ready"
        and health.get("model_loaded") is True,
        "pilot_disclosure": all(
            value.get("disclosure") == PILOT_DISCLOSURE
            for value in (health, info, baseline_first, baseline_last)
        ),
        "baseline_generated_8_tokens": len(first_ids) == 8,
        "direct_chat_template": baseline_first["provenance"].get("prompt_format")
        == "chat"
        and baseline_first["provenance"].get("chat_template", {}).get("enable_thinking")
        is False
        and baseline_first["provenance"]
        .get("formatted_prompt", "")
        .endswith("<think></think>"),
        "baseline_is_direct_response": bool(
            baseline_first["clean"].get("completion", "").strip()
        )
        and "<think>" not in baseline_first["clean"].get("completion", ""),
        "zero_strength_token_parity": zero["clean"]["completion_token_ids"]
        == zero["intervened"]["completion_token_ids"],
        "zero_strength_readouts_close": _close(
            zero["clean"]["readouts"], zero["intervened"]["readouts"]
        ),
        "clean_replay_token_parity": first_ids == last_ids,
        "clean_replay_readouts_close": _close(
            baseline_first["clean"]["readouts"],
            baseline_last["clean"]["readouts"],
        ),
        "no_hook_leaks": all(
            item["response"]["diagnostics"]["hooks_before"]
            == item["response"]["diagnostics"]["hooks_after"]
            for item in operations
        ),
        "all_finite": _all_finite(
            {"first": baseline_first, "operations": operations, "last": baseline_last}
        ),
    }
    first_memory = baseline_first["provenance"]["memory"]
    last_memory = baseline_last["provenance"]["memory"]
    if first_memory.get("available") and last_memory.get("available"):
        checks["no_monotonic_memory_growth"] = (
            last_memory["allocated_bytes"]
            <= first_memory["allocated_bytes"] + 64 * 1024 * 1024
        )
    else:
        checks["no_monotonic_memory_growth"] = True

    return {
        "schema": "nemotron-steering-smoke/v1",
        "disclosure": PILOT_DISCLOSURE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server": client.base_url,
        "health": health,
        "immutable_info": info,
        "tokens": {"source": source, "target": target},
        "baseline_first": {
            "wall_seconds": baseline_first_wall,
            "response": baseline_first,
        },
        "operations": operations,
        "baseline_last": {
            "wall_seconds": time.monotonic() - baseline_last_started,
            "response": baseline_last,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "elapsed_seconds": time.monotonic() - started,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", default="artifacts/nano-steering-smoke.json")
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    client = Client(args.url, args.timeout)
    report = run_smoke(client)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), "checks": report["checks"]}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
