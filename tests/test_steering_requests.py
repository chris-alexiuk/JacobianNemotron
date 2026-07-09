"""Request contracts for Nemotron chat formatting."""

from __future__ import annotations

import pytest

from nemotron_steering.errors import ValidationError
from nemotron_steering.requests import parse_request


def _chat_body(messages: list[dict[str, str]]) -> dict[str, object]:
    return {"messages": messages}


def test_chat_defaults_to_non_thinking_generation() -> None:
    request = parse_request(
        _chat_body([{"role": "user", "content": "Answer briefly."}]),
        require_intervention=False,
    )

    assert request.enable_thinking is False


def test_default_readouts_cover_layers_13_through_50() -> None:
    request = parse_request(
        _chat_body([{"role": "user", "content": "Answer briefly."}]),
        require_intervention=False,
    )

    assert request.layers == tuple(range(13, 51))


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_chat_accepts_explicit_boolean_thinking_mode(enable_thinking: bool) -> None:
    body = _chat_body([{"role": "user", "content": "Solve this."}])
    body["enable_thinking"] = enable_thinking

    request = parse_request(body, require_intervention=False)

    assert request.enable_thinking is enable_thinking


@pytest.mark.parametrize("enable_thinking", [None, 0, 1, "false", []])
def test_chat_rejects_non_boolean_thinking_mode(enable_thinking: object) -> None:
    body = _chat_body([{"role": "user", "content": "Solve this."}])
    body["enable_thinking"] = enable_thinking

    with pytest.raises(ValidationError):
        parse_request(body, require_intervention=False)


@pytest.mark.parametrize("enable_thinking", [False, True])
def test_raw_prompt_rejects_thinking_mode(enable_thinking: bool) -> None:
    with pytest.raises(ValidationError):
        parse_request(
            {"prompt": "Raw continuation", "enable_thinking": enable_thinking},
            require_intervention=False,
        )


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "user", "content": "First question."}],
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "First question."},
        ],
        [
            {"role": "user", "content": "First question."},
            {"role": "assistant", "content": "First answer."},
            {"role": "user", "content": "Follow-up question."},
        ],
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "First question."},
            {"role": "assistant", "content": "First answer."},
            {"role": "user", "content": "Follow-up question."},
        ],
    ],
)
def test_chat_accepts_optional_leading_system_then_alternating_turns(
    messages: list[dict[str, str]],
) -> None:
    request = parse_request(_chat_body(messages), require_intervention=False)

    assert request.messages == tuple(messages)


@pytest.mark.parametrize(
    "messages",
    [
        [{"role": "system", "content": "No user turn."}],
        [{"role": "assistant", "content": "Starts incorrectly."}],
        [
            {"role": "user", "content": "First question."},
            {"role": "system", "content": "System appears late."},
            {"role": "user", "content": "Follow-up question."},
        ],
        [
            {"role": "user", "content": "First question."},
            {"role": "user", "content": "Second question."},
        ],
        [
            {"role": "user", "content": "First question."},
            {"role": "assistant", "content": "First answer."},
            {"role": "assistant", "content": "Second answer."},
            {"role": "user", "content": "Follow-up question."},
        ],
        [
            {"role": "user", "content": "First question."},
            {"role": "assistant", "content": "Ends incorrectly."},
        ],
        [
            {"role": "system", "content": "Be concise."},
            {"role": "system", "content": "Duplicate system."},
            {"role": "user", "content": "Question."},
        ],
    ],
)
def test_chat_rejects_invalid_role_order(messages: list[dict[str, str]]) -> None:
    with pytest.raises(ValidationError):
        parse_request(_chat_body(messages), require_intervention=False)
