from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from urllib.error import URLError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ollama_adapter import (
    OllamaAdapterError,
    OllamaChatAdapter,
)


class FakeResponse:
    def __init__(
        self,
        payload: dict[str, Any],
    ) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> bool:
        return False

    def read(self) -> bytes:
        return json.dumps(
            self.payload
        ).encode("utf-8")


class QueueOpener:
    def __init__(
        self,
        payloads: list[dict[str, Any]],
    ) -> None:
        self.payloads = list(payloads)
        self.requests: list[tuple[Any, float]] = []

    def __call__(
        self,
        request: Any,
        timeout: float,
    ) -> FakeResponse:
        self.requests.append(
            (request, timeout)
        )

        if not self.payloads:
            raise AssertionError(
                "No fake response remains."
            )

        return FakeResponse(
            self.payloads.pop(0)
        )


def main() -> None:
    model_payload = {
        "models": [
            {
                "name": "qwen3.5:4b",
                "model": "qwen3.5:4b",
                "size": 3_389_983_735,
                "digest": "test-digest",
                "details": {
                    "parameter_size": "4.7B",
                },
            }
        ]
    }

    list_opener = QueueOpener(
        [model_payload]
    )

    adapter = OllamaChatAdapter(
        base_url="http://localhost:11434",
        model="qwen3.5:4b",
        timeout_seconds=30,
        opener=list_opener,
    )

    models = adapter.list_models()

    assert len(models) == 1
    assert models[0]["name"] == "qwen3.5:4b"
    assert models[0]["size"] == 3_389_983_735

    list_request, list_timeout = (
        list_opener.requests[0]
    )

    assert list_request.full_url == (
        "http://localhost:11434/api/tags"
    )
    assert list_request.get_method() == "GET"
    assert list_timeout == 30

    availability_opener = QueueOpener(
        [model_payload]
    )

    available_adapter = OllamaChatAdapter(
        model="qwen3.5:4b",
        opener=availability_opener,
    )

    available_adapter.ensure_model_available()

    chat_payload = {
        "model": "qwen3.5:4b",
        "message": {
            "role": "assistant",
            "content": "LOCAL RESPONSE [1]",
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 120,
        "eval_count": 8,
        "total_duration": 1_000_000,
    }

    chat_opener = QueueOpener(
        [chat_payload]
    )

    chat_adapter = OllamaChatAdapter(
        model="qwen3.5:4b",
        temperature=0,
        num_predict=128,
        keep_alive="5m",
        think=False,
        opener=chat_opener,
    )

    result = chat_adapter.chat(
        [
            {
                "role": "system",
                "content": "Use supplied sources.",
            },
            {
                "role": "user",
                "content": "Answer the question.",
            },
        ]
    )

    assert result["content"] == (
        "LOCAL RESPONSE [1]"
    )
    assert result["model"] == "qwen3.5:4b"
    assert result["eval_count"] == 8

    chat_request, _ = chat_opener.requests[0]

    sent_payload = json.loads(
        chat_request.data.decode("utf-8")
    )

    assert sent_payload["stream"] is False
    assert sent_payload["think"] is False
    assert sent_payload["keep_alive"] == "5m"
    assert sent_payload["options"] == {
        "temperature": 0.0,
        "num_predict": 128,
    }

    assert len(sent_payload["messages"]) == 2

    missing_opener = QueueOpener(
        [
            {
                "models": [],
            }
        ]
    )

    missing_adapter = OllamaChatAdapter(
        model="missing-model",
        opener=missing_opener,
    )

    try:
        missing_adapter.ensure_model_available()
    except OllamaAdapterError as exc:
        assert "is not installed" in str(exc)
    else:
        raise AssertionError(
            "Missing model should raise an error."
        )

    def failing_opener(
        request: Any,
        timeout: float,
    ) -> Any:
        raise URLError("simulated offline service")

    offline_adapter = OllamaChatAdapter(
        opener=failing_opener,
    )

    try:
        offline_adapter.list_models()
    except OllamaAdapterError as exc:
        assert "Could not connect" in str(exc)
    else:
        raise AssertionError(
            "Offline service should raise an error."
        )

    try:
        chat_adapter.chat([])
    except ValueError as exc:
        assert "At least one" in str(exc)
    else:
        raise AssertionError(
            "Empty messages should be rejected."
        )

    print("=" * 80)
    print("OLLAMA ADAPTER VALIDATION PASSED")
    print("=" * 80)
    print("Model listing: PASS")
    print("Model availability: PASS")
    print("Chat payload: PASS")
    print("Response parsing: PASS")
    print("Missing model handling: PASS")
    print("Connection failure handling: PASS")
    print("Input validation: PASS")


if __name__ == "__main__":
    main()
