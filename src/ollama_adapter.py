from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(
    PROJECT_ROOT / ".env",
    override=False,
)


class OllamaAdapterError(RuntimeError):
    """Raised when the local Ollama service cannot complete a request."""


class OllamaChatAdapter:
    """Provider adapter for the local Ollama REST API."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
        keep_alive: str | None = None,
        think: bool = False,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            )
        ).rstrip("/")

        self.model = (
            model
            or os.getenv(
                "OLLAMA_MODEL",
                "qwen3.5:4b",
            )
        )

        self.timeout_seconds = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.getenv(
                "OLLAMA_TIMEOUT_SECONDS",
                "300",
            )
        )

        self.temperature = float(
            temperature
            if temperature is not None
            else os.getenv(
                "OLLAMA_TEMPERATURE",
                "0",
            )
        )

        self.num_predict = int(
            num_predict
            if num_predict is not None
            else os.getenv(
                "OLLAMA_NUM_PREDICT",
                "512",
            )
        )

        self.keep_alive = (
            keep_alive
            or os.getenv(
                "OLLAMA_KEEP_ALIVE",
                "5m",
            )
        )

        self.think = bool(think)
        self._opener = opener or urlopen

        if not self.base_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError(
                "OLLAMA_BASE_URL must begin with "
                "http:// or https://."
            )

        if not self.model.strip():
            raise ValueError(
                "The Ollama model name cannot be empty."
            )

        if self.timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be positive."
            )

        if self.num_predict < 1:
            raise ValueError(
                "num_predict must be at least 1."
            )

    def _request_json(
        self,
        path: str,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"

        request_data = None

        if payload is not None:
            request_data = json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8")

        request = Request(
            url=url,
            data=request_data,
            method=method,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

        try:
            with self._opener(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                raw_response = response.read().decode(
                    "utf-8"
                )

        except HTTPError as exc:
            try:
                error_body = exc.read().decode(
                    "utf-8",
                    errors="replace",
                )
            except Exception:
                error_body = ""

            detail = error_body.strip() or str(exc)

            raise OllamaAdapterError(
                f"Ollama returned HTTP {exc.code}: "
                f"{detail}"
            ) from exc

        except URLError as exc:
            raise OllamaAdapterError(
                "Could not connect to the local Ollama "
                f"service at {self.base_url}. "
                "Confirm that Ollama is running."
            ) from exc

        except TimeoutError as exc:
            raise OllamaAdapterError(
                "The Ollama request exceeded the timeout "
                f"of {self.timeout_seconds:g} seconds."
            ) from exc

        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise OllamaAdapterError(
                "Ollama returned invalid JSON."
            ) from exc

        if not isinstance(parsed, dict):
            raise OllamaAdapterError(
                "Ollama returned an unexpected response."
            )

        if parsed.get("error"):
            raise OllamaAdapterError(
                f"Ollama error: {parsed['error']}"
            )

        return parsed

    def list_models(
        self,
    ) -> list[dict[str, Any]]:
        """Return locally available Ollama models."""
        response = self._request_json(
            path="/api/tags",
        )

        models = response.get("models", [])

        if not isinstance(models, list):
            raise OllamaAdapterError(
                "Ollama returned an invalid model list."
            )

        return [
            {
                "name": model.get(
                    "name",
                    model.get("model"),
                ),
                "model": model.get(
                    "model",
                    model.get("name"),
                ),
                "size": model.get("size"),
                "digest": model.get("digest"),
                "details": model.get(
                    "details",
                    {},
                ),
            }
            for model in models
            if isinstance(model, dict)
        ]

    def ensure_model_available(self) -> None:
        """Raise an error when the configured model is absent."""
        model_names = {
            str(
                model.get("name")
                or model.get("model")
                or ""
            )
            for model in self.list_models()
        }

        if self.model not in model_names:
            available = (
                ", ".join(sorted(model_names))
                if model_names
                else "none"
            )

            raise OllamaAdapterError(
                f"Model {self.model!r} is not installed. "
                f"Available models: {available}."
            )

    def chat(
        self,
        messages: list[dict[str, str]],
    ) -> dict[str, Any]:
        """Generate one non-streaming local chat response."""
        if not messages:
            raise ValueError(
                "At least one chat message is required."
            )

        allowed_roles = {
            "system",
            "user",
            "assistant",
        }

        cleaned_messages: list[
            dict[str, str]
        ] = []

        for message in messages:
            role = str(
                message.get("role", "")
            ).strip()

            content = str(
                message.get("content", "")
            ).strip()

            if role not in allowed_roles:
                raise ValueError(
                    f"Unsupported message role: {role!r}."
                )

            if not content:
                raise ValueError(
                    "Chat message content cannot be empty."
                )

            cleaned_messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

        payload = {
            "model": self.model,
            "messages": cleaned_messages,
            "stream": False,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }

        response = self._request_json(
            path="/api/chat",
            method="POST",
            payload=payload,
        )

        message = response.get("message")

        if not isinstance(message, dict):
            raise OllamaAdapterError(
                "Ollama response has no assistant message."
            )

        content = str(
            message.get("content", "")
        ).strip()

        if not content:
            raise OllamaAdapterError(
                "Ollama returned an empty response."
            )

        return {
            "content": content,
            "model": response.get(
                "model",
                self.model,
            ),
            "done": response.get("done"),
            "done_reason": response.get(
                "done_reason"
            ),
            "prompt_eval_count": response.get(
                "prompt_eval_count"
            ),
            "eval_count": response.get(
                "eval_count"
            ),
            "load_duration": response.get(
                "load_duration"
            ),
            "prompt_eval_duration": response.get(
                "prompt_eval_duration"
            ),
            "eval_duration": response.get(
                "eval_duration"
            ),
            "total_duration": response.get(
                "total_duration"
            ),
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test the local Ollama chat adapter."
        )
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to send to the local model.",
    )
    parser.add_argument(
        "--model",
        help="Override the configured Ollama model.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List locally installed models.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete response as JSON.",
    )

    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = build_parser().parse_args()

    adapter = OllamaChatAdapter(
        model=args.model,
    )

    if args.list_models:
        print(
            json.dumps(
                adapter.list_models(),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if not args.prompt:
        raise SystemExit(
            "Provide a prompt or use --list-models."
        )

    adapter.ensure_model_available()

    result = adapter.chat(
        messages=[
            {
                "role": "user",
                "content": args.prompt,
            }
        ]
    )

    if args.json:
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(result["content"])


if __name__ == "__main__":
    main()
