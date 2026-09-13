"""Minimal OpenAI Responses API client for semantic instantiation."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class OpenAIResponsesClient:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int = 60
    max_output_tokens: int = 1200

    @classmethod
    def from_env(cls) -> "OpenAIResponsesClient":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for semantic-mode=llm")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("OPENAI_MODEL")
        if not model:
            raise RuntimeError("OPENAI_MODEL is required for semantic-mode=llm")
        timeout_seconds = int(os.environ.get("OPENAI_TIMEOUT_SECONDS", "60"))
        max_output_tokens = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "1200"))
        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            model=model,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
        )

    def __repr__(self) -> str:
        return (
            "OpenAIResponsesClient("
            f"base_url={self.base_url!r}, model={self.model!r}, "
            f"timeout_seconds={self.timeout_seconds}, "
            f"max_output_tokens={self.max_output_tokens})"
        )

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": self.max_output_tokens,
        }
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Responses API HTTP {exc.code}: {body[:500]}") from exc
        return extract_response_text(data)


def extract_response_text(response: Dict[str, Any]) -> str:
    texts = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            content_type = content.get("type")
            if content_type in {"output_text", "text"}:
                texts.append(str(content.get("text", "")))
    if texts:
        return "\n".join(texts).strip()
    if "output_text" in response:
        return str(response["output_text"]).strip()
    raise RuntimeError("Responses API result did not contain output text")
