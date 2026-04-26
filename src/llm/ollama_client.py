"""
Ollama REST API client with retry logic, context isolation, and streaming support.
"""
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Generator, Optional
import httpx
from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)


class OllamaClient:
    """
    Thin wrapper around the Ollama /api/chat endpoint.
    Keeps system and user prompts in separate context slots to prevent
    injection via conversation history bleed.
    """

    def __init__(self, model: Optional[str] = None):
        cfg = get_settings().ollama
        self.base_url = cfg.base_url.rstrip("/")
        self.model = model or cfg.default_model
        self.timeout = cfg.timeout
        self.max_retries = cfg.max_retries
        self._client = httpx.Client(timeout=self.timeout)

    def _build_payload(
        self,
        messages: list[Message],
        temperature: float = 0.7,
        stream: bool = False,
    ) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": {"temperature": temperature},
        }

    def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Single-turn chat with explicit context isolation.
        System prompt is always injected as the first message of role 'system',
        ensuring user content cannot override it via conversation threading.
        """
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=user_prompt))

        payload = self._build_payload(messages, temperature=temperature, stream=False)
        return self._post_with_retry(payload)

    def stream_chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=user_prompt))

        payload = self._build_payload(messages, temperature=temperature, stream=True)

        with self._client.stream("POST", f"{self.base_url}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        yield delta
                    if chunk.get("done"):
                        break

    def _post_with_retry(self, payload: dict) -> LLMResponse:
        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(1, self.max_retries + 1):
            try:
                t0 = time.perf_counter()
                resp = self._client.post(f"{self.base_url}/api/chat", json=payload)
                resp.raise_for_status()
                latency = (time.perf_counter() - t0) * 1000

                body = resp.json()
                content = body.get("message", {}).get("content", "")
                return LLMResponse(
                    content=content,
                    model=body.get("model", self.model),
                    prompt_tokens=body.get("prompt_eval_count", 0),
                    completion_tokens=body.get("eval_count", 0),
                    latency_ms=round(latency, 2),
                    raw=body,
                )
            except (httpx.HTTPError, httpx.ConnectError) as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning("Ollama request failed (attempt %d/%d): %s. Retrying in %ds.",
                               attempt, self.max_retries, exc, wait)
                time.sleep(wait)

        raise RuntimeError(f"Ollama unreachable after {self.max_retries} retries: {last_exc}") from last_exc

    def is_available(self) -> bool:
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        resp = self._client.get(f"{self.base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
