from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, TypeVar

from openai import APIConnectionError, AsyncOpenAI, RateLimitError
from pydantic import BaseModel

from core.config import CACHE_DIR, MODEL_LIMITS, RATE_STATE_PATH, TOKEN_LOG, groq_keys, require_any_api_key, role_endpoint


T = TypeVar("T", bound=BaseModel)


class JsonParseError(RuntimeError):
    pass


class DailyBudgetExhausted(RuntimeError):
    pass


class OversizeRequestError(RuntimeError):
    pass


def _extract_json_block(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        candidate = stripped
    else:
        fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", stripped, re.DOTALL)
        if fenced:
            candidate = fenced.group(1).strip()
        else:
            first_obj = stripped.find("{")
            first_arr = stripped.find("[")
            candidates = [idx for idx in [first_obj, first_arr] if idx != -1]
            if not candidates:
                raise JsonParseError("No JSON object or array found in model output.")
            candidate = stripped[min(candidates):].strip()

    opening = candidate[0]
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(candidate):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == opening:
            depth += 1
            continue
        if char == closing:
            depth -= 1
            if depth == 0:
                return candidate[: index + 1]

    raise JsonParseError("Could not find a complete JSON block in model output.")


class LLMRegistry:
    def __init__(self) -> None:
        self._groq_keys = groq_keys()
        self._groq_index = 0
        self._lock = Lock()
        self._groq_usage: dict[str, int] = {key: 0 for key in self._groq_keys}
        TOKEN_LOG.parent.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._rate_state = self._load_rate_state()

    @staticmethod
    def _today_key() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load_rate_state(self) -> dict[str, Any]:
        if RATE_STATE_PATH.exists():
            return json.loads(RATE_STATE_PATH.read_text(encoding="utf-8"))
        return {"days": {}, "temporary_backoff": {}}

    def _save_rate_state(self) -> None:
        RATE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        RATE_STATE_PATH.write_text(json.dumps(self._rate_state, indent=2), encoding="utf-8")

    def _day_bucket(self, key: str, model: str) -> dict[str, Any]:
        today = self._today_key()
        days = self._rate_state.setdefault("days", {})
        day_state = days.setdefault(today, {})
        key_state = day_state.setdefault(key[-6:], {})
        model_state = key_state.setdefault(
            model,
            {
                "requests_day": 0,
                "tokens_day": 0,
                "minute_window_start": 0,
                "requests_minute": 0,
                "tokens_minute": 0,
            },
        )
        return model_state

    @staticmethod
    def _approx_tokens(*parts: str, max_tokens: int = 0) -> int:
        text = "".join(parts)
        return max(1, len(text) // 4) + max_tokens

    def _cache_path(
        self,
        *,
        role: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Path:
        key = json.dumps(
            {
                "role": role,
                "model": model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
        )
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return CACHE_DIR / f"{digest}.json"

    def _load_cache(self, path: Path, response_model: type[T]) -> T | None:
        if not path.exists():
            return None
        try:
            return response_model.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _write_cache(path: Path, parsed: BaseModel) -> None:
        path.write_text(parsed.model_dump_json(indent=2), encoding="utf-8")

    def _reset_minute_if_needed(self, state: dict[str, Any]) -> None:
        current_minute = int(time.time() // 60)
        if state["minute_window_start"] != current_minute:
            state["minute_window_start"] = current_minute
            state["requests_minute"] = 0
            state["tokens_minute"] = 0

    def _candidate_wait_seconds(self, key: str, model: str, estimated_tokens: int) -> float:
        limits = MODEL_LIMITS.get(model, {})
        state = self._day_bucket(key, model)
        self._reset_minute_if_needed(state)

        tpd = limits.get("tpd")
        rpd = limits.get("rpd")
        tpm = limits.get("tpm")
        rpm = limits.get("rpm")

        if tpd is not None and state["tokens_day"] + estimated_tokens > tpd:
            return float("inf")
        if rpd is not None and state["requests_day"] + 1 > rpd:
            return float("inf")

        wait_seconds = 0.0
        if tpm is not None and state["tokens_minute"] + estimated_tokens > tpm:
            wait_seconds = max(wait_seconds, 61 - (time.time() % 60))
        if rpm is not None and state["requests_minute"] + 1 > rpm:
            wait_seconds = max(wait_seconds, 61 - (time.time() % 60))

        backoff_until = self._rate_state.setdefault("temporary_backoff", {}).get(f"{key[-6:]}::{model}", 0)
        if backoff_until > time.time():
            wait_seconds = max(wait_seconds, backoff_until - time.time())
        return wait_seconds

    def _reserve_key_for_model(self, model: str, estimated_tokens: int) -> tuple[str, float]:
        if not self._groq_keys:
            raise RuntimeError("No Groq keys are configured.")

        waits = [(key, self._candidate_wait_seconds(key, model, estimated_tokens)) for key in self._groq_keys]
        immediate = [item for item in waits if item[1] == 0]
        if immediate:
            key = immediate[self._groq_index % len(immediate)][0]
            self._groq_index = (self._groq_index + 1) % len(self._groq_keys)
            return key, 0.0

        finite = [item for item in waits if item[1] != float("inf")]
        if not finite:
            raise DailyBudgetExhausted(
                f"All Groq keys exhausted for model {model} today. Resume after UTC date rollover.",
            )
        key, wait_seconds = min(finite, key=lambda item: item[1])
        return key, wait_seconds

    def _apply_usage(self, key: str, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        total_tokens = prompt_tokens + completion_tokens
        state = self._day_bucket(key, model)
        self._reset_minute_if_needed(state)
        state["requests_day"] += 1
        state["tokens_day"] += total_tokens
        state["requests_minute"] += 1
        state["tokens_minute"] += total_tokens
        self._save_rate_state()

    def _next_groq_key(self) -> str:
        if not self._groq_keys:
            raise RuntimeError("No Groq keys are configured.")
        with self._lock:
            key = self._groq_keys[self._groq_index]
            self._groq_index = (self._groq_index + 1) % len(self._groq_keys)
        return key

    @staticmethod
    def _client(api_key: str, base_url: str | None) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _log_usage(
        self,
        role: str,
        provider: str,
        model: str,
        key_suffix: str,
        usage: Any,
        cache_hit: bool = False,
    ) -> None:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0

        if provider == "groq":
            for key in self._groq_keys:
                if key.endswith(key_suffix):
                    self._groq_usage[key] += total_tokens
                    running_total = self._groq_usage[key]
                    break
            else:
                running_total = total_tokens
        else:
            running_total = total_tokens

        entry = {
            "timestamp": time.time(),
            "role": role,
            "provider": provider,
            "model": model,
            "key_suffix": key_suffix,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "running_total": running_total,
            "cache_hit": cache_hit,
        }
        with TOKEN_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    async def complete_structured(
        self,
        *,
        role: str,
        system_prompt: str,
        user_prompt: str,
        response_model: type[T],
        temperature: float = 0.3,
        max_tokens: int = 4000,
        retries: int = 3,
    ) -> T:
        require_any_api_key()
        self._groq_keys = groq_keys()
        for key in self._groq_keys:
            self._groq_usage.setdefault(key, 0)
        endpoint = role_endpoint(role)
        last_error: Exception | None = None
        limits = MODEL_LIMITS.get(endpoint.model, {})
        cache_path = self._cache_path(
            role=role,
            model=endpoint.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        cached = self._load_cache(cache_path, response_model)
        if cached is not None:
            self._log_usage(
                role=role,
                provider=endpoint.provider,
                model=endpoint.model,
                key_suffix="cache",
                usage=type("Usage", (), {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})(),
                cache_hit=True,
            )
            return cached

        for _attempt in range(retries):
            api_key = endpoint.api_key
            base_url = endpoint.base_url

            if endpoint.provider == "groq":
                estimated_tokens = self._approx_tokens(system_prompt, user_prompt, max_tokens=max_tokens)
                tpm_limit = limits.get("tpm")
                if tpm_limit is not None and estimated_tokens > tpm_limit:
                    raise OversizeRequestError(
                        f"Estimated request size {estimated_tokens} tokens exceeds per-minute model limit "
                        f"{tpm_limit} for {endpoint.model}. Shrink the payload before retrying.",
                    )
                api_key, wait_seconds = self._reserve_key_for_model(endpoint.model, estimated_tokens)
                if wait_seconds > 0:
                    time.sleep(wait_seconds)

            if not api_key:
                raise RuntimeError(f"No API key configured for role '{role}'.")

            client = self._client(api_key=api_key, base_url=base_url)

            try:
                response = await client.chat.completions.create(
                    model=endpoint.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                message = response.choices[0].message.content or ""
                payload = _extract_json_block(message)
                parsed = response_model.model_validate_json(payload)
                self._write_cache(cache_path, parsed)
                if endpoint.provider == "groq":
                    self._apply_usage(
                        key=api_key,
                        model=endpoint.model,
                        prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
                    )
                self._log_usage(
                    role=role,
                    provider=endpoint.provider,
                    model=endpoint.model,
                    key_suffix=api_key[-6:],
                    usage=response.usage,
                )
                return parsed
            except RateLimitError as exc:
                last_error = exc
                backoff = self._rate_state.setdefault("temporary_backoff", {})
                backoff[f"{api_key[-6:]}::{endpoint.model}"] = time.time() + 65
                self._save_rate_state()
                continue
            except APIConnectionError as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(f"Failed structured completion for role '{role}': {last_error}")


llm_registry = LLMRegistry()
