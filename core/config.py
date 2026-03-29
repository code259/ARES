from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
PROMPTS_DIR = ROOT / "prompts"
LOGS_DIR = OUTPUTS_DIR / "logs"
TOKEN_LOG = LOGS_DIR / "token_usage.jsonl"
RATE_STATE_PATH = LOGS_DIR / "rate_state.json"
CACHE_DIR = OUTPUTS_DIR / "cache" / "llm"


@dataclass(frozen=True)
class ModelEndpoint:
    provider: str
    model: str
    role: str
    api_key: str | None = None
    base_url: str | None = None


DEFAULT_GROQ_MODELS = {
    "scout": "groq/compound",
    "librarian": "moonshotai/kimi-k2-instruct",
    "architect": "openai/gpt-oss-120b",
    "enumerator": "qwen/qwen3-32b",
    "consolidator": "qwen/qwen3-32b",
    "adversary": "openai/gpt-oss-120b",
    "ranker": "qwen/qwen3-32b",
    "spec_writer": "openai/gpt-oss-120b",
    "manual_import": "qwen/qwen3-32b",
}

DEFAULT_PRIMARY_MODELS = {
    **DEFAULT_GROQ_MODELS,
    "architect": "gemini-3-flash-preview",
    "adversary": "gemini-3-flash-preview",
    "spec_writer": "gemini-3-flash-preview",
}

DEFAULT_FALLBACK_MODELS = {
    "architect": "gemini-2.5-flash",
    "adversary": "gemini-2.5-flash",
    "spec_writer": "gemini-2.5-flash",
}


MODEL_LIMITS: dict[str, dict[str, int | None]] = {
    "groq/compound": {"rpm": 30, "rpd": 250, "tpm": 70000, "tpd": None},
    "groq/compound-mini": {"rpm": 30, "rpd": 250, "tpm": 70000, "tpd": None},
    "llama-3.1-8b-instant": {"rpm": 30, "rpd": 14400, "tpm": 6000, "tpd": 500000},
    "llama-3.3-70b-versatile": {"rpm": 30, "rpd": 1000, "tpm": 12000, "tpd": 100000},
    "moonshotai/kimi-k2-instruct": {"rpm": 60, "rpd": 1000, "tpm": 10000, "tpd": 300000},
    "moonshotai/kimi-k2-instruct-0905": {"rpm": 60, "rpd": 1000, "tpm": 10000, "tpd": 300000},
    "openai/gpt-oss-120b": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
    "openai/gpt-oss-20b": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
    "qwen/qwen3-32b": {"rpm": 60, "rpd": 1000, "tpm": 6000, "tpd": 500000},
    "gemini-3-flash-preview": {"rpm": 5, "rpd": 20, "tpm": 250000, "tpd": None},
    "gemini-2.5-flash": {"rpm": 5, "rpd": 20, "tpm": 250000, "tpd": None},
}


def groq_keys() -> list[str]:
    numbered: list[tuple[int, str]] = []
    for name, value in os.environ.items():
        if not name.startswith("GROQ_KEY_") or not value:
            continue
        suffix = name.removeprefix("GROQ_KEY_")
        try:
            order = int(suffix)
        except ValueError:
            continue
        numbered.append((order, value))
    return [value for _, value in sorted(numbered)]


SAFE_PROMPT_TOKENS = {
    "groq/compound": 12000,
    "groq/compound-mini": 12000,
    "llama-3.1-8b-instant": 4500,
    "llama-3.3-70b-versatile": 8000,
    "moonshotai/kimi-k2-instruct": 7000,
    "moonshotai/kimi-k2-instruct-0905": 7000,
    "openai/gpt-oss-120b": 5500,
    "openai/gpt-oss-20b": 5500,
    "qwen/qwen3-32b": 4200,
    "gemini-3-flash-preview": 120000,
    "gemini-2.5-flash": 120000,
}


def gemini_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY") or None


def provider_for_model(model: str) -> str:
    if model.startswith("gemini"):
        return "gemini"
    return "groq"


def _build_endpoint(role: str, model: str) -> ModelEndpoint:
    provider = provider_for_model(model)
    if provider == "gemini":
        return ModelEndpoint(
            provider="gemini",
            model=model,
            role=role,
            api_key=gemini_api_key(),
            base_url=None,
        )
    groq_key = groq_keys()[0] if groq_keys() else None
    return ModelEndpoint(
        provider="groq",
        model=model,
        role=role,
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
    )


def role_endpoint(role: str) -> ModelEndpoint:
    role_upper = role.upper()
    model = os.getenv(f"{role_upper}_MODEL") or DEFAULT_PRIMARY_MODELS.get(role, "llama-3.3-70b-versatile")
    return _build_endpoint(role, model)


def role_fallback_endpoints(role: str) -> list[ModelEndpoint]:
    role_upper = role.upper()
    endpoints: list[ModelEndpoint] = []
    primary = role_endpoint(role)
    endpoints.append(primary)

    fallback_model = os.getenv(f"{role_upper}_FALLBACK_MODEL") or DEFAULT_FALLBACK_MODELS.get(role)
    if fallback_model and fallback_model != primary.model:
        endpoints.append(_build_endpoint(role, fallback_model))

    groq_model = DEFAULT_GROQ_MODELS.get(role)
    if groq_model and all(endpoint.model != groq_model for endpoint in endpoints):
        endpoints.append(_build_endpoint(role, groq_model))
    return endpoints


def require_any_api_key() -> None:
    if groq_keys() or gemini_api_key():
        return
    raise RuntimeError(
        "No API keys configured. Add at least one GROQ_KEY_* or GEMINI_API_KEY in .env.",
    )
