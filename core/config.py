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
    "manual_import": "llama-3.1-8b-instant",
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
}


def groq_keys() -> list[str]:
    keys = [
        os.getenv("GROQ_KEY_1"),
        os.getenv("GROQ_KEY_2"),
        os.getenv("GROQ_KEY_3"),
    ]
    return [key for key in keys if key]


def role_endpoint(role: str) -> ModelEndpoint:
    role_upper = role.upper()
    model = os.getenv(f"{role_upper}_MODEL") or DEFAULT_GROQ_MODELS.get(role, "llama-3.3-70b-versatile")
    groq_key = groq_keys()[0] if groq_keys() else None

    return ModelEndpoint(
        provider="groq",
        model=model,
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
    )


def require_any_api_key() -> None:
    if groq_keys():
        return
    raise RuntimeError(
        "No API keys configured. Add at least one GROQ_KEY_* in .env.",
    )
