from __future__ import annotations

from pathlib import Path

from core.config import PROMPTS_DIR
from core.llm import llm_registry
from core.schemas import ScoutOutput


PROMPT = (PROMPTS_DIR / "scout.txt").read_text(encoding="utf-8")


async def generate_queries(bottleneck: str, pipeline_description: str) -> list[str]:
    result = await llm_registry.complete_structured(
        role="scout",
        system_prompt=PROMPT,
        user_prompt=f"BOTTLENECK:\n{bottleneck}\n\nCURRENT SYSTEM:\n{pipeline_description}",
        response_model=ScoutOutput,
        temperature=0.2,
    )
    return result.queries
