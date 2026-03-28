from __future__ import annotations

import json

from core.config import PROMPTS_DIR
from core.llm import llm_registry
from core.memory import graveyard_summary, save_hypotheses
from core.schemas import HypothesisList, PaperRecord


GROUNDED_PROMPT = (PROMPTS_DIR / "architect_grounded.txt").read_text(encoding="utf-8")
FREE_RANGE_PROMPT = (PROMPTS_DIR / "architect_free_range.txt").read_text(encoding="utf-8")


async def generate_grounded(
    *,
    bottleneck: str,
    pipeline_description: str,
    papers: list[PaperRecord],
    method_families: str,
    brief_version: str,
) -> list:
    payload = await llm_registry.complete_structured(
        role="architect",
        system_prompt=GROUNDED_PROMPT,
        user_prompt=(
            f"BOTTLENECK:\n{bottleneck}\n\n"
            f"PIPELINE:\n{pipeline_description}\n\n"
            f"PAPER RECORDS:\n{json.dumps([paper.model_dump() for paper in papers], indent=2)}\n\n"
            f"METHOD FAMILIES IDENTIFIED:\n{method_families}\n\n"
            f"GRAVEYARD — IDEAS ALREADY KILLED (do not regenerate these):\n{graveyard_summary()}\n\n"
            f"BRIEF VERSION: {brief_version}"
        ),
        response_model=HypothesisList,
        temperature=0.4,
    )
    for hypothesis in payload.hypotheses:
        hypothesis.source = "grounded"
        hypothesis.brief_version = brief_version
    save_hypotheses(payload.hypotheses)
    return payload.hypotheses


async def generate_free_range(
    *,
    bottleneck: str,
    pipeline_description: str,
    brief_version: str,
) -> list:
    payload = await llm_registry.complete_structured(
        role="architect",
        system_prompt=FREE_RANGE_PROMPT,
        user_prompt=(
            f"BOTTLENECK:\n{bottleneck}\n\n"
            f"PIPELINE (brief):\n{pipeline_description}\n\n"
            f"GRAVEYARD — IDEAS ALREADY KILLED (do not regenerate these):\n{graveyard_summary()}\n\n"
            f"BRIEF VERSION: {brief_version}"
        ),
        response_model=HypothesisList,
        temperature=0.6,
    )
    for hypothesis in payload.hypotheses:
        hypothesis.source = "free_range"
        hypothesis.brief_version = brief_version
    save_hypotheses(payload.hypotheses)
    return payload.hypotheses
