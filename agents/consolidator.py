from __future__ import annotations

import json

from core.config import PROMPTS_DIR
from core.llm import llm_registry
from core.schemas import ConsensusClusters, Hypothesis


PROMPT = (PROMPTS_DIR / "consolidator.txt").read_text(encoding="utf-8")


async def cluster_and_flag(all_hypotheses: list[Hypothesis]) -> ConsensusClusters:
    return await llm_registry.complete_structured(
        role="consolidator",
        system_prompt=PROMPT,
        user_prompt=f"ALL HYPOTHESES (from all passes):\n{json.dumps([hypothesis.model_dump() for hypothesis in all_hypotheses], indent=2)}",
        response_model=ConsensusClusters,
        temperature=0.2,
    )
