from __future__ import annotations

import json

from core.config import PROMPTS_DIR
from core.llm import llm_registry
from core.memory import save_hypotheses
from core.schemas import Hypothesis, HypothesisList


PROMPT = (PROMPTS_DIR / "enumerator.txt").read_text(encoding="utf-8")


async def enumerate_variants(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    payload = await llm_registry.complete_structured(
        role="enumerator",
        system_prompt=PROMPT,
        user_prompt=f"HYPOTHESES:\n{json.dumps([hypothesis.model_dump() for hypothesis in hypotheses], indent=2)}",
        response_model=HypothesisList,
        temperature=0.5,
    )
    for hypothesis in payload.hypotheses:
        if not hypothesis.source:
            hypothesis.source = "variant"
    save_hypotheses(payload.hypotheses)
    return payload.hypotheses
