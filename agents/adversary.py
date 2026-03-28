from __future__ import annotations

import json

from core.config import PROMPTS_DIR
from core.llm import llm_registry
from core.memory import add_to_graveyard, save_reviews
from core.schemas import Hypothesis, KilledIdea, KillStage, Review, ReviewList, Verdict


PROMPT = (PROMPTS_DIR / "adversary.txt").read_text(encoding="utf-8")


async def red_team(
    *,
    hypotheses: list[Hypothesis],
    bottleneck: str,
) -> tuple[list[Hypothesis], list[Review]]:
    payload = await llm_registry.complete_structured(
        role="adversary",
        system_prompt=PROMPT,
        user_prompt=f"BOTTLENECK:\n{bottleneck}\n\nHYPOTHESES:\n{json.dumps([hypothesis.model_dump() for hypothesis in hypotheses], indent=2)}",
        response_model=ReviewList,
        temperature=0.2,
    )
    reviews = payload.reviews
    save_reviews(reviews)

    lookup = {hypothesis.id: hypothesis for hypothesis in hypotheses}
    survivors: list[Hypothesis] = []
    for review in reviews:
        hypothesis = lookup.get(review.hypothesis_id)
        if not hypothesis:
            continue
        if review.verdict == Verdict.KILL:
            add_to_graveyard(
                KilledIdea(
                    id=hypothesis.id,
                    name=hypothesis.name,
                    hypothesis=hypothesis.hypothesis,
                    source=hypothesis.source,
                    brief_version=hypothesis.brief_version,
                    kill_reason=review.complexity_vs_gain or "Killed during red-team review.",
                    fatal_flaws=review.fatal_flaws,
                    killed_at_stage=KillStage.RED_TEAM,
                ),
            )
        else:
            survivors.append(hypothesis)
    return survivors, reviews
