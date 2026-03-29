from __future__ import annotations

import json

from core.compaction import chunk_serialized_items
from core.config import PROMPTS_DIR, SAFE_PROMPT_TOKENS, role_endpoint
from core.llm import OversizeRequestError, llm_registry
from core.memory import add_to_graveyard, save_chunk_manifests, save_reviews
from core.schemas import Hypothesis, KilledIdea, KillStage, Review, ReviewList, Verdict


PROMPT = (PROMPTS_DIR / "adversary.txt").read_text(encoding="utf-8")


async def _review_payload(hypotheses: list[Hypothesis], bottleneck: str) -> list[Review]:
    payload = await llm_registry.complete_structured(
        role="adversary",
        system_prompt=PROMPT,
        user_prompt=(
            "TARGET OUTPUT SHAPE:\n"
            "- Return exactly one review object per input hypothesis.\n"
            "- Keep each list field concise.\n"
            "- Keep complexity_vs_gain and revision_direction short.\n\n"
            f"BOTTLENECK:\n{bottleneck}\n\n"
            f"HYPOTHESES:\n{json.dumps([hypothesis.model_dump() for hypothesis in hypotheses], separators=(',', ':'))}"
        ),
        response_model=ReviewList,
        temperature=0.2,
        max_tokens=2200,
    )
    return payload.reviews


def _split_hypothesis_chunk(chunk: list[Hypothesis]) -> list[list[Hypothesis]]:
    midpoint = len(chunk) // 2
    if midpoint <= 0:
        return [chunk]
    return [chunk[:midpoint], chunk[midpoint:]]


async def _review_with_fallback(chunk: list[Hypothesis], bottleneck: str) -> list[Review]:
    try:
        return await _review_payload(chunk, bottleneck)
    except OversizeRequestError:
        if len(chunk) <= 1:
            raise
        reviews: list[Review] = []
        for subchunk in _split_hypothesis_chunk(chunk):
            reviews.extend(await _review_with_fallback(subchunk, bottleneck))
        return reviews


async def red_team(
    *,
    hypotheses: list[Hypothesis],
    bottleneck: str,
) -> tuple[list[Hypothesis], list[Review]]:
    try:
        reviews = await _review_payload(hypotheses, bottleneck)
    except OversizeRequestError:
        endpoint = role_endpoint("adversary")
        chunks = chunk_serialized_items(
            role="adversary",
            brief_version=hypotheses[0].brief_version if hypotheses else "v_unknown",
            model=endpoint.model,
            items=hypotheses,
            serialize_item=lambda hypothesis: hypothesis.model_dump_json(),
            base_context_parts=[
                PROMPT,
                "\n\n",
                "TARGET OUTPUT SHAPE:\n- Return exactly one review object per input hypothesis.\n- Keep each list field concise.\n- Keep complexity_vs_gain and revision_direction short.\n\n",
                "BOTTLENECK:\n",
                bottleneck,
                "\n\nHYPOTHESES:\n",
            ],
            max_completion_tokens=2200,
            safe_input_tokens=max(1000, SAFE_PROMPT_TOKENS.get(endpoint.model, 5000) - 900),
        )
        save_chunk_manifests("redteam", hypotheses[0].brief_version if hypotheses else "v_unknown", [manifest for manifest, _ in chunks])
        reviews = []
        for _manifest, chunk in chunks:
            reviews.extend(await _review_with_fallback(chunk, bottleneck))
    brief_version = hypotheses[0].brief_version if hypotheses else ""
    for review in reviews:
        review.brief_version = brief_version
    save_reviews(reviews, brief_version)

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
