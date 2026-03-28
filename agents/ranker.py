from __future__ import annotations

import json

from core.config import PROMPTS_DIR
from core.llm import llm_registry
from core.schemas import ConsensusClusters, Hypothesis, RankList, Review


PROMPT = (PROMPTS_DIR / "ranker.txt").read_text(encoding="utf-8")


def _consensus_flags(survivors: list[Hypothesis], clusters: ConsensusClusters) -> dict[str, bool]:
    flags = {hypothesis.id: False for hypothesis in survivors}
    for cluster in clusters.clusters:
        if not cluster.consensus_flag:
            continue
        for member_id in cluster.member_ids:
            if member_id in flags:
                flags[member_id] = True
    return flags


async def rank_hypotheses(
    *,
    survivors: list[Hypothesis],
    reviews: list[Review],
    clusters: ConsensusClusters,
) -> RankList:
    flags = _consensus_flags(survivors, clusters)
    payload = await llm_registry.complete_structured(
        role="ranker",
        system_prompt=PROMPT,
        user_prompt=(
            f"SURVIVING HYPOTHESES:\n{json.dumps([hypothesis.model_dump() for hypothesis in survivors], indent=2)}\n\n"
            f"REVIEWS:\n{json.dumps([review.model_dump() for review in reviews], indent=2)}\n\n"
            f"CONSENSUS FLAGS:\n{json.dumps(flags, indent=2)}"
        ),
        response_model=RankList,
        temperature=0.1,
    )
    for rank in payload.ranks:
        if flags.get(rank.hypothesis_id):
            rank.consensus_flag = True
            rank.composite_score = min(10.0, rank.composite_score + 0.5)
    payload.ranks.sort(key=lambda item: item.composite_score, reverse=True)
    for index, rank in enumerate(payload.ranks, start=1):
        rank.recommended_order = index
    return payload
