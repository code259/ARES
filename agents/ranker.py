from __future__ import annotations

import json

from core.compaction import chunk_serialized_items
from core.config import PROMPTS_DIR, SAFE_PROMPT_TOKENS, role_endpoint
from core.llm import OversizeRequestError, llm_registry
from core.memory import save_chunk_manifests
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
    review_map = {review.hypothesis_id: review for review in reviews}

    async def _rank_payload(chunk_survivors: list[Hypothesis]) -> list:
        chunk_reviews = [review_map[hypothesis.id] for hypothesis in chunk_survivors if hypothesis.id in review_map]
        chunk_flags = {hypothesis.id: flags.get(hypothesis.id, False) for hypothesis in chunk_survivors}
        payload = await llm_registry.complete_structured(
            role="ranker",
            system_prompt=PROMPT,
            user_prompt=(
                "TARGET OUTPUT SHAPE:\n"
                "- Return exactly one rank object per surviving hypothesis in this call.\n"
                "- Keep rationale concise but specific.\n\n"
                f"SURVIVING HYPOTHESES:\n{json.dumps([hypothesis.model_dump() for hypothesis in chunk_survivors], separators=(',', ':'))}\n\n"
                f"REVIEWS:\n{json.dumps([review.model_dump() for review in chunk_reviews], separators=(',', ':'))}\n\n"
                f"CONSENSUS FLAGS:\n{json.dumps(chunk_flags, separators=(',', ':'))}"
            ),
            response_model=RankList,
            temperature=0.1,
            max_tokens=1800,
        )
        return payload.ranks

    def _split_hypothesis_chunk(chunk: list[Hypothesis]) -> list[list[Hypothesis]]:
        midpoint = len(chunk) // 2
        if midpoint <= 0:
            return [chunk]
        return [chunk[:midpoint], chunk[midpoint:]]

    async def _rank_with_fallback(chunk: list[Hypothesis]) -> list:
        try:
            return await _rank_payload(chunk)
        except OversizeRequestError:
            if len(chunk) <= 1:
                raise
            ranks = []
            for subchunk in _split_hypothesis_chunk(chunk):
                ranks.extend(await _rank_with_fallback(subchunk))
            return ranks

    try:
        ranks = await _rank_payload(survivors)
    except OversizeRequestError:
        endpoint = role_endpoint("ranker")
        chunks = chunk_serialized_items(
            role="ranker",
            brief_version=survivors[0].brief_version if survivors else "v_unknown",
            model=endpoint.model,
            items=survivors,
            serialize_item=lambda hypothesis: hypothesis.model_dump_json(),
            base_context_parts=[
                PROMPT,
                "\n\n",
                "TARGET OUTPUT SHAPE:\n- Return exactly one rank object per surviving hypothesis in this call.\n- Keep rationale concise but specific.\n\n",
                "SURVIVING HYPOTHESES:\n",
                "\n\nREVIEWS:\n",
                "\n\nCONSENSUS FLAGS:\n",
            ],
            max_completion_tokens=1800,
            safe_input_tokens=max(1000, SAFE_PROMPT_TOKENS.get(endpoint.model, 4000) - 700),
        )
        save_chunk_manifests("ranker", survivors[0].brief_version if survivors else "v_unknown", [manifest for manifest, _ in chunks])
        ranks = []
        for _manifest, chunk in chunks:
            ranks.extend(await _rank_with_fallback(chunk))

    payload = RankList(ranks=ranks)
    for rank in payload.ranks:
        if flags.get(rank.hypothesis_id):
            rank.consensus_flag = True
            rank.composite_score = min(10.0, rank.composite_score + 0.5)
    payload.ranks.sort(key=lambda item: item.composite_score, reverse=True)
    for index, rank in enumerate(payload.ranks, start=1):
        rank.recommended_order = index
    return payload
