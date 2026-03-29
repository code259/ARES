from __future__ import annotations

import json

from core.compaction import chunk_serialized_items
from core.config import PROMPTS_DIR, SAFE_PROMPT_TOKENS, role_endpoint
from core.llm import OversizeRequestError, llm_registry
from core.memory import save_chunk_manifests
from core.schemas import ConsensusClusters, Hypothesis


PROMPT = (PROMPTS_DIR / "consolidator.txt").read_text(encoding="utf-8")


async def _cluster_payload(all_hypotheses: list[Hypothesis]) -> ConsensusClusters:
    return await llm_registry.complete_structured(
        role="consolidator",
        system_prompt=PROMPT,
        user_prompt=f"ALL HYPOTHESES (from all passes):\n{json.dumps([hypothesis.model_dump() for hypothesis in all_hypotheses], separators=(',', ':'))}",
        response_model=ConsensusClusters,
        temperature=0.2,
        max_tokens=1500,
    )


def _split_hypothesis_chunk(chunk: list[Hypothesis]) -> list[list[Hypothesis]]:
    midpoint = len(chunk) // 2
    if midpoint <= 0:
        return [chunk]
    return [chunk[:midpoint], chunk[midpoint:]]


async def _cluster_with_fallback(chunk: list[Hypothesis]) -> ConsensusClusters:
    try:
        return await _cluster_payload(chunk)
    except OversizeRequestError:
        if len(chunk) <= 1:
            raise
        partial_clusters = []
        for subchunk in _split_hypothesis_chunk(chunk):
            partial = await _cluster_with_fallback(subchunk)
            partial_clusters.extend(partial.clusters)
        return ConsensusClusters(clusters=partial_clusters)


async def cluster_and_flag(all_hypotheses: list[Hypothesis]) -> ConsensusClusters:
    try:
        return await _cluster_payload(all_hypotheses)
    except OversizeRequestError:
        endpoint = role_endpoint("consolidator")
        chunks = chunk_serialized_items(
            role="consolidator",
            brief_version=all_hypotheses[0].brief_version if all_hypotheses else "v_unknown",
            model=endpoint.model,
            items=all_hypotheses,
            serialize_item=lambda hypothesis: hypothesis.model_dump_json(),
            base_context_parts=[PROMPT, "\n\n", "ALL HYPOTHESES (from all passes):\n"],
            max_completion_tokens=1500,
            safe_input_tokens=max(1200, SAFE_PROMPT_TOKENS.get(endpoint.model, 4000) - 250),
            metadata_factory=lambda chunk: {"sources": ",".join(sorted({hypothesis.source for hypothesis in chunk}))},
        )
        save_chunk_manifests("consolidator", all_hypotheses[0].brief_version if all_hypotheses else "v_unknown", [manifest for manifest, _ in chunks])
        partial_clusters = []
        for _manifest, chunk in chunks:
            partial = await _cluster_with_fallback(chunk)
            partial_clusters.extend(partial.clusters)

        representative_ids = {cluster.representative_hypothesis_id for cluster in partial_clusters}
        representatives = [hypothesis for hypothesis in all_hypotheses if hypothesis.id in representative_ids]
        if not representatives:
            return ConsensusClusters(clusters=partial_clusters)
        final_clusters = await _cluster_with_fallback(representatives)
        merged = {cluster.cluster_id: cluster for cluster in partial_clusters}
        for cluster in final_clusters.clusters:
            merged[cluster.cluster_id] = cluster
        return ConsensusClusters(clusters=list(merged.values()))
