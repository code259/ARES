from __future__ import annotations

import json

from core.compaction import chunk_serialized_items
from core.config import PROMPTS_DIR, SAFE_PROMPT_TOKENS, role_endpoint
from core.llm import OversizeRequestError, llm_registry
from core.memory import save_chunk_manifests, save_hypotheses
from core.schemas import Hypothesis, PartialHypothesis, PartialHypothesisList


PROMPT = (PROMPTS_DIR / "enumerator.txt").read_text(encoding="utf-8")


def _coerce_variant(partial: PartialHypothesis, parent_lookup: dict[str, Hypothesis]) -> Hypothesis:
    source = partial.source
    parent = None
    if source and source in parent_lookup:
        parent = parent_lookup[source]
    elif partial.brief_version:
        parent = next((item for item in parent_lookup.values() if item.brief_version == partial.brief_version), None)
    else:
        parent = next(iter(parent_lookup.values())) if parent_lookup else None

    hypothesis_text = partial.hypothesis or partial.why_it_should_work_here or partial.name
    return Hypothesis(
        id=partial.id,
        name=partial.name or (parent.name + " variant" if parent else "Variant hypothesis"),
        hypothesis=hypothesis_text,
        source=partial.source or (f"{parent.name}_variant" if parent else "variant"),
        brief_version=partial.brief_version or (parent.brief_version if parent else ""),
        method_family=partial.method_family or (parent.method_family if parent else "variant"),
        how_it_replaces_or_reduces_docking=partial.how_it_replaces_or_reduces_docking or (parent.how_it_replaces_or_reduces_docking if parent else ""),
        why_it_should_work_here=partial.why_it_should_work_here or (parent.why_it_should_work_here if parent else ""),
        data_requirements=partial.data_requirements or (parent.data_requirements if parent else ""),
        expected_speedup=partial.expected_speedup or (parent.expected_speedup if parent else ""),
        risk_level=partial.risk_level or (parent.risk_level if parent else None),
        novelty=partial.novelty or (parent.novelty if parent else None),
        minimal_prototype=partial.minimal_prototype or partial.killer_experiment or (parent.minimal_prototype if parent else ""),
        killer_experiment=partial.killer_experiment or partial.minimal_prototype or (parent.killer_experiment if parent else ""),
        kill_criteria=partial.kill_criteria or (parent.kill_criteria if parent else ""),
        paper_refs=partial.paper_refs or (parent.paper_refs if parent else []),
    )


async def _enumerate_payload(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    payload = await llm_registry.complete_structured(
        role="enumerator",
        system_prompt=PROMPT,
        user_prompt=(
            "TARGET OUTPUT COUNT:\n"
            "- Aim for 2 strong variants per parent hypothesis in this call.\n"
            "- Prefer fewer variants if quality would otherwise drop.\n\n"
            f"HYPOTHESES:\n{json.dumps([hypothesis.model_dump() for hypothesis in hypotheses], separators=(',', ':'))}"
        ),
        response_model=PartialHypothesisList,
        temperature=0.5,
        max_tokens=1800,
    )
    parent_lookup = {hypothesis.id: hypothesis for hypothesis in hypotheses}
    return [_coerce_variant(item, parent_lookup) for item in payload.hypotheses]


def _split_hypothesis_chunk(chunk: list[Hypothesis]) -> list[list[Hypothesis]]:
    midpoint = len(chunk) // 2
    if midpoint <= 0:
        return [chunk]
    return [chunk[:midpoint], chunk[midpoint:]]


async def _enumerate_with_fallback(chunk: list[Hypothesis]) -> list[Hypothesis]:
    try:
        return await _enumerate_payload(chunk)
    except OversizeRequestError:
        if len(chunk) <= 1:
            raise
        variants: list[Hypothesis] = []
        for subchunk in _split_hypothesis_chunk(chunk):
            variants.extend(await _enumerate_with_fallback(subchunk))
        return variants


async def enumerate_variants(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    try:
        variants = await _enumerate_payload(hypotheses)
    except OversizeRequestError:
        endpoint = role_endpoint("enumerator")
        chunks = chunk_serialized_items(
            role="enumerator",
            brief_version=hypotheses[0].brief_version if hypotheses else "v_unknown",
            model=endpoint.model,
            items=hypotheses,
            serialize_item=lambda hypothesis: hypothesis.model_dump_json(),
            base_context_parts=[PROMPT, "\n\nTARGET OUTPUT COUNT:\n- Aim for 2 strong variants per parent hypothesis in this call.\n- Prefer fewer variants if quality would otherwise drop.\n\nHYPOTHESES:\n"],
            max_completion_tokens=1800,
            safe_input_tokens=max(1200, SAFE_PROMPT_TOKENS.get(endpoint.model, 4000) - 250),
        )
        save_chunk_manifests("enumerator", hypotheses[0].brief_version if hypotheses else "v_unknown", [manifest for manifest, _ in chunks])
        variants = []
        for _manifest, chunk in chunks:
            variants.extend(await _enumerate_with_fallback(chunk))

    for hypothesis in variants:
        if not hypothesis.source:
            hypothesis.source = "variant"
    save_hypotheses(variants)
    return variants
