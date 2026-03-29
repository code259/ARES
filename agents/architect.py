from __future__ import annotations

import json

from core.compaction import chunk_evidence_records, compact_evidence_records, serialize_evidence
from core.config import PROMPTS_DIR, SAFE_PROMPT_TOKENS, role_endpoint
from core.context_compaction import compact_context_text
from core.llm import OversizeRequestError, llm_registry
from core.memory import graveyard_summary, load_stage_state, save_chunk_manifests, save_hypotheses, save_stage_state
from core.schemas import Hypothesis, HypothesisList, PaperRecord, StageState


GROUNDED_PROMPT = (PROMPTS_DIR / "architect_grounded.txt").read_text(encoding="utf-8")
FREE_RANGE_PROMPT = (PROMPTS_DIR / "architect_free_range.txt").read_text(encoding="utf-8")


def _base_grounded_context(
    *,
    bottleneck: str,
    pipeline_description: str,
    method_families: str,
    brief_version: str,
) -> list[str]:
    return [
        "BOTTLENECK:\n",
        bottleneck,
        "\n\nPIPELINE:\n",
        pipeline_description,
        "\n\nMETHOD FAMILIES IDENTIFIED:\n",
        method_families,
        "\n\nGRAVEYARD — IDEAS ALREADY KILLED (do not regenerate these):\n",
        graveyard_summary(brief_version),
        "\n\nBRIEF VERSION: ",
        brief_version,
        "\n\nPAPER EVIDENCE:\n",
    ]


def _grounded_target_instruction(target_count: int) -> str:
    return (
        f"TARGET OUTPUT COUNT:\n"
        f"- Aim for exactly {target_count} distinct grounded hypotheses for this call.\n"
        f"- Prioritize attackable, high-signal ideas over breadth.\n"
        f"- Do not pad with weak or repetitive ideas.\n"
        f"- If fewer than {target_count} genuinely strong ideas exist in this chunk, return only the strong ones."
    )


def _free_range_target_instruction(target_count: int) -> str:
    return (
        f"TARGET OUTPUT COUNT:\n"
        f"- Aim for exactly {target_count} distinct free-range hypotheses.\n"
        f"- Favor non-obvious but plausible ideas.\n"
        f"- Do not pad with generic or low-quality suggestions.\n"
        f"- If fewer than {target_count} genuinely strong ideas exist, return only the strong ones."
    )


async def _run_grounded_payload(
    *,
    bottleneck: str,
    pipeline_description: str,
    evidence_json: str,
    method_families: str,
    brief_version: str,
    target_count: int,
) -> list[Hypothesis]:
    payload = await llm_registry.complete_structured(
        role="architect",
        system_prompt=GROUNDED_PROMPT,
        user_prompt=(
            f"BOTTLENECK:\n{bottleneck}\n\n"
            f"PIPELINE:\n{pipeline_description}\n\n"
            f"METHOD FAMILIES IDENTIFIED:\n{method_families}\n\n"
            f"GRAVEYARD — IDEAS ALREADY KILLED (do not regenerate these):\n{graveyard_summary(brief_version)}\n\n"
            f"BRIEF VERSION: {brief_version}\n\n"
            f"{_grounded_target_instruction(target_count)}\n\n"
            f"PAPER EVIDENCE:\n{evidence_json}"
        ),
        response_model=HypothesisList,
        temperature=0.4,
        max_tokens=2200,
    )
    for hypothesis in payload.hypotheses:
        hypothesis.source = "grounded"
        hypothesis.brief_version = brief_version
    return payload.hypotheses


def _split_chunk(records):
    midpoint = len(records) // 2
    if midpoint <= 0:
        return [records]
    return [records[:midpoint], records[midpoint:]]


async def _run_chunk_with_fallback(
    *,
    chunk_records,
    bottleneck: str,
    pipeline_description: str,
    method_families: str,
    brief_version: str,
    target_count: int,
) -> list[Hypothesis]:
    try:
        return await _run_grounded_payload(
            bottleneck=bottleneck,
            pipeline_description=pipeline_description,
            evidence_json=serialize_evidence(chunk_records),
            method_families=method_families,
            brief_version=brief_version,
            target_count=target_count,
        )
    except OversizeRequestError:
        if len(chunk_records) <= 1:
            raise
        collected: list[Hypothesis] = []
        for subchunk in _split_chunk(chunk_records):
            collected.extend(
                await _run_chunk_with_fallback(
                    chunk_records=subchunk,
                    bottleneck=bottleneck,
                    pipeline_description=pipeline_description,
                    method_families=method_families,
                    brief_version=brief_version,
                    target_count=max(2, min(3, len(subchunk))),
                ),
            )
        return collected


def _dedupe_hypotheses(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    by_key: dict[str, Hypothesis] = {}
    for hypothesis in hypotheses:
        key = f"{hypothesis.name.strip().lower()}::{hypothesis.method_family.strip().lower()}"
        existing = by_key.get(key)
        if existing is None or len(hypothesis.hypothesis) > len(existing.hypothesis):
            by_key[key] = hypothesis
    return list(by_key.values())


async def generate_grounded(
    *,
    bottleneck: str,
    pipeline_description: str,
    papers: list[PaperRecord],
    method_families: str,
    brief_version: str,
) -> list:
    compact_pipeline = compact_context_text(pipeline_description, max_chars=5000)
    compact_bottleneck = compact_context_text(bottleneck, max_chars=1800)
    evidence_records = compact_evidence_records(papers)
    evidence_json = serialize_evidence(evidence_records)
    all_hypotheses: list[Hypothesis] = []

    try:
        hypotheses = await _run_grounded_payload(
            bottleneck=compact_bottleneck,
            pipeline_description=compact_pipeline,
            evidence_json=evidence_json,
            method_families=method_families,
            brief_version=brief_version,
            target_count=8,
        )
        all_hypotheses.extend(hypotheses)
    except OversizeRequestError:
        endpoint = role_endpoint("architect")
        base_context = _base_grounded_context(
            bottleneck=compact_bottleneck,
            pipeline_description=compact_pipeline,
            method_families=method_families,
            brief_version=brief_version,
        )
        base_context = [GROUNDED_PROMPT, "\n\n"] + base_context
        safe_limit = SAFE_PROMPT_TOKENS.get(endpoint.model, 5500)
        chunk_plan = chunk_evidence_records(
            role="architect",
            brief_version=brief_version,
            model=endpoint.model,
            evidence_records=evidence_records,
            base_context_parts=base_context,
            max_completion_tokens=1500,
            safe_input_tokens=max(1500, safe_limit - 250),
        )
        manifests = [manifest for manifest, _ in chunk_plan]
        save_chunk_manifests("generate_grounded", brief_version, manifests)
        stage_state = load_stage_state("generate_grounded", brief_version) or StageState(
            stage="generate_grounded",
            brief_version=brief_version,
            metadata={"model": endpoint.model},
        )
        completed = set(stage_state.completed_units)
        failed = set(stage_state.failed_units)

        for manifest, chunk in chunk_plan:
            if manifest.chunk_id in completed:
                continue
            try:
                chunk_hypotheses = await _run_chunk_with_fallback(
                    chunk_records=chunk,
                    bottleneck=compact_bottleneck,
                    pipeline_description=compact_pipeline,
                    method_families=method_families,
                    brief_version=brief_version,
                    target_count=max(2, min(3, len(chunk))),
                )
                all_hypotheses.extend(chunk_hypotheses)
                completed.add(manifest.chunk_id)
                if manifest.chunk_id in failed:
                    failed.remove(manifest.chunk_id)
                stage_state.completed_units = sorted(completed)
                stage_state.failed_units = sorted(failed)
                stage_state.metadata["last_completed_chunk"] = manifest.chunk_id
                save_stage_state(stage_state)
            except Exception:
                failed.add(manifest.chunk_id)
                stage_state.failed_units = sorted(failed)
                save_stage_state(stage_state)
                raise

        stage_state.status = "completed"
        stage_state.completed_units = sorted(completed)
        stage_state.failed_units = sorted(failed)
        save_stage_state(stage_state)

    deduped = _dedupe_hypotheses(all_hypotheses)
    save_hypotheses(deduped)
    return deduped


async def generate_free_range(
    *,
    bottleneck: str,
    pipeline_description: str,
    brief_version: str,
) -> list:
    compact_pipeline = compact_context_text(pipeline_description, max_chars=4500)
    compact_bottleneck = compact_context_text(bottleneck, max_chars=1800)
    payload = await llm_registry.complete_structured(
        role="architect",
        system_prompt=FREE_RANGE_PROMPT,
        user_prompt=(
            f"BOTTLENECK:\n{compact_bottleneck}\n\n"
            f"PIPELINE (brief):\n{compact_pipeline}\n\n"
            f"GRAVEYARD — IDEAS ALREADY KILLED (do not regenerate these):\n{graveyard_summary(brief_version)}\n\n"
            f"BRIEF VERSION: {brief_version}\n\n"
            f"{_free_range_target_instruction(6)}"
        ),
        response_model=HypothesisList,
        temperature=0.7,
        max_tokens=2200,
    )
    for hypothesis in payload.hypotheses:
        hypothesis.source = "free_range"
        hypothesis.brief_version = brief_version
    save_hypotheses(payload.hypotheses)
    return payload.hypotheses
