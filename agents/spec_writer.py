from __future__ import annotations

from core.config import PROMPTS_DIR
from core.context_compaction import compact_context_text
from core.llm import llm_registry
from core.memory import load_stage_state, save_spec, save_stage_state
from core.schemas import ExperimentSpec, Hypothesis, Review, StageState


PROMPT = (PROMPTS_DIR / "spec_writer.txt").read_text(encoding="utf-8")


async def write_spec(
    *,
    hypothesis: Hypothesis,
    review: Review,
    pipeline_description: str,
) -> ExperimentSpec:
    compact_pipeline = compact_context_text(pipeline_description, max_chars=4500)
    result = await llm_registry.complete_structured(
        role="spec_writer",
        system_prompt=PROMPT,
        user_prompt=(
            "TARGET OUTPUT SHAPE:\n"
            "- Return one complete experiment spec.\n"
            "- Keep each field concise but implementation-ready.\n"
            "- Use short paragraphs, not long essays.\n"
            "- Keep codex_instructions to 3-4 sentences.\n"
            "- Keep approach and training_plan compact and stepwise.\n\n"
            f"HYPOTHESIS:\n{hypothesis.model_dump_json()}\n\n"
            f"REVIEW:\n{review.model_dump_json()}\n\n"
            f"PIPELINE CONTEXT:\n{compact_pipeline}"
        ),
        response_model=ExperimentSpec,
        temperature=0.2,
        max_tokens=3200,
    )
    result.brief_version = hypothesis.brief_version
    save_spec(result, hypothesis.brief_version)
    return result


async def write_specs_for_ranked(
    *,
    survivors: list[Hypothesis],
    reviews: list[Review],
    pipeline_description: str,
    top_n: int = 10,
) -> list[ExperimentSpec]:
    review_map = {review.hypothesis_id: review for review in reviews}
    brief_version = survivors[0].brief_version if survivors else "v_unknown"
    stage_state = load_stage_state("spec_writer", brief_version) or StageState(
        stage="spec_writer",
        brief_version=brief_version,
        metadata={"top_n": str(top_n)},
    )
    completed = set(stage_state.completed_units)
    failed = set(stage_state.failed_units)
    specs: list[ExperimentSpec] = []
    for hypothesis in survivors[:top_n]:
        if hypothesis.id in completed:
            continue
        review = review_map.get(hypothesis.id)
        if not review:
            continue
        try:
            spec = await write_spec(
                hypothesis=hypothesis,
                review=review,
                pipeline_description=pipeline_description,
            )
            specs.append(spec)
            completed.add(hypothesis.id)
            if hypothesis.id in failed:
                failed.remove(hypothesis.id)
            stage_state.completed_units = sorted(completed)
            stage_state.failed_units = sorted(failed)
            stage_state.metadata["last_completed_hypothesis"] = hypothesis.id
            save_stage_state(stage_state)
        except Exception:
            failed.add(hypothesis.id)
            stage_state.failed_units = sorted(failed)
            save_stage_state(stage_state)
            raise
    stage_state.status = "completed"
    stage_state.completed_units = sorted(completed)
    stage_state.failed_units = sorted(failed)
    save_stage_state(stage_state)
    return specs
