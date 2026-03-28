from __future__ import annotations

from core.config import PROMPTS_DIR
from core.llm import llm_registry
from core.memory import save_spec
from core.schemas import ExperimentSpec, Hypothesis, Review


PROMPT = (PROMPTS_DIR / "spec_writer.txt").read_text(encoding="utf-8")


async def write_spec(
    *,
    hypothesis: Hypothesis,
    review: Review,
    pipeline_description: str,
) -> ExperimentSpec:
    result = await llm_registry.complete_structured(
        role="spec_writer",
        system_prompt=PROMPT,
        user_prompt=(
            f"HYPOTHESIS:\n{hypothesis.model_dump_json(indent=2)}\n\n"
            f"REVIEW:\n{review.model_dump_json(indent=2)}\n\n"
            f"PIPELINE CONTEXT:\n{pipeline_description}"
        ),
        response_model=ExperimentSpec,
        temperature=0.2,
    )
    save_spec(result)
    return result


async def write_specs_for_ranked(
    *,
    survivors: list[Hypothesis],
    reviews: list[Review],
    pipeline_description: str,
    top_n: int = 10,
) -> list[ExperimentSpec]:
    review_map = {review.hypothesis_id: review for review in reviews}
    specs: list[ExperimentSpec] = []
    for hypothesis in survivors[:top_n]:
        review = review_map.get(hypothesis.id)
        if not review:
            continue
        specs.append(
            await write_spec(
                hypothesis=hypothesis,
                review=review,
                pipeline_description=pipeline_description,
            ),
        )
    return specs
