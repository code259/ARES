from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from core.llm import llm_registry
from core.memory import latest_brief_version, save_hypotheses
from core.schemas import HypothesisList, HypothesisSource


PARSE_PROMPT = """
You are a structured data extractor. The user will provide free-form research hypotheses.
Parse them into a JSON object with a single key "hypotheses", whose value is an array of
hypothesis objects matching the target schema exactly.

Rules:
- Use empty strings for missing string fields.
- Default risk_level to "medium" if not stated.
- Default novelty to "moderate" if not stated.
- Default paper_refs to [].
- Preserve technical meaning. Do not invent claims not implied by the text.
- Return only JSON.
""".strip()


async def import_manual_output(
    *,
    raw_text: str,
    source: HypothesisSource,
    brief_version: str | None = None,
) -> list:
    current_brief = brief_version or latest_brief_version() or "v_manual"
    payload = await llm_registry.complete_structured(
        role="manual_import",
        system_prompt=PARSE_PROMPT,
        user_prompt=f"SOURCE: {source.value}\nBRIEF_VERSION: {current_brief}\n\nRAW TEXT:\n{raw_text}",
        response_model=HypothesisList,
        temperature=0.1,
    )
    for hypothesis in payload.hypotheses:
        hypothesis.source = source.value
        hypothesis.brief_version = current_brief
    save_hypotheses(payload.hypotheses)
    return payload.hypotheses


async def _main_async(args: argparse.Namespace) -> None:
    hypotheses = await import_manual_output(
        raw_text=Path(args.file).read_text(encoding="utf-8"),
        source=HypothesisSource(args.source),
        brief_version=args.brief_version,
    )
    print(f"Imported {len(hypotheses)} hypotheses from {args.source}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=[item.value for item in HypothesisSource if item != HypothesisSource.VARIANT])
    parser.add_argument("--file", required=True)
    parser.add_argument("--brief-version", default=None)
    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
