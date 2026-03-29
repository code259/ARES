from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import re

from core.llm import llm_registry
from core.memory import latest_brief_version, list_brief_versions, mark_run_stage, save_hypotheses
from core.schemas import Hypothesis, HypothesisSource, ImportedHypothesis, ImportedHypothesisList


PARSE_PROMPT = """
You are a structured data extractor. The user will provide free-form research hypotheses.
Parse them into a JSON object with a single key "hypotheses", whose value is an array of
hypothesis objects matching the import schema below.

Rules:
- Use empty strings for missing string fields.
- Default risk_level to "medium" if not stated.
- Default novelty to "moderate" if not stated.
- Default paper_refs to [].
- Preserve technical meaning. Do not invent claims not implied by the text.
- If the source gives "core idea", map it into "hypothesis".
- If the source gives "minimal experiment" or "2-week prototype", map it into "minimal_prototype".
- If the source gives "biggest risk", use it to help infer "kill_criteria" if no explicit kill criterion is given.
- Return only JSON.
""".strip()


def _split_manual_hypotheses(raw_text: str) -> list[str]:
    text = raw_text.strip()
    pattern = re.compile(r"(?m)^(?P<num>\d+)\.\s")
    matches = list(pattern.finditer(text))
    if matches:
        chunks: list[str] = []
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
        if chunks:
            return chunks

    divider_chunks = [chunk.strip() for chunk in re.split(r"\n[═─-]{8,}\n", text) if chunk.strip()]
    if len(divider_chunks) > 1:
        return divider_chunks

    paragraph_chunks: list[str] = []
    current = []
    current_len = 0
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if current and current_len + len(block) > 5000:
            paragraph_chunks.append("\n\n".join(current))
            current = [block]
            current_len = len(block)
        else:
            current.append(block)
            current_len += len(block)
    if current:
        paragraph_chunks.append("\n\n".join(current))
    return paragraph_chunks or [text]


def _coerce_imported_hypothesis(
    imported: ImportedHypothesis,
    *,
    source: HypothesisSource,
    brief_version: str,
) -> Hypothesis:
    inferred_name = imported.name.strip()
    if not inferred_name:
        candidate = (imported.hypothesis or imported.why_it_should_work_here or "").strip()
        if candidate:
            first_line = candidate.splitlines()[0].strip()
            inferred_name = first_line[:120].strip(" -:.") or "Imported hypothesis"
        else:
            inferred_name = "Imported hypothesis"

    hypothesis_text = imported.hypothesis or imported.why_it_should_work_here or imported.name
    minimal_prototype = imported.minimal_prototype or imported.killer_experiment
    kill_criteria = imported.kill_criteria or imported.risk_level.value
    return Hypothesis(
        name=inferred_name,
        hypothesis=hypothesis_text,
        source=source.value,
        brief_version=brief_version,
        method_family=imported.method_family or "manual_import",
        how_it_replaces_or_reduces_docking=imported.how_it_replaces_or_reduces_docking or "",
        why_it_should_work_here=imported.why_it_should_work_here or "",
        data_requirements=imported.data_requirements or "",
        expected_speedup=imported.expected_speedup or "",
        risk_level=imported.risk_level,
        novelty=imported.novelty,
        minimal_prototype=minimal_prototype,
        killer_experiment=imported.killer_experiment or minimal_prototype,
        kill_criteria=kill_criteria,
        paper_refs=imported.paper_refs,
    )


async def import_manual_output(
    *,
    raw_text: str,
    source: HypothesisSource,
    brief_version: str | None = None,
) -> list:
    if brief_version is None:
        versions = list_brief_versions()
        if len(versions) > 1:
            raise RuntimeError(
                "Multiple brief versions exist. Please pass --brief-version to attach manual imports to the intended run.",
            )
    current_brief = brief_version or latest_brief_version() or "v_manual"
    chunks = _split_manual_hypotheses(raw_text)
    all_hypotheses: list[Hypothesis] = []
    for index, chunk in enumerate(chunks, start=1):
        payload = await llm_registry.complete_structured(
            role="manual_import",
            system_prompt=PARSE_PROMPT,
            user_prompt=(
                f"SOURCE: {source.value}\n"
                f"BRIEF_VERSION: {current_brief}\n"
                f"CHUNK_INDEX: {index}/{len(chunks)}\n\n"
                f"RAW TEXT:\n{chunk}"
            ),
            response_model=ImportedHypothesisList,
            temperature=0.1,
            max_tokens=1400,
        )
        all_hypotheses.extend(
            _coerce_imported_hypothesis(item, source=source, brief_version=current_brief)
            for item in payload.hypotheses
        )

    save_hypotheses(all_hypotheses)
    mark_run_stage(current_brief, "manual_import", manual_source=source.value)
    return all_hypotheses


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
