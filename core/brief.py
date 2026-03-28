from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from core.config import OUTPUTS_DIR
from core.memory import graveyard_summary
from core.schemas import Hypothesis, PaperRecord


def _family_for_paper(paper: PaperRecord) -> str:
    text = f"{paper.method} {paper.core_idea} {paper.possible_transfer}".lower()
    if "surrogate" in text or "docking" in text:
        return "Docking Surrogates"
    if "multi-fidelity" in text or "multifidelity" in text:
        return "Multi-Fidelity Pipelines"
    if "few-shot" in text or "meta" in text or "transfer" in text:
        return "Low-Data ML"
    if "ligand" in text:
        return "Ligand-Only Models"
    if "graph" in text or "3d" in text or "equivariant" in text or "structure" in text:
        return "Structural DL"
    return "Other"


def compile_brief(
    *,
    bottleneck: str,
    pipeline: str,
    papers: list[PaperRecord],
    hypotheses: list[Hypothesis],
    brief_version: str,
) -> str:
    families: dict[str, list[PaperRecord]] = defaultdict(list)
    for paper in papers:
        families[_family_for_paper(paper)].append(paper)

    lines: list[str] = [
        f"BRIEF VERSION: {brief_version}",
        "",
        "SECTION 1: Problem definition",
        pipeline,
        "",
        "SECTION 2: Current bottleneck",
        bottleneck,
        "",
        "SECTION 3: Method families from literature",
    ]

    for family, family_papers in sorted(families.items()):
        lines.append(f"- {family}")
        for paper in family_papers:
            lines.append(f"  Title: {paper.title}")
            lines.append(f"  Core idea: {paper.core_idea}")
            lines.append(f"  Project relevance: {paper.relevance_to_project}")
            lines.append(f"  Possible transfer: {paper.possible_transfer}")

    lines.extend(
        [
            "",
            "SECTION 4: Hypotheses already in system",
        ],
    )

    if hypotheses:
        for hypothesis in hypotheses:
            lines.append(f"- {hypothesis.name}: {hypothesis.hypothesis}")
    else:
        lines.append("- None yet.")

    lines.extend(
        [
            "",
            "SECTION 5: Graveyard",
            graveyard_summary(),
            "",
            "SECTION 6: Open directions",
            "- Combinations across families not yet explored",
            "- Low-data methods that preserve structural signal",
            "- Distillation, surrogates, and cascades that remove docking from inference",
        ],
    )

    brief = "\n".join(lines).strip() + "\n"
    out = OUTPUTS_DIR / "context_briefs" / f"{brief_version}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(brief, encoding="utf-8")
    return brief
