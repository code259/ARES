from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from core.compaction import infer_method_family
from core.config import OUTPUTS_DIR, PROMPTS_DIR
from core.memory import graveyard_summary
from core.schemas import Hypothesis, PaperRecord

MAX_PAPERS_PER_FAMILY_IN_BRIEF = 4
MAX_HYPOTHESES_IN_BRIEF = 20
MANUAL_PACKET_DIR = OUTPUTS_DIR / "manual_packets"


def _family_for_paper(paper: PaperRecord) -> str:
    return infer_method_family(paper).replace("_", " ").title()


def _truncate(text: str, limit: int = 220) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3].rstrip() + "..."


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
        lines.append(f"- {family} ({len(family_papers)} papers)")
        shown_papers = family_papers[:MAX_PAPERS_PER_FAMILY_IN_BRIEF]
        for paper in shown_papers:
            lines.append(f"  Title: {_truncate(paper.title, 140)}")
            lines.append(f"  Core idea: {_truncate(paper.core_idea)}")
            lines.append(f"  Project relevance: {_truncate(paper.relevance_to_project)}")
            lines.append(f"  Possible transfer: {_truncate(paper.possible_transfer)}")
        remaining = len(family_papers) - len(shown_papers)
        if remaining > 0:
            lines.append(f"  ... {remaining} more papers in this family omitted for brevity.")

    lines.extend(
        [
            "",
            "SECTION 4: Hypotheses already in system",
        ],
    )

    if hypotheses:
        shown_hypotheses = hypotheses[:MAX_HYPOTHESES_IN_BRIEF]
        for hypothesis in shown_hypotheses:
            lines.append(f"- {_truncate(hypothesis.name, 100)}: {_truncate(hypothesis.hypothesis, 220)}")
        remaining = len(hypotheses) - len(shown_hypotheses)
        if remaining > 0:
            lines.append(f"- ... {remaining} more hypotheses omitted for brevity.")
    else:
        lines.append("- None yet.")

    lines.extend(
        [
            "",
            "SECTION 5: Graveyard",
            graveyard_summary(brief_version),
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


def compile_manual_problem_context(*, bottleneck: str, pipeline: str, brief_version: str) -> str:
    return (
        f"BRIEF VERSION: {brief_version}\n\n"
        "SECTION 1: Full pipeline context\n"
        f"{pipeline.strip()}\n\n"
        "SECTION 2: Concrete bottleneck to solve\n"
        f"{bottleneck.strip()}\n"
    )


def compile_manual_literature_digest(*, papers: list[PaperRecord], brief_version: str) -> str:
    families: dict[str, list[PaperRecord]] = defaultdict(list)
    for paper in papers:
        families[_family_for_paper(paper)].append(paper)

    lines = [
        f"BRIEF VERSION: {brief_version}",
        "",
        f"LITERATURE DIGEST: {len(papers)} papers grouped by method family",
        "",
    ]
    for family, family_papers in sorted(families.items()):
        lines.append(f"## {family} ({len(family_papers)} papers)")
        for paper in family_papers[:8]:
            lines.append(f"- Title: {_truncate(paper.title, 160)}")
            lines.append(f"  Core idea: {_truncate(paper.core_idea, 320)}")
            lines.append(f"  Relevance: {_truncate(paper.relevance_to_project, 260)}")
            lines.append(f"  Transfer path: {_truncate(paper.possible_transfer, 260)}")
            lines.append(f"  Failure modes: {_truncate(paper.failure_modes, 220)}")
        remaining = len(family_papers) - min(len(family_papers), 8)
        if remaining > 0:
            lines.append(f"- ... {remaining} additional papers in this family not expanded here.")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def compile_manual_hypothesis_inventory(*, hypotheses: list[Hypothesis], brief_version: str) -> str:
    lines = [
        f"BRIEF VERSION: {brief_version}",
        "",
        f"CURRENT HYPOTHESIS INVENTORY: {len(hypotheses)} hypotheses already in-system",
        "",
    ]
    if not hypotheses:
        lines.append("- None yet.")
    else:
        for hypothesis in hypotheses:
            lines.append(f"- [{hypothesis.source}] {_truncate(hypothesis.name, 100)}")
            lines.append(f"  Core idea: {_truncate(hypothesis.hypothesis, 260)}")
            lines.append(f"  Method family: {_truncate(hypothesis.method_family, 80)}")
            lines.append(f"  Why it might work: {_truncate(hypothesis.why_it_should_work_here, 220)}")
    return "\n".join(lines).strip() + "\n"


def compile_manual_graveyard(*, brief_version: str) -> str:
    return (
        f"BRIEF VERSION: {brief_version}\n\n"
        "GRAVEYARD: ideas already killed or deprioritized\n"
        f"{graveyard_summary(brief_version)}\n"
    )


def write_manual_packet(
    *,
    bottleneck: str,
    pipeline: str,
    papers: list[PaperRecord],
    hypotheses: list[Hypothesis],
    brief_version: str,
) -> Path:
    packet_dir = MANUAL_PACKET_DIR / brief_version
    packet_dir.mkdir(parents=True, exist_ok=True)

    problem_context = compile_manual_problem_context(
        bottleneck=bottleneck,
        pipeline=pipeline,
        brief_version=brief_version,
    )
    literature_digest = compile_manual_literature_digest(
        papers=papers,
        brief_version=brief_version,
    )
    hypothesis_inventory = compile_manual_hypothesis_inventory(
        hypotheses=hypotheses,
        brief_version=brief_version,
    )
    graveyard = compile_manual_graveyard(brief_version=brief_version)

    (packet_dir / "problem_context.md").write_text(problem_context, encoding="utf-8")
    (packet_dir / "literature_digest.md").write_text(literature_digest, encoding="utf-8")
    (packet_dir / "hypothesis_inventory.md").write_text(hypothesis_inventory, encoding="utf-8")
    (packet_dir / "graveyard.md").write_text(graveyard, encoding="utf-8")

    for prompt_name in [
        "manual_grounded_stage1_context.txt",
        "manual_grounded_stage2_brainstorm.txt",
        "manual_grounded_stage3_finalize.txt",
    ]:
        prompt_text = (PROMPTS_DIR / prompt_name).read_text(encoding="utf-8")
        (packet_dir / prompt_name).write_text(prompt_text, encoding="utf-8")

    readme = (
        f"MANUAL GROUNDED WORKFLOW FOR {brief_version}\n\n"
        "Use one top-tier model at a time (Claude, Gemini, ChatGPT).\n\n"
        "Step 1:\n"
        "- Paste `manual_grounded_stage1_context.txt`\n"
        "- Then paste, in order: `problem_context.md`, `literature_digest.md`, `hypothesis_inventory.md`, `graveyard.md`\n\n"
        "Step 2:\n"
        "- Paste `manual_grounded_stage2_brainstorm.txt`\n"
        "- Let the model propose a broad candidate set only\n\n"
        "Step 3:\n"
        "- Paste `manual_grounded_stage3_finalize.txt`\n"
        "- Ask it to convert the best candidates into final grounded hypotheses in import-friendly numbered text\n\n"
        "Only run grounded manual passes. Manual free-range passes are no longer required.\n"
    )
    (packet_dir / "README.md").write_text(readme, encoding="utf-8")
    return packet_dir
