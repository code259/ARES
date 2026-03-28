from __future__ import annotations

import argparse
import asyncio
from collections import Counter

from rich.console import Console

from agents.adversary import red_team
from agents.architect import generate_free_range, generate_grounded
from agents.consolidator import cluster_and_flag
from agents.enumerator import enumerate_variants
from agents.librarian import process_papers_batch
from agents.ranker import rank_hypotheses
from agents.scout import generate_queries
from agents.spec_writer import write_specs_for_ranked
from core.brief import compile_brief
from core.memory import (
    latest_brief_version,
    load_clusters,
    load_hypotheses,
    load_papers,
    load_ranks,
    load_reviews,
    new_brief_version,
    read_bottleneck,
    read_pipeline_description,
    save_brief,
    save_clusters,
    save_ranks,
)
from core.pdf_ingest import ingest_manual_pdfs
from core.retrieval import fetch_papers_for_queries
from core.schemas import ConsensusClusters


console = Console()
TARGET_SPEC_COUNT = 10


def summarize_method_families() -> str:
    families = Counter()
    for paper in load_papers():
        text = f"{paper.method} {paper.core_idea}".lower()
        if "surrogate" in text or "docking" in text:
            families["docking_surrogates"] += 1
        elif "multi-fidelity" in text or "multifidelity" in text:
            families["multi_fidelity"] += 1
        elif "few-shot" in text or "meta" in text or "transfer" in text:
            families["low_data_ml"] += 1
        elif "ligand" in text:
            families["ligand_only"] += 1
        elif "graph" in text or "3d" in text or "equivariant" in text or "structure" in text:
            families["structural_dl"] += 1
        else:
            families["other"] += 1
    if not families:
        return "- No paper families yet."
    return "\n".join(f"- {family}: {count} papers" for family, count in families.most_common())


async def stage_ingest(brief_version: str) -> None:
    bottleneck = read_bottleneck()
    pipeline = read_pipeline_description()
    brief = compile_brief(
        bottleneck=bottleneck,
        pipeline=pipeline,
        papers=load_papers(),
        hypotheses=load_hypotheses(brief_version),
        brief_version=brief_version,
    )
    save_brief(brief_version, brief)
    console.print(f"[green]Saved initial brief {brief_version}[/green]")


async def stage_retrieve(_brief_version: str) -> list[str]:
    bottleneck = read_bottleneck()
    pipeline = read_pipeline_description()
    queries = await generate_queries(bottleneck, pipeline)
    console.print(f"[green]Generated {len(queries)} retrieval queries[/green]")
    texts = await fetch_papers_for_queries(queries)
    console.print(f"[green]Retrieved {len(texts)} paper texts[/green]")
    return texts


async def stage_ingest_papers(_brief_version: str) -> list:
    bottleneck = read_bottleneck()
    texts = await stage_retrieve(_brief_version)
    papers = await process_papers_batch(texts, bottleneck)
    console.print(f"[green]Structured {len(papers)} papers[/green]")
    return papers


async def stage_manual_papers(_brief_version: str) -> list:
    papers = await ingest_manual_pdfs()
    console.print(f"[green]Ingested {len(papers)} manual PDFs[/green]")
    return papers


async def stage_generate(brief_version: str) -> list:
    bottleneck = read_bottleneck()
    pipeline = read_pipeline_description()
    papers = load_papers()
    if not papers:
        console.print("[yellow]No papers found locally. Running retrieval + librarian first.[/yellow]")
        papers = await stage_ingest_papers(brief_version)
    families = summarize_method_families()
    grounded = await generate_grounded(
        bottleneck=bottleneck,
        pipeline_description=pipeline,
        papers=papers,
        method_families=families,
        brief_version=brief_version,
    )
    free_range = await generate_free_range(
        bottleneck=bottleneck,
        pipeline_description=pipeline,
        brief_version=brief_version,
    )
    console.print(f"[green]Generated {len(grounded)} grounded and {len(free_range)} free-range hypotheses[/green]")
    return grounded + free_range


async def stage_manual_pause(brief_version: str) -> None:
    brief = compile_brief(
        bottleneck=read_bottleneck(),
        pipeline=read_pipeline_description(),
        papers=load_papers(),
        hypotheses=load_hypotheses(brief_version),
        brief_version=brief_version,
    )
    path = save_brief(brief_version, brief)
    console.print(f"[bold yellow]Manual step required.[/bold yellow]")
    console.print(f"Brief saved to: {path}")
    console.print("Import manual outputs from Claude, Gemini, and ChatGPT with `python -m core.manual_import ...`.")


async def stage_consolidate(brief_version: str) -> ConsensusClusters:
    hypotheses = load_hypotheses()
    clusters = await cluster_and_flag(hypotheses)
    save_clusters(brief_version, clusters)
    console.print(f"[green]Saved {len(clusters.clusters)} consensus clusters[/green]")
    return clusters


async def stage_enumerate(brief_version: str) -> list:
    hypotheses = load_hypotheses(brief_version)
    variants = await enumerate_variants(hypotheses)
    console.print(f"[green]Generated {len(variants)} variants[/green]")
    return variants


async def stage_redteam(_brief_version: str) -> tuple[list, list]:
    hypotheses = load_hypotheses()
    survivors, reviews = await red_team(
        hypotheses=hypotheses,
        bottleneck=read_bottleneck(),
    )
    console.print(f"[green]Red-team survivors: {len(survivors)} / {len(hypotheses)}[/green]")
    return survivors, reviews


async def stage_rank(brief_version: str):
    survivors, reviews = await stage_redteam(brief_version)
    clusters = load_clusters(brief_version) or ConsensusClusters(clusters=[])
    ranks = await rank_hypotheses(
        survivors=survivors,
        reviews=reviews,
        clusters=clusters,
    )
    save_ranks(brief_version, ranks)
    console.print(f"[green]Ranked {len(ranks.ranks)} hypotheses[/green]")
    return survivors, reviews, ranks


async def stage_spec(brief_version: str) -> list:
    pipeline = read_pipeline_description()
    rank_list = load_ranks(brief_version)
    if rank_list is None:
        survivors, reviews, rank_list = await stage_rank(brief_version)
    else:
        survivors = [hypothesis for hypothesis in load_hypotheses() if hypothesis.id in {rank.hypothesis_id for rank in rank_list.ranks}]
        reviews = [review for review in load_reviews() if review.hypothesis_id in {rank.hypothesis_id for rank in rank_list.ranks}]
    rank_order = {rank.hypothesis_id: rank.recommended_order for rank in rank_list.ranks}
    survivors.sort(key=lambda item: rank_order.get(item.id, 999))
    specs = await write_specs_for_ranked(
        survivors=survivors,
        reviews=reviews,
        pipeline_description=pipeline,
        top_n=TARGET_SPEC_COUNT,
    )
    console.print(f"[green]Wrote {len(specs)} specs[/green]")
    return specs


async def run_all() -> None:
    brief_version = new_brief_version()
    console.print(f"[bold]Starting pipeline for {brief_version}[/bold]")
    await stage_ingest(brief_version)
    await stage_generate(brief_version)
    await stage_manual_pause(brief_version)
    console.print("[yellow]Pause here for manual frontier-model passes, then continue with consolidate/spec stages.[/yellow]")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        default="all",
        choices=[
            "all",
            "ingest",
            "retrieve",
            "papers",
            "manual-papers",
            "generate",
            "manual-pause",
            "consolidate",
            "enumerate",
            "redteam",
            "rank",
            "spec",
        ],
    )
    parser.add_argument("--brief-version", default=None)
    args = parser.parse_args()

    brief_version = args.brief_version or latest_brief_version() or new_brief_version()
    mapping = {
        "all": run_all,
        "ingest": lambda: stage_ingest(brief_version),
        "retrieve": lambda: stage_retrieve(brief_version),
        "papers": lambda: stage_ingest_papers(brief_version),
        "manual-papers": lambda: stage_manual_papers(brief_version),
        "generate": lambda: stage_generate(brief_version),
        "manual-pause": lambda: stage_manual_pause(brief_version),
        "consolidate": lambda: stage_consolidate(brief_version),
        "enumerate": lambda: stage_enumerate(brief_version),
        "redteam": lambda: stage_redteam(brief_version),
        "rank": lambda: stage_rank(brief_version),
        "spec": lambda: stage_spec(brief_version),
    }
    asyncio.run(mapping[args.stage]())


if __name__ == "__main__":
    main()
