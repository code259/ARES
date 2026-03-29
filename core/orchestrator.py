from __future__ import annotations

import argparse
import asyncio

from rich.console import Console

from agents.adversary import red_team
from agents.architect import generate_free_range, generate_grounded
from agents.consolidator import cluster_and_flag
from agents.enumerator import enumerate_variants
from agents.librarian import process_papers_batch
from agents.ranker import rank_hypotheses
from agents.scout import generate_queries
from agents.spec_writer import write_specs_for_ranked
from core.brief import compile_brief, write_manual_packet
from core.compaction import compact_evidence_records, summarize_method_families_from_evidence
from core.memory import (
    attach_papers_to_run,
    create_run_manifest,
    ensure_run_manifest,
    latest_brief_version,
    load_clusters,
    load_hypotheses,
    load_papers,
    load_ranks,
    load_reviews,
    paper_storage_key,
    mark_run_stage,
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


async def stage_ingest(brief_version: str) -> None:
    ensure_run_manifest(brief_version)
    bottleneck = read_bottleneck()
    pipeline = read_pipeline_description()
    brief = compile_brief(
        bottleneck=bottleneck,
        pipeline=pipeline,
        papers=load_papers(brief_version),
        hypotheses=load_hypotheses(brief_version),
        brief_version=brief_version,
    )
    save_brief(brief_version, brief)
    mark_run_stage(brief_version, "ingest")
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
    attach_papers_to_run(_brief_version, papers)
    mark_run_stage(_brief_version, "papers")
    console.print(f"[green]Structured {len(papers)} papers[/green]")
    return papers


async def stage_manual_papers(_brief_version: str) -> list:
    papers = await ingest_manual_pdfs()
    attach_papers_to_run(_brief_version, papers)
    mark_run_stage(_brief_version, "manual-papers")
    console.print(f"[green]Ingested {len(papers)} manual PDFs[/green]")
    return papers


async def stage_generate(brief_version: str) -> list:
    bottleneck = read_bottleneck()
    pipeline = read_pipeline_description()
    manifest = ensure_run_manifest(brief_version)
    papers = load_papers(brief_version)
    should_expand_retrieval = bool(manifest.metadata.get("force_retrieval_before_generate")) and "papers" not in manifest.stages_completed
    if should_expand_retrieval:
        console.print("[yellow]Run is configured for retrieval expansion. Fetching additional literature before grounded generation.[/yellow]")
        await stage_ingest_papers(brief_version)
        papers = load_papers(brief_version)
        manifest = ensure_run_manifest(brief_version)
    elif not papers:
        console.print("[yellow]No papers found locally. Running retrieval + librarian first.[/yellow]")
        papers = await stage_ingest_papers(brief_version)
    if not papers:
        raise RuntimeError(f"No papers available for run {brief_version} after retrieval/ingestion.")
    evidence_records = compact_evidence_records(papers)
    families = summarize_method_families_from_evidence(evidence_records)
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
    mark_run_stage(brief_version, "generate", status="generated")
    console.print(f"[green]Generated {len(grounded)} grounded and {len(free_range)} free-range hypotheses[/green]")
    return grounded + free_range


async def stage_manual_pause(brief_version: str) -> None:
    bottleneck = read_bottleneck()
    pipeline = read_pipeline_description()
    papers = load_papers(brief_version)
    hypotheses = load_hypotheses(brief_version)
    brief = compile_brief(
        bottleneck=bottleneck,
        pipeline=pipeline,
        papers=papers,
        hypotheses=hypotheses,
        brief_version=brief_version,
    )
    path = save_brief(brief_version, brief)
    packet_dir = write_manual_packet(
        bottleneck=bottleneck,
        pipeline=pipeline,
        papers=papers,
        hypotheses=hypotheses,
        brief_version=brief_version,
    )
    mark_run_stage(brief_version, "manual-pause", status="manual_pause")
    console.print(f"[bold yellow]Manual step required.[/bold yellow]")
    console.print(f"Brief saved to: {path}")
    console.print(f"Manual packet saved to: {packet_dir}")
    console.print("Use the staged grounded prompts in that packet for Claude, Gemini, and ChatGPT, then import those grounded outputs with `python -m core.manual_import ...`.")


async def stage_consolidate(brief_version: str) -> ConsensusClusters:
    hypotheses = load_hypotheses(brief_version)
    clusters = await cluster_and_flag(hypotheses)
    save_clusters(brief_version, clusters)
    mark_run_stage(brief_version, "consolidate")
    console.print(f"[green]Saved {len(clusters.clusters)} consensus clusters[/green]")
    return clusters


async def stage_enumerate(brief_version: str) -> list:
    hypotheses = load_hypotheses(brief_version)
    variants = await enumerate_variants(hypotheses)
    mark_run_stage(brief_version, "enumerate")
    console.print(f"[green]Generated {len(variants)} variants[/green]")
    return variants


async def stage_redteam(brief_version: str) -> tuple[list, list]:
    hypotheses = load_hypotheses(brief_version)
    survivors, reviews = await red_team(
        hypotheses=hypotheses,
        bottleneck=read_bottleneck(),
    )
    mark_run_stage(brief_version, "redteam")
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
    mark_run_stage(brief_version, "rank")
    console.print(f"[green]Ranked {len(ranks.ranks)} hypotheses[/green]")
    return survivors, reviews, ranks


async def stage_spec(brief_version: str) -> list:
    pipeline = read_pipeline_description()
    rank_list = load_ranks(brief_version)
    if rank_list is None:
        survivors, reviews, rank_list = await stage_rank(brief_version)
    else:
        survivors = [hypothesis for hypothesis in load_hypotheses(brief_version) if hypothesis.id in {rank.hypothesis_id for rank in rank_list.ranks}]
        reviews = [review for review in load_reviews(brief_version) if review.hypothesis_id in {rank.hypothesis_id for rank in rank_list.ranks}]
    rank_order = {rank.hypothesis_id: rank.recommended_order for rank in rank_list.ranks}
    survivors.sort(key=lambda item: rank_order.get(item.id, 999))
    specs = await write_specs_for_ranked(
        survivors=survivors,
        reviews=reviews,
        pipeline_description=pipeline,
        top_n=TARGET_SPEC_COUNT,
    )
    mark_run_stage(brief_version, "spec", status="completed")
    console.print(f"[green]Wrote {len(specs)} specs[/green]")
    return specs


async def stage_new_run() -> str:
    parent = latest_brief_version() or ""
    brief_version = new_brief_version()
    source_papers = load_papers(parent) if parent else load_papers()
    manifest = create_run_manifest(
        brief_version,
        parent_brief_version=parent,
        paper_ids=[paper_storage_key(paper) for paper in source_papers],
        status="initialized",
        metadata={
            "created_via": "new-run",
            "force_retrieval_before_generate": True,
            "paper_selection_mode": "inherit_and_expand",
        },
    )
    await stage_ingest(brief_version)
    console.print(f"[green]Created run {manifest.brief_version}[/green]")
    if parent:
        console.print(f"[green]Parent run: {parent}[/green]")
    console.print(f"[green]Attached {len(manifest.paper_ids)} starting papers to the new run[/green]")
    console.print("[green]This run will expand literature via retrieval before grounded generation.[/green]")
    return manifest.brief_version


async def run_all() -> None:
    brief_version = await stage_new_run()
    console.print(f"[bold]Starting pipeline for {brief_version}[/bold]")
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
            "new-run",
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
        "new-run": stage_new_run,
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
