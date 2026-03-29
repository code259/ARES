from __future__ import annotations

import re
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import TypeVar
import json

from pydantic import BaseModel

from core.config import DATA_DIR
from core.schemas import ChunkManifest, ConsensusClusters, ExperimentSpec, Hypothesis, KilledIdea, PaperRecord, RankList, Review, RunManifest, StageState


T = TypeVar("T", bound=BaseModel)

BRIEFS = DATA_DIR / "briefs"
GRAVEYARD = DATA_DIR / "graveyard"
HYPOTHESES = DATA_DIR / "hypotheses"
REVIEWS = DATA_DIR / "reviews"
SPECS = DATA_DIR / "specs"
PAPERS = DATA_DIR / "papers"
CONSENSUS = DATA_DIR / "consensus"
RANKS = DATA_DIR / "ranks"
MANUAL_INPUTS = DATA_DIR / "manual_inputs"
CONTEXT = DATA_DIR / "context"
STATE = DATA_DIR / "state"
MANIFESTS = DATA_DIR / "manifests"
RUNS = DATA_DIR / "runs"

for directory in [BRIEFS, GRAVEYARD, HYPOTHESES, REVIEWS, SPECS, PAPERS, CONSENSUS, RANKS, MANUAL_INPUTS, CONTEXT, STATE, MANIFESTS, RUNS]:
    directory.mkdir(parents=True, exist_ok=True)


def _safe_slug(value: str, max_len: int = 64) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return slug[:max_len].strip("_") or "record"


def _write_model(path: Path, model: BaseModel) -> None:
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_models(directory: Path, model_cls: type[T]) -> list[T]:
    items: list[T] = []
    for path in sorted(directory.glob("*.json")):
        items.append(model_cls.model_validate_json(path.read_text(encoding="utf-8")))
    return items


def _dedupe_by_key(items: list[T], key_fn) -> list[T]:
    deduped: dict[tuple, T] = {}
    for item in items:
        deduped[key_fn(item)] = item
    return list(deduped.values())


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def paper_storage_key(paper: PaperRecord) -> str:
    return _safe_slug(paper.title)


def _run_manifest_path(brief_version: str) -> Path:
    return RUNS / f"{brief_version}.json"


def new_brief_version() -> str:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    pattern = re.compile(rf"^v{date}_(\d+)$")
    indices: list[int] = []
    for version in list_brief_versions():
        match = pattern.match(version)
        if match:
            indices.append(int(match.group(1)))
    next_index = (max(indices) + 1) if indices else 1
    return f"v{date}_{next_index:02d}"


def latest_brief_version() -> str | None:
    versions = list_brief_versions()
    return versions[-1] if versions else None


def list_brief_versions() -> list[str]:
    versions = {path.stem for path in BRIEFS.glob("v*.md")}
    versions.update(path.stem for path in RUNS.glob("v*.json"))
    return sorted(versions)


def list_run_versions() -> list[str]:
    return list_brief_versions()


def prior_run_versions(brief_version: str) -> list[str]:
    return [version for version in list_run_versions() if version < brief_version]


def create_run_manifest(
    brief_version: str,
    *,
    paper_ids: list[str] | None = None,
    parent_brief_version: str = "",
    status: str = "initialized",
    metadata: dict | None = None,
) -> RunManifest:
    manifest = RunManifest(
        brief_version=brief_version,
        status=status,
        parent_brief_version=parent_brief_version,
        bottleneck_hash=_hash_text(read_bottleneck()) if (CONTEXT / "bottleneck.txt").exists() else "",
        pipeline_hash=_hash_text(read_pipeline_description()) if (CONTEXT / "pipeline_description.txt").exists() else "",
        paper_ids=sorted(set(paper_ids or [])),
        metadata=metadata or {},
    )
    save_run_manifest(manifest)
    return manifest


def save_run_manifest(manifest: RunManifest) -> Path:
    manifest.updated_at = _now_iso()
    path = _run_manifest_path(manifest.brief_version)
    _write_model(path, manifest)
    return path


def load_run_manifest(brief_version: str, *, bootstrap: bool = True) -> RunManifest | None:
    path = _run_manifest_path(brief_version)
    if path.exists():
        return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
    if not bootstrap:
        return None
    if (BRIEFS / f"{brief_version}.md").exists():
        manifest = RunManifest(
            brief_version=brief_version,
            status="legacy_imported",
            bottleneck_hash=_hash_text(read_bottleneck()) if (CONTEXT / "bottleneck.txt").exists() else "",
            pipeline_hash=_hash_text(read_pipeline_description()) if (CONTEXT / "pipeline_description.txt").exists() else "",
            paper_ids=sorted(path.stem for path in PAPERS.glob("*.json")),
            metadata={"bootstrapped": True},
        )
        save_run_manifest(manifest)
        return manifest
    return None


def ensure_run_manifest(
    brief_version: str,
    *,
    parent_brief_version: str = "",
    paper_ids: list[str] | None = None,
    status: str = "initialized",
    metadata: dict | None = None,
) -> RunManifest:
    existing = load_run_manifest(brief_version)
    if existing is not None:
        changed = False
        if paper_ids:
            merged = sorted(set(existing.paper_ids).union(paper_ids))
            if merged != existing.paper_ids:
                existing.paper_ids = merged
                changed = True
        if parent_brief_version and not existing.parent_brief_version:
            existing.parent_brief_version = parent_brief_version
            changed = True
        if metadata:
            merged_meta = dict(existing.metadata)
            merged_meta.update(metadata)
            if merged_meta != existing.metadata:
                existing.metadata = merged_meta
                changed = True
        if changed:
            save_run_manifest(existing)
        return existing
    return create_run_manifest(
        brief_version,
        paper_ids=paper_ids,
        parent_brief_version=parent_brief_version,
        status=status,
        metadata=metadata,
    )


def mark_run_stage(
    brief_version: str,
    stage: str,
    *,
    status: str | None = None,
    manual_source: str | None = None,
) -> RunManifest:
    manifest = ensure_run_manifest(brief_version)
    if stage not in manifest.stages_completed:
        manifest.stages_completed.append(stage)
        manifest.stages_completed.sort()
    if manual_source and manual_source not in manifest.manual_sources_imported:
        manifest.manual_sources_imported.append(manual_source)
        manifest.manual_sources_imported.sort()
    if status:
        manifest.status = status
    save_run_manifest(manifest)
    return manifest


def attach_papers_to_run(brief_version: str, papers: list[PaperRecord]) -> RunManifest:
    return ensure_run_manifest(brief_version, paper_ids=[paper_storage_key(paper) for paper in papers])


def set_run_metadata(brief_version: str, **updates) -> RunManifest:
    manifest = ensure_run_manifest(brief_version)
    merged = dict(manifest.metadata)
    merged.update(updates)
    manifest.metadata = merged
    save_run_manifest(manifest)
    return manifest


def save_brief(version: str, content: str) -> Path:
    path = BRIEFS / f"{version}.md"
    path.write_text(content, encoding="utf-8")
    ensure_run_manifest(version)
    return path


def load_brief(version: str) -> str:
    path = BRIEFS / f"{version}.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def add_to_graveyard(killed: KilledIdea) -> None:
    version = killed.brief_version or "legacy"
    _write_model(GRAVEYARD / f"{version}__{killed.id}.json", killed)


def load_graveyard(brief_version: str | None = None) -> list[KilledIdea]:
    killed = _dedupe_by_key(_load_models(GRAVEYARD, KilledIdea), lambda item: (item.brief_version, item.id))
    if brief_version is None:
        return killed
    return [item for item in killed if item.brief_version == brief_version]


def graveyard_summary(brief_version: str | None = None) -> str:
    killed = load_graveyard(brief_version)
    if not killed:
        return "No ideas killed yet."
    return "\n".join(f"- {item.name}: {item.kill_reason}" for item in killed)


def save_hypothesis(hypothesis: Hypothesis) -> None:
    path = HYPOTHESES / f"{hypothesis.brief_version}__{hypothesis.id}.json"
    if not hypothesis.id.startswith("hyp_"):
        existing = sorted(HYPOTHESES.glob(f"*__{hypothesis.id}.json"))
        if existing:
            hypothesis.id = f"hyp_{hypothesis.id}_{hypothesis.brief_version}"
            path = HYPOTHESES / f"{hypothesis.brief_version}__{hypothesis.id}.json"
    _write_model(path, hypothesis)


def save_hypotheses(hypotheses: list[Hypothesis]) -> None:
    for hypothesis in hypotheses:
        save_hypothesis(hypothesis)


def load_hypotheses(brief_version: str | None = None) -> list[Hypothesis]:
    hypotheses = _dedupe_by_key(_load_models(HYPOTHESES, Hypothesis), lambda item: (item.brief_version, item.id))
    if brief_version is None:
        return hypotheses
    return [hypothesis for hypothesis in hypotheses if hypothesis.brief_version == brief_version]


def clone_hypotheses_to_run(hypotheses: list[Hypothesis], brief_version: str) -> list[Hypothesis]:
    cloned: list[Hypothesis] = []
    for hypothesis in hypotheses:
        copied = hypothesis.model_copy(deep=True)
        copied.brief_version = brief_version
        cloned.append(copied)
    return cloned


def save_review(review: Review, brief_version: str | None = None) -> None:
    version = brief_version or review.brief_version or "legacy"
    review.brief_version = version
    _write_model(REVIEWS / f"{version}__{review.hypothesis_id}.json", review)


def save_reviews(reviews: list[Review], brief_version: str | None = None) -> None:
    for review in reviews:
        save_review(review, brief_version)


def load_reviews(brief_version: str | None = None) -> list[Review]:
    reviews = _dedupe_by_key(_load_models(REVIEWS, Review), lambda item: (item.brief_version, item.hypothesis_id))
    if brief_version is None:
        return reviews
    target_ids = {hypothesis.id for hypothesis in load_hypotheses(brief_version)}
    filtered = [review for review in reviews if review.hypothesis_id in target_ids]
    return filtered


def save_paper(paper: PaperRecord) -> None:
    _write_model(PAPERS / f"{paper_storage_key(paper)}.json", paper)


def save_papers(papers: list[PaperRecord]) -> None:
    for paper in papers:
        save_paper(paper)


def load_papers(brief_version: str | None = None) -> list[PaperRecord]:
    papers = _load_models(PAPERS, PaperRecord)
    if brief_version is None:
        return papers
    manifest = load_run_manifest(brief_version)
    if manifest is None:
        return papers
    if not manifest.paper_ids:
        return papers
    allowed = set(manifest.paper_ids)
    return [paper for paper in papers if paper_storage_key(paper) in allowed]


def save_spec(spec: ExperimentSpec, brief_version: str | None = None) -> None:
    version = brief_version or spec.brief_version or "legacy"
    spec.brief_version = version
    _write_model(SPECS / f"{version}__{spec.hypothesis_id}.json", spec)


def load_specs(brief_version: str | None = None) -> list[ExperimentSpec]:
    specs = _dedupe_by_key(_load_models(SPECS, ExperimentSpec), lambda item: (item.brief_version, item.hypothesis_id))
    if brief_version is None:
        return specs
    target_ids = {hypothesis.id for hypothesis in load_hypotheses(brief_version)}
    return [spec for spec in specs if spec.hypothesis_id in target_ids]


def save_clusters(brief_version: str, clusters: ConsensusClusters) -> Path:
    path = CONSENSUS / f"{brief_version}_clusters.json"
    _write_model(path, clusters)
    return path


def load_clusters(brief_version: str) -> ConsensusClusters | None:
    path = CONSENSUS / f"{brief_version}_clusters.json"
    if not path.exists():
        return None
    return ConsensusClusters.model_validate_json(path.read_text(encoding="utf-8"))


def save_ranks(brief_version: str, ranks: RankList) -> Path:
    path = RANKS / f"{brief_version}_ranks.json"
    _write_model(path, ranks)
    return path


def load_ranks(brief_version: str) -> RankList | None:
    path = RANKS / f"{brief_version}_ranks.json"
    if not path.exists():
        return None
    return RankList.model_validate_json(path.read_text(encoding="utf-8"))


def read_bottleneck() -> str:
    return (CONTEXT / "bottleneck.txt").read_text(encoding="utf-8").strip()


def read_pipeline_description() -> str:
    return (CONTEXT / "pipeline_description.txt").read_text(encoding="utf-8").strip()


def save_stage_state(stage_state: StageState) -> Path:
    path = STATE / f"{stage_state.brief_version}_{stage_state.stage}.json"
    _write_model(path, stage_state)
    return path


def load_stage_state(stage: str, brief_version: str) -> StageState | None:
    path = STATE / f"{brief_version}_{stage}.json"
    if not path.exists():
        return None
    return StageState.model_validate_json(path.read_text(encoding="utf-8"))


def save_chunk_manifests(stage: str, brief_version: str, manifests: list[ChunkManifest]) -> Path:
    path = MANIFESTS / f"{brief_version}_{stage}.json"
    path.write_text(
        json.dumps([manifest.model_dump() for manifest in manifests], indent=2),
        encoding="utf-8",
    )
    return path


def load_chunk_manifests(stage: str, brief_version: str) -> list[ChunkManifest]:
    path = MANIFESTS / f"{brief_version}_{stage}.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [ChunkManifest.model_validate(item) for item in raw]
