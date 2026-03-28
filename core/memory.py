from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from core.config import DATA_DIR
from core.schemas import ConsensusClusters, ExperimentSpec, Hypothesis, KilledIdea, PaperRecord, RankList, Review


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

for directory in [BRIEFS, GRAVEYARD, HYPOTHESES, REVIEWS, SPECS, PAPERS, CONSENSUS, RANKS, MANUAL_INPUTS, CONTEXT]:
    directory.mkdir(parents=True, exist_ok=True)


def _safe_slug(value: str, max_len: int = 64) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip())
    return slug[:max_len].strip("_") or "record"


def _write_model(path: Path, model: BaseModel) -> None:
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def _load_models(directory: Path, model_cls: type[T]) -> list[T]:
    items: list[T] = []
    for path in sorted(directory.glob("*.json")):
        items.append(model_cls.model_validate_json(path.read_text(encoding="utf-8")))
    return items


def new_brief_version() -> str:
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    existing = sorted(BRIEFS.glob(f"v{date}_*.md"))
    return f"v{date}_{len(existing) + 1:02d}"


def latest_brief_version() -> str | None:
    versions = sorted(BRIEFS.glob("v*.md"))
    return versions[-1].stem if versions else None


def save_brief(version: str, content: str) -> Path:
    path = BRIEFS / f"{version}.md"
    path.write_text(content, encoding="utf-8")
    return path


def load_brief(version: str) -> str:
    path = BRIEFS / f"{version}.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def add_to_graveyard(killed: KilledIdea) -> None:
    _write_model(GRAVEYARD / f"{killed.id}.json", killed)


def load_graveyard() -> list[KilledIdea]:
    return _load_models(GRAVEYARD, KilledIdea)


def graveyard_summary() -> str:
    killed = load_graveyard()
    if not killed:
        return "No ideas killed yet."
    return "\n".join(f"- {item.name}: {item.kill_reason}" for item in killed)


def save_hypothesis(hypothesis: Hypothesis) -> None:
    _write_model(HYPOTHESES / f"{hypothesis.id}.json", hypothesis)


def save_hypotheses(hypotheses: list[Hypothesis]) -> None:
    for hypothesis in hypotheses:
        save_hypothesis(hypothesis)


def load_hypotheses(brief_version: str | None = None) -> list[Hypothesis]:
    hypotheses = _load_models(HYPOTHESES, Hypothesis)
    if brief_version is None:
        return hypotheses
    return [hypothesis for hypothesis in hypotheses if hypothesis.brief_version == brief_version]


def save_review(review: Review) -> None:
    _write_model(REVIEWS / f"{review.hypothesis_id}.json", review)


def save_reviews(reviews: list[Review]) -> None:
    for review in reviews:
        save_review(review)


def load_reviews() -> list[Review]:
    return _load_models(REVIEWS, Review)


def save_paper(paper: PaperRecord) -> None:
    _write_model(PAPERS / f"{_safe_slug(paper.title)}.json", paper)


def save_papers(papers: list[PaperRecord]) -> None:
    for paper in papers:
        save_paper(paper)


def load_papers() -> list[PaperRecord]:
    return _load_models(PAPERS, PaperRecord)


def save_spec(spec: ExperimentSpec) -> None:
    _write_model(SPECS / f"{spec.hypothesis_id}.json", spec)


def load_specs() -> list[ExperimentSpec]:
    return _load_models(SPECS, ExperimentSpec)


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
