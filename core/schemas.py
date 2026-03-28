from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

import ulid
from pydantic import BaseModel, Field


def new_hypothesis_id() -> str:
    return f"hyp_{ulid.new()}"


class InferenceCost(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NoveltyLevel(str, Enum):
    INCREMENTAL = "incremental"
    MODERATE = "moderate"
    HIGH = "high"


class HypothesisSource(str, Enum):
    GROUNDED = "grounded"
    FREE_RANGE = "free_range"
    MANUAL_CLAUDE = "manual_claude"
    MANUAL_GEMINI = "manual_gemini"
    MANUAL_GPT4 = "manual_gpt4"
    VARIANT = "variant"


class Verdict(str, Enum):
    KILL = "kill"
    REVISE = "revise"
    PROCEED = "proceed"


class KillStage(str, Enum):
    RED_TEAM = "red_team"
    MANUAL_REVIEW = "manual_review"
    CONSOLIDATION = "consolidation"


class PaperRecord(BaseModel):
    title: str
    problem: str
    method: str
    inputs: str
    outputs: str
    training_data: str
    inference_cost: InferenceCost
    core_idea: str
    relevance_to_project: str
    possible_transfer: str
    failure_modes: str
    citations: list[str] = Field(default_factory=list)


class Hypothesis(BaseModel):
    id: str = Field(default_factory=new_hypothesis_id)
    name: str
    hypothesis: str
    source: str
    brief_version: str
    method_family: str
    how_it_replaces_or_reduces_docking: str
    why_it_should_work_here: str
    data_requirements: str
    expected_speedup: str
    risk_level: RiskLevel
    novelty: NoveltyLevel
    minimal_prototype: str
    killer_experiment: str
    kill_criteria: str
    paper_refs: list[str] = Field(default_factory=list)


class KilledIdea(BaseModel):
    id: str
    name: str
    hypothesis: str
    source: str
    brief_version: str
    kill_reason: str
    fatal_flaws: list[str] = Field(default_factory=list)
    killed_at_stage: KillStage
    date: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )


class Review(BaseModel):
    hypothesis_id: str
    fatal_flaws: list[str] = Field(default_factory=list)
    hidden_assumptions: list[str] = Field(default_factory=list)
    data_mismatch: list[str] = Field(default_factory=list)
    benchmark_risks: list[str] = Field(default_factory=list)
    complexity_vs_gain: str
    verdict: Verdict
    revision_direction: str = ""


class RankRecord(BaseModel):
    hypothesis_id: str
    feasibility_score: float = Field(ge=1, le=10)
    novelty_score: float = Field(ge=1, le=10)
    speedup_potential_score: float = Field(ge=1, le=10)
    data_risk_score: float = Field(ge=1, le=10)
    composite_score: float = Field(ge=1, le=10)
    recommended_order: int
    rationale: str
    consensus_flag: bool = False


class ExperimentSpec(BaseModel):
    hypothesis_id: str
    goal: str
    approach: str
    model_changes: str
    data_pipeline: str
    training_plan: str
    evaluation_metrics: str
    baseline: str
    expected_outcome: str
    failure_modes: str
    time_estimate: str
    branch_name: str
    codex_instructions: str


class ConsensusCluster(BaseModel):
    cluster_id: str
    method_family: str
    representative_hypothesis_id: str
    member_ids: list[str] = Field(default_factory=list)
    sources_represented: list[str] = Field(default_factory=list)
    consensus_flag: bool
    consensus_rationale: str


class ScoutOutput(BaseModel):
    queries: list[str] = Field(default_factory=list)


class HypothesisList(BaseModel):
    hypotheses: list[Hypothesis] = Field(default_factory=list)


class ReviewList(BaseModel):
    reviews: list[Review] = Field(default_factory=list)


class RankList(BaseModel):
    ranks: list[RankRecord] = Field(default_factory=list)


class SpecList(BaseModel):
    specs: list[ExperimentSpec] = Field(default_factory=list)


class ConsensusClusters(BaseModel):
    clusters: list[ConsensusCluster] = Field(default_factory=list)
