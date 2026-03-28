# Research System — Implementation Guide
### Step-by-step agent instructions for full build

---

## Framework Decision

Two viable approaches. Both are documented below. **Recommendation: PydanticAI.**

### Option A — PydanticAI (Recommended)

**Why:** PydanticAI was designed exactly for this use case — structured LLM outputs with schema enforcement, typed agent definitions, dependency injection for shared clients, and first-class support for multi-model setups. Every agent in this system produces a structured JSON output; PydanticAI makes that the default rather than a workaround.

- Schema validation is automatic — if the LLM returns malformed JSON, PydanticAI retries before the error reaches your orchestrator
- `Agent[OutputType]` generic enforces the return type at the Python type level
- `RunContext` / dependency injection handles shared state (Groq client, memory store, brief version) without global variables
- Works with Groq, OpenAI-compatible endpoints, and any model via `openai`-compatible adapter

**Use when:** You want correctness guarantees, type safety, and minimal boilerplate around structured outputs.

### Option B — LangGraph (Alternative)

**Why:** LangGraph gives you an explicit, inspectable DAG for the pipeline. Each stage is a node; edges define flow. Good if you want to visualize the pipeline, add conditional routing (e.g., "if red-team kills >80%, restart generation"), or checkpoint state for resumption after failure.

- `StateGraph` maps directly to the `INGEST → RETRIEVE → ... → SPEC` pipeline
- Built-in checkpointing via `MemorySaver` or `SqliteSaver` — if a run crashes at Adversary stage, resume from there
- Conditional edges handle the "if <3 ideas survive red-team, re-run Architect" logic cleanly
- Heavier dependency footprint, more boilerplate per node

**Use when:** You want pipeline visualization, conditional re-runs, or crash recovery mid-pipeline.

### Decision for this guide

This guide implements **PydanticAI** as the primary framework with notes throughout on how the LangGraph equivalent would differ. A minimal LangGraph scaffold is provided at the end as Appendix B.

---

## Prerequisites

```
Python 3.11+
uv (package manager — faster than pip for this project)
```

---

## Step 1 — Project Initialization

```bash
# Create project
mkdir research_system && cd research_system
uv init
uv venv
source .venv/bin/activate

# Core dependencies
uv add pydantic-ai
uv add groq
uv add openai          # for gpt-oss-120b via OpenAI-compatible endpoint
uv add httpx           # for Semantic Scholar + arXiv API calls
uv add tenacity        # retry logic for API calls
uv add rich            # logging + progress display
uv add python-dotenv
uv add ulid-py         # for hypothesis IDs (sortable, unique)

# Optional: LangGraph alternative
# uv add langgraph
```

Create `.env`:

```bash
# Groq accounts (rotate across these)
GROQ_KEY_1=gsk_...
GROQ_KEY_2=gsk_...
GROQ_KEY_3=gsk_...

# OpenAI-compatible endpoint for gpt-oss-120b
OSS_BASE_URL=https://...
OSS_API_KEY=...

# Kimi K2 (OpenAI-compatible)
KIMI_BASE_URL=https://api.moonshot.cn/v1
KIMI_API_KEY=...
```

---

## Step 2 — Directory Structure

Create the full directory tree:

```bash
mkdir -p data/{papers,hypotheses,reviews,specs,graveyard,briefs,consensus}
mkdir -p agents core prompts outputs/{context_briefs,logs}
touch agents/__init__.py core/__init__.py
touch outputs/logs/token_usage.jsonl
```

Final layout:

```
research_system/
├── .env
├── pyproject.toml
├── data/
│   ├── papers/
│   ├── hypotheses/
│   ├── reviews/
│   ├── specs/
│   ├── graveyard/
│   ├── briefs/
│   └── consensus/
├── agents/
│   ├── __init__.py
│   ├── librarian.py
│   ├── architect.py
│   ├── adversary.py
│   ├── enumerator.py
│   ├── scout.py
│   ├── ranker.py
│   └── consolidator.py
├── core/
│   ├── __init__.py
│   ├── orchestrator.py
│   ├── memory.py
│   ├── groq_client.py
│   └── schemas.py
├── prompts/
│   ├── scout.txt
│   ├── librarian.txt
│   ├── architect_grounded.txt
│   ├── architect_free_range.txt
│   ├── enumerator.txt
│   ├── consolidator.txt
│   ├── adversary.txt
│   ├── ranker.txt
│   ├── spec_writer.txt
│   ├── manual_grounded.txt
│   └── manual_free_range.txt
└── outputs/
    ├── context_briefs/
    └── logs/
        └── token_usage.jsonl
```

---

## Step 3 — Schemas (`core/schemas.py`)

All Pydantic models. This file is the source of truth for every data type in the system. Every agent's output type references these models.

```python
# core/schemas.py
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field
import ulid


def new_id() -> str:
    return f"hyp_{ulid.new()}"


# ── Enums ────────────────────────────────────────────────────────────────────

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

class Verdict(str, Enum):
    KILL = "kill"
    REVISE = "revise"
    PROCEED = "proceed"

class KillStage(str, Enum):
    RED_TEAM = "red_team"
    MANUAL_REVIEW = "manual_review"
    CONSOLIDATION = "consolidation"


# ── Core Models ───────────────────────────────────────────────────────────────

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
    id: str = Field(default_factory=new_id)
    name: str
    hypothesis: str
    source: HypothesisSource
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
    source: HypothesisSource
    brief_version: str
    kill_reason: str
    fatal_flaws: list[str]
    killed_at_stage: KillStage
    date: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class Review(BaseModel):
    hypothesis_id: str
    fatal_flaws: list[str]
    hidden_assumptions: list[str]
    data_mismatch: list[str]
    benchmark_risks: list[str]
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


# ── Compound outputs used by agents ──────────────────────────────────────────

class ScoutOutput(BaseModel):
    queries: list[str]

class HypothesisList(BaseModel):
    hypotheses: list[Hypothesis]

class ReviewList(BaseModel):
    reviews: list[Review]

class RankList(BaseModel):
    ranks: list[RankRecord]

class SpecList(BaseModel):
    specs: list[ExperimentSpec]

class ConsensusClusters(BaseModel):
    class Cluster(BaseModel):
        cluster_id: str
        method_family: str
        representative_hypothesis_id: str
        member_ids: list[str]
        sources_represented: list[str]
        consensus_flag: bool
        consensus_rationale: str

    clusters: list[Cluster]
```

---

## Step 4 — Groq Client with Key Rotation (`core/groq_client.py`)

Implements round-robin rotation across accounts. Tracks per-account token usage. Fails over on rate limit errors.

```python
# core/groq_client.py
from __future__ import annotations
import json
import time
from pathlib import Path
from threading import Lock
from groq import Groq, RateLimitError
from dotenv import load_dotenv
import os

load_dotenv()

TOKEN_LOG = Path("outputs/logs/token_usage.jsonl")
TOKEN_LOG.parent.mkdir(parents=True, exist_ok=True)

_KEYS = [k for k in [
    os.getenv("GROQ_KEY_1"),
    os.getenv("GROQ_KEY_2"),
    os.getenv("GROQ_KEY_3"),
] if k]

assert _KEYS, "At least one GROQ_KEY_* must be set in .env"


class GroqPool:
    """
    Round-robin pool across Groq API keys.
    Tracks token usage per key. Fails over on RateLimitError.
    Daily budget: ~600k tokens across all keys.
    """

    def __init__(self, keys: list[str]):
        self._keys = keys
        self._clients = [Groq(api_key=k) for k in keys]
        self._index = 0
        self._usage: dict[str, int] = {k: 0 for k in keys}
        self._lock = Lock()

    def _next(self) -> tuple[Groq, str]:
        with self._lock:
            client = self._clients[self._index]
            key = self._keys[self._index]
            self._index = (self._index + 1) % len(self._clients)
        return client, key

    def _log_tokens(self, key: str, prompt_tokens: int, completion_tokens: int):
        self._usage[key] += prompt_tokens + completion_tokens
        entry = {
            "timestamp": time.time(),
            "key_suffix": key[-6:],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "running_total": self._usage[key],
        }
        with open(TOKEN_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        retries: int = 3,
    ) -> str:
        """
        Call Groq with automatic key rotation and rate-limit failover.
        Returns the content string of the first message choice.
        """
        attempted = set()
        last_error = None

        for attempt in range(retries * len(self._clients)):
            client, key = self._next()
            if key in attempted and len(attempted) >= len(self._clients):
                break
            attempted.add(key)
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                usage = resp.usage
                self._log_tokens(key, usage.prompt_tokens, usage.completion_tokens)
                return resp.choices[0].message.content
            except RateLimitError as e:
                last_error = e
                time.sleep(1)
                continue
            except Exception as e:
                raise e

        raise RuntimeError(f"All Groq keys exhausted or rate-limited. Last error: {last_error}")

    def total_tokens_used(self) -> int:
        return sum(self._usage.values())


# Singleton — import this everywhere
groq_pool = GroqPool(_KEYS)
```

---

## Step 5 — Memory Manager (`core/memory.py`)

Handles all persistent state: graveyard reads/writes, brief versioning, hypothesis storage, review storage.

```python
# core/memory.py
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from core.schemas import (
    Hypothesis, KilledIdea, Review, RankRecord,
    ExperimentSpec, PaperRecord, ConsensusClusters
)


DATA = Path("data")
BRIEFS = DATA / "briefs"
GRAVEYARD = DATA / "graveyard"
HYPOTHESES = DATA / "hypotheses"
REVIEWS = DATA / "reviews"
SPECS = DATA / "specs"
PAPERS = DATA / "papers"
CONSENSUS = DATA / "consensus"

for d in [BRIEFS, GRAVEYARD, HYPOTHESES, REVIEWS, SPECS, PAPERS, CONSENSUS]:
    d.mkdir(parents=True, exist_ok=True)


# ── Brief versioning ──────────────────────────────────────────────────────────

def new_brief_version() -> str:
    date = datetime.utcnow().strftime("%Y%m%d")
    existing = sorted(BRIEFS.glob(f"v{date}_*.json"))
    increment = len(existing) + 1
    return f"v{date}_{increment:02d}"


def save_brief(version: str, content: str) -> Path:
    path = BRIEFS / f"{version}.md"
    path.write_text(content)
    return path


def load_brief(version: str) -> str:
    path = BRIEFS / f"{version}.md"
    return path.read_text() if path.exists() else ""


def latest_brief_version() -> str | None:
    versions = sorted(BRIEFS.glob("v*.md"))
    return versions[-1].stem if versions else None


# ── Graveyard (failure memory) ────────────────────────────────────────────────

def add_to_graveyard(killed: KilledIdea) -> None:
    path = GRAVEYARD / f"{killed.id}.json"
    path.write_text(killed.model_dump_json(indent=2))


def load_graveyard() -> list[KilledIdea]:
    return [
        KilledIdea.model_validate_json(p.read_text())
        for p in GRAVEYARD.glob("*.json")
    ]


def graveyard_summary() -> str:
    """
    Returns a compact string for injection into Architect prompts.
    Format: one line per killed idea — name + primary kill reason.
    """
    killed = load_graveyard()
    if not killed:
        return "No ideas killed yet."
    lines = [f"- {k.name}: {k.kill_reason}" for k in killed]
    return "\n".join(lines)


# ── Hypothesis storage ────────────────────────────────────────────────────────

def save_hypothesis(h: Hypothesis) -> None:
    path = HYPOTHESES / f"{h.id}.json"
    path.write_text(h.model_dump_json(indent=2))


def save_hypotheses(hypotheses: list[Hypothesis]) -> None:
    for h in hypotheses:
        save_hypothesis(h)


def load_hypotheses(brief_version: str | None = None) -> list[Hypothesis]:
    all_h = [
        Hypothesis.model_validate_json(p.read_text())
        for p in HYPOTHESES.glob("*.json")
    ]
    if brief_version:
        return [h for h in all_h if h.brief_version == brief_version]
    return all_h


# ── Reviews ───────────────────────────────────────────────────────────────────

def save_review(r: Review) -> None:
    path = REVIEWS / f"{r.hypothesis_id}.json"
    path.write_text(r.model_dump_json(indent=2))


def load_reviews() -> list[Review]:
    return [
        Review.model_validate_json(p.read_text())
        for p in REVIEWS.glob("*.json")
    ]


# ── Papers ────────────────────────────────────────────────────────────────────

def save_paper(p: PaperRecord) -> None:
    safe_title = p.title[:60].replace("/", "_").replace(" ", "_")
    path = PAPERS / f"{safe_title}.json"
    path.write_text(p.model_dump_json(indent=2))


def load_papers() -> list[PaperRecord]:
    return [
        PaperRecord.model_validate_json(p.read_text())
        for p in PAPERS.glob("*.json")
    ]


# ── Specs ─────────────────────────────────────────────────────────────────────

def save_spec(s: ExperimentSpec) -> None:
    path = SPECS / f"{s.hypothesis_id}.json"
    path.write_text(s.model_dump_json(indent=2))


def load_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec.model_validate_json(p.read_text())
        for p in SPECS.glob("*.json")
    ]
```

---

## Step 6 — PydanticAI Agent Base Pattern

Before building individual agents, establish the base pattern. Every agent follows this structure.

```python
# Pattern: how every agent is built

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.groq import GroqModel
from core.schemas import SomeOutputType

# Define model
model = GroqModel("llama-3.3-70b-versatile")  # or OpenAIModel for gpt-oss

# Define agent with output type
agent: Agent[None, SomeOutputType] = Agent(
    model=model,
    result_type=SomeOutputType,
    system_prompt="Your system prompt here",
)

# Run agent
async def run_agent(user_input: str) -> SomeOutputType:
    result = await agent.run(user_input)
    return result.data  # .data is the validated Pydantic model
```

PydanticAI handles:
- JSON parsing from LLM output
- Validation against `SomeOutputType`
- Automatic retry on validation failure (configurable)

For agents that need shared state (groq pool, brief version), use `RunContext` with a dependency type.

---

## Step 7 — Scout Agent (`agents/scout.py`)

Generates 20–25 targeted search queries from bottleneck + pipeline description.

```python
# agents/scout.py
from __future__ import annotations
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from core.schemas import ScoutOutput
from pathlib import Path

PROMPT = Path("prompts/scout.txt").read_text()

# Use Groq compound model for speed
_model = GroqModel("llama-3.3-70b-versatile")

scout_agent: Agent[None, ScoutOutput] = Agent(
    model=_model,
    result_type=ScoutOutput,
    system_prompt=PROMPT,
)


async def generate_queries(
    bottleneck: str,
    pipeline_description: str,
) -> list[str]:
    user_msg = f"""
BOTTLENECK:
{bottleneck}

CURRENT SYSTEM:
{pipeline_description}
""".strip()

    result = await scout_agent.run(user_msg)
    return result.data.queries
```

Write `prompts/scout.txt` — copy verbatim from the spec's PROMPT 1.

---

## Step 8 — Librarian Agent (`agents/librarian.py`)

Converts raw paper text to a validated `PaperRecord`. Uses Kimi K2 via OpenAI-compatible endpoint.

```python
# agents/librarian.py
from __future__ import annotations
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from core.schemas import PaperRecord
from core.memory import save_paper
from pathlib import Path
import os

PROMPT = Path("prompts/librarian.txt").read_text()

_model = OpenAIModel(
    "kimi-k2",
    base_url=os.getenv("KIMI_BASE_URL"),
    api_key=os.getenv("KIMI_API_KEY"),
)

librarian_agent: Agent[None, PaperRecord] = Agent(
    model=_model,
    result_type=PaperRecord,
    system_prompt=PROMPT,
)


async def process_paper(paper_text: str, bottleneck: str) -> PaperRecord:
    user_msg = f"""
PAPER TEXT:
{paper_text}

PROJECT CONTEXT:
{bottleneck}
""".strip()

    result = await librarian_agent.run(user_msg)
    record = result.data
    save_paper(record)
    return record


async def process_papers_batch(
    papers: list[str],
    bottleneck: str,
) -> list[PaperRecord]:
    """Process multiple papers. Skips failures, logs them."""
    records = []
    for i, text in enumerate(papers):
        try:
            record = await process_paper(text, bottleneck)
            records.append(record)
            print(f"[Librarian] {i+1}/{len(papers)}: {record.title[:60]}")
        except Exception as e:
            print(f"[Librarian] Failed on paper {i+1}: {e}")
    return records
```

Write `prompts/librarian.txt` — copy verbatim from the spec's PROMPT 2.

---

## Step 9 — Architect Agent (`agents/architect.py`)

Two modes: grounded and free-range. Both use gpt-oss-120b via OpenAI-compatible endpoint. Graveyard summary is injected at call time.

```python
# agents/architect.py
from __future__ import annotations
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from core.schemas import HypothesisList, PaperRecord
from core.memory import graveyard_summary, save_hypotheses
from pathlib import Path
import json
import os

GROUNDED_PROMPT = Path("prompts/architect_grounded.txt").read_text()
FREE_RANGE_PROMPT = Path("prompts/architect_free_range.txt").read_text()

_model = OpenAIModel(
    "gpt-oss-120b",
    base_url=os.getenv("OSS_BASE_URL"),
    api_key=os.getenv("OSS_API_KEY"),
)

architect_grounded: Agent[None, HypothesisList] = Agent(
    model=_model,
    result_type=HypothesisList,
    system_prompt=GROUNDED_PROMPT,
)

architect_free_range: Agent[None, HypothesisList] = Agent(
    model=_model,
    result_type=HypothesisList,
    system_prompt=FREE_RANGE_PROMPT,
)


async def generate_grounded(
    bottleneck: str,
    pipeline_description: str,
    papers: list[PaperRecord],
    method_families: str,
    brief_version: str,
) -> list:
    graveyard = graveyard_summary()
    papers_json = json.dumps([p.model_dump() for p in papers], indent=2)

    user_msg = f"""
BOTTLENECK:
{bottleneck}

PIPELINE:
{pipeline_description}

PAPER RECORDS:
{papers_json}

METHOD FAMILIES IDENTIFIED:
{method_families}

GRAVEYARD — IDEAS ALREADY KILLED (do not regenerate these):
{graveyard}

BRIEF VERSION: {brief_version}
""".strip()

    result = await architect_grounded.run(user_msg)
    hypotheses = result.data.hypotheses
    # Stamp brief_version on each hypothesis
    for h in hypotheses:
        h.brief_version = brief_version
    save_hypotheses(hypotheses)
    return hypotheses


async def generate_free_range(
    bottleneck: str,
    pipeline_description: str,
    brief_version: str,
) -> list:
    graveyard = graveyard_summary()

    user_msg = f"""
BOTTLENECK:
{bottleneck}

PIPELINE (brief):
{pipeline_description}

GRAVEYARD — IDEAS ALREADY KILLED (do not regenerate these):
{graveyard}

BRIEF VERSION: {brief_version}
""".strip()

    result = await architect_free_range.run(user_msg)
    hypotheses = result.data.hypotheses
    for h in hypotheses:
        h.brief_version = brief_version
    save_hypotheses(hypotheses)
    return hypotheses
```

Write `prompts/architect_grounded.txt` — PROMPT 3 from spec.
Write `prompts/architect_free_range.txt` — PROMPT 4 from spec.

---

## Step 10 — Enumerator Agent (`agents/enumerator.py`)

Generates 2–3 variants per hypothesis. Uses Qwen3-32b via Groq.

```python
# agents/enumerator.py
from __future__ import annotations
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from core.schemas import HypothesisList, Hypothesis
from core.memory import save_hypotheses
from pathlib import Path
import json

PROMPT = Path("prompts/enumerator.txt").read_text()

_model = GroqModel("qwen-qwq-32b")  # adjust to exact Groq model string

enumerator_agent: Agent[None, HypothesisList] = Agent(
    model=_model,
    result_type=HypothesisList,
    system_prompt=PROMPT,
)


async def enumerate_variants(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    user_msg = f"""
HYPOTHESES:
{json.dumps([h.model_dump() for h in hypotheses], indent=2)}
""".strip()

    result = await enumerator_agent.run(user_msg)
    variants = result.data.hypotheses
    save_hypotheses(variants)
    return variants
```

Write `prompts/enumerator.txt` — PROMPT 5 from spec.

---

## Step 11 — Consolidator Agent (`agents/consolidator.py`)

Clusters all hypotheses from all passes. Flags consensus ideas (appeared in 2+ independent sources). This runs after manual LLM passes are imported.

```python
# agents/consolidator.py
from __future__ import annotations
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from core.schemas import ConsensusClusters, Hypothesis
from pathlib import Path
import json

PROMPT = Path("prompts/consolidator.txt").read_text()

_model = GroqModel("qwen-qwq-32b")

consolidator_agent: Agent[None, ConsensusClusters] = Agent(
    model=_model,
    result_type=ConsensusClusters,
    system_prompt=PROMPT,
)


async def cluster_and_flag(all_hypotheses: list[Hypothesis]) -> ConsensusClusters:
    user_msg = f"""
ALL HYPOTHESES (from all passes):
{json.dumps([h.model_dump() for h in all_hypotheses], indent=2)}
""".strip()

    result = await consolidator_agent.run(user_msg)
    return result.data
```

Write `prompts/consolidator.txt` — PROMPT 6 from spec.

---

## Step 12 — Adversary Agent (`agents/adversary.py`)

Red-teams all hypotheses. Target: kill 40%+. Writes kills to graveyard.

```python
# agents/adversary.py
from __future__ import annotations
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from core.schemas import ReviewList, Hypothesis, Review, KilledIdea, Verdict, KillStage
from core.memory import save_review, add_to_graveyard
from pathlib import Path
import json
import os

PROMPT = Path("prompts/adversary.txt").read_text()

_model = OpenAIModel(
    "gpt-oss-120b",
    base_url=os.getenv("OSS_BASE_URL"),
    api_key=os.getenv("OSS_API_KEY"),
)

adversary_agent: Agent[None, ReviewList] = Agent(
    model=_model,
    result_type=ReviewList,
    system_prompt=PROMPT,
)

# Hypothesis lookup by id
_hyp_map: dict[str, Hypothesis] = {}


async def red_team(
    hypotheses: list[Hypothesis],
    bottleneck: str,
) -> tuple[list[Hypothesis], list[Review]]:
    """
    Run adversarial review. Returns (survivors, all_reviews).
    Kills are written to graveyard automatically.
    """
    global _hyp_map
    _hyp_map = {h.id: h for h in hypotheses}

    user_msg = f"""
BOTTLENECK:
{bottleneck}

HYPOTHESES:
{json.dumps([h.model_dump() for h in hypotheses], indent=2)}
""".strip()

    result = await adversary_agent.run(user_msg)
    reviews = result.data.reviews

    survivors = []
    for review in reviews:
        save_review(review)
        hyp = _hyp_map.get(review.hypothesis_id)
        if not hyp:
            continue

        if review.verdict == Verdict.KILL:
            killed = KilledIdea(
                id=hyp.id,
                name=hyp.name,
                hypothesis=hyp.hypothesis,
                source=hyp.source,
                brief_version=hyp.brief_version,
                kill_reason=review.complexity_vs_gain,
                fatal_flaws=review.fatal_flaws,
                killed_at_stage=KillStage.RED_TEAM,
            )
            add_to_graveyard(killed)
        else:
            survivors.append(hyp)

    kill_count = len(hypotheses) - len(survivors)
    kill_rate = kill_count / len(hypotheses) * 100
    print(f"[Adversary] Killed {kill_count}/{len(hypotheses)} ({kill_rate:.0f}%)")

    return survivors, reviews
```

Write `prompts/adversary.txt` — PROMPT 7 from spec.

---

## Step 13 — Ranker Agent (`agents/ranker.py`)

Scores and orders surviving hypotheses. Applies +0.5 bonus for consensus-flagged ideas.

```python
# agents/ranker.py
from __future__ import annotations
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from core.schemas import RankList, Hypothesis, Review, ConsensusClusters
from pathlib import Path
import json

PROMPT = Path("prompts/ranker.txt").read_text()

_model = GroqModel("qwen-qwq-32b")

ranker_agent: Agent[None, RankList] = Agent(
    model=_model,
    result_type=RankList,
    system_prompt=PROMPT,
)


def _build_consensus_flags(
    hypothesis_ids: list[str],
    clusters: ConsensusClusters,
) -> dict[str, bool]:
    flags: dict[str, bool] = {hid: False for hid in hypothesis_ids}
    for cluster in clusters.clusters:
        if cluster.consensus_flag:
            for member_id in cluster.member_ids:
                if member_id in flags:
                    flags[member_id] = True
    return flags


async def rank_hypotheses(
    survivors: list[Hypothesis],
    reviews: list[Review],
    clusters: ConsensusClusters,
) -> RankList:
    consensus_flags = _build_consensus_flags([h.id for h in survivors], clusters)

    user_msg = f"""
SURVIVING HYPOTHESES:
{json.dumps([h.model_dump() for h in survivors], indent=2)}

REVIEWS:
{json.dumps([r.model_dump() for r in reviews], indent=2)}

CONSENSUS FLAGS:
{json.dumps(consensus_flags, indent=2)}
""".strip()

    result = await ranker_agent.run(user_msg)
    ranks = result.data.ranks

    # Apply consensus bonus post-hoc (ensures it's always applied regardless of LLM)
    for rank in ranks:
        if consensus_flags.get(rank.hypothesis_id):
            rank.composite_score = min(10.0, rank.composite_score + 0.5)
            rank.consensus_flag = True

    # Re-sort by composite_score descending
    ranks.sort(key=lambda r: r.composite_score, reverse=True)
    for i, rank in enumerate(ranks):
        rank.recommended_order = i + 1

    return result.data
```

Write `prompts/ranker.txt` — PROMPT 8 from spec.

---

## Step 14 — Spec Writer Agent (`agents/spec_writer.py`)

Produces one `ExperimentSpec` per surviving, ranked hypothesis. Includes `codex_instructions` for immediate Codex handoff.

```python
# agents/spec_writer.py
from __future__ import annotations
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from core.schemas import ExperimentSpec, Hypothesis, Review
from core.memory import save_spec
from pathlib import Path
import json
import os

PROMPT = Path("prompts/spec_writer.txt").read_text()

_model = OpenAIModel(
    "gpt-oss-120b",
    base_url=os.getenv("OSS_BASE_URL"),
    api_key=os.getenv("OSS_API_KEY"),
)

spec_agent: Agent[None, ExperimentSpec] = Agent(
    model=_model,
    result_type=ExperimentSpec,
    system_prompt=PROMPT,
)


async def write_spec(
    hypothesis: Hypothesis,
    review: Review,
    pipeline_description: str,
) -> ExperimentSpec:
    user_msg = f"""
HYPOTHESIS:
{hypothesis.model_dump_json(indent=2)}

REVIEW:
{review.model_dump_json(indent=2)}

PIPELINE CONTEXT:
{pipeline_description}
""".strip()

    result = await spec_agent.run(user_msg)
    spec = result.data
    save_spec(spec)
    return spec


async def write_specs_for_ranked(
    survivors: list[Hypothesis],
    reviews: list[Review],
    pipeline_description: str,
    top_n: int = 10,
) -> list[ExperimentSpec]:
    """Write specs for top N ranked survivors."""
    review_map = {r.hypothesis_id: r for r in reviews}
    specs = []
    for hyp in survivors[:top_n]:
        review = review_map.get(hyp.id)
        if not review:
            continue
        try:
            spec = await write_spec(hyp, review, pipeline_description)
            specs.append(spec)
            print(f"[SpecWriter] Wrote spec: {hyp.name}")
        except Exception as e:
            print(f"[SpecWriter] Failed on {hyp.name}: {e}")
    return specs
```

Write `prompts/spec_writer.txt` — PROMPT 9 from spec.

---

## Step 15 — Manual Hypothesis Importer (`core/manual_import.py`)

Parses free-form output from Claude/Gemini/ChatGPT into the `Hypothesis` schema. This is the bridge between the manual LLM passes and the automated pipeline.

```python
# core/manual_import.py
"""
After running manual LLM passes, paste the raw output into a text file under
data/manual_inputs/{source}_{datetime}.txt

Then call: python -m core.manual_import --source manual_claude --file data/manual_inputs/...txt

This runs a small PydanticAI agent to parse the free-form text into Hypothesis objects.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.models.groq import GroqModel
from core.schemas import HypothesisList, HypothesisSource
from core.memory import save_hypotheses, latest_brief_version

_model = GroqModel("llama-3.3-70b-versatile")

_PARSE_PROMPT = """
You are a structured data extractor. The user will give you free-form text containing 
research hypotheses written by a human or LLM. Parse each hypothesis into the schema.

Return a JSON object with a "hypotheses" array. Each item must match the Hypothesis schema exactly.
For fields you cannot determine from the text, use reasonable defaults:
- risk_level: "medium"
- novelty: "moderate"  
- paper_refs: []

Do not invent content. If a field is genuinely absent, use an empty string.
Return only the JSON. No markdown.
"""

parse_agent: Agent[None, HypothesisList] = Agent(
    model=_model,
    result_type=HypothesisList,
    system_prompt=_PARSE_PROMPT,
)


async def import_manual_output(
    raw_text: str,
    source: HypothesisSource,
    brief_version: str | None = None,
) -> list:
    bv = brief_version or latest_brief_version() or "v_manual"

    user_msg = f"""
SOURCE: {source.value}
BRIEF_VERSION: {bv}

RAW TEXT:
{raw_text}
"""
    result = await parse_agent.run(user_msg)
    hypotheses = result.data.hypotheses
    for h in hypotheses:
        h.source = source
        h.brief_version = bv
    save_hypotheses(hypotheses)
    print(f"[ManualImport] Imported {len(hypotheses)} hypotheses from {source.value}")
    return hypotheses


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        choices=["manual_claude", "manual_gemini", "manual_gpt4"])
    parser.add_argument("--file", required=True)
    parser.add_argument("--brief-version", default=None)
    args = parser.parse_args()

    raw = Path(args.file).read_text()
    source = HypothesisSource(args.source)
    asyncio.run(import_manual_output(raw, source, args.brief_version))
```

---

## Step 16 — Orchestrator (`core/orchestrator.py`)

The top-level controller. Runs all stages in order. Called directly from CLI.

```python
# core/orchestrator.py
"""
Main pipeline runner. Stages:
  1. Ingest context
  2. Paper expansion (queries → retrieve → librarian)
  3. Synthesis (method families)
  4. Dual hypothesis generation (grounded + free-range)
  5. [PAUSE] Manual LLM passes — you paste brief, import outputs
  6. Merge + dedup (consolidator)
  7. Enumeration
  8. Red-team (adversary)
  9. Rank
  10. Spec writing

Usage:
  python -m core.orchestrator --stage all
  python -m core.orchestrator --stage generate
  python -m core.orchestrator --stage redteam
  python -m core.orchestrator --stage spec
"""
from __future__ import annotations
import asyncio
import argparse
import json
from pathlib import Path
from rich.console import Console
from rich.progress import track

from core.memory import (
    new_brief_version, save_brief, load_papers, load_hypotheses,
    load_reviews, latest_brief_version
)
from core.brief import compile_brief
from agents.scout import generate_queries
from agents.librarian import process_papers_batch
from agents.architect import generate_grounded, generate_free_range
from agents.enumerator import enumerate_variants
from agents.consolidator import cluster_and_flag
from agents.adversary import red_team
from agents.ranker import rank_hypotheses
from agents.spec_writer import write_specs_for_ranked

console = Console()


# ── Config — edit these for your project ─────────────────────────────────────

BOTTLENECK = """
Current potency prediction model requires molecular docking at inference time.
Docking is compute-intensive and does not scale beyond ~200k molecules.
Environment is low-data (limited labeled examples).
Need to either eliminate docking from inference or dramatically reduce its cost
while preserving predictive performance.
""".strip()

PIPELINE_DESCRIPTION = """
[FILL IN: brief description of your current model architecture, 
what it takes as input, what it predicts, and where docking fits]
""".strip()

TARGET_SPEC_COUNT = 10


# ── Stage functions ───────────────────────────────────────────────────────────

async def stage_generate(brief_version: str):
    """Runs grounded + free-range Architect passes."""
    console.print(f"[bold green]Stage: Generate[/] brief_version={brief_version}")

    papers = load_papers()
    if not papers:
        console.print("[yellow]No papers loaded. Run paper ingestion first.[/]")

    # Build method families string from papers (simple summary)
    method_families = _summarize_method_families(papers)

    console.print(f"Generating grounded hypotheses from {len(papers)} papers...")
    grounded = await generate_grounded(
        BOTTLENECK, PIPELINE_DESCRIPTION, papers, method_families, brief_version
    )
    console.print(f"[green]Grounded: {len(grounded)} hypotheses[/]")

    console.print("Generating free-range hypotheses...")
    free_range = await generate_free_range(BOTTLENECK, PIPELINE_DESCRIPTION, brief_version)
    console.print(f"[green]Free-range: {len(free_range)} hypotheses[/]")

    total = len(grounded) + len(free_range)
    console.print(f"[bold]Total generated: {total}[/]")
    return grounded + free_range


async def stage_manual_pause(brief_version: str):
    """
    Generates the compact brief for manual LLM passes.
    Prints instructions. Pipeline pauses here until you import results.
    """
    console.print("[bold yellow]MANUAL STEP REQUIRED[/]")

    papers = load_papers()
    hypotheses = load_hypotheses(brief_version)
    brief_content = compile_brief(
        bottleneck=BOTTLENECK,
        pipeline=PIPELINE_DESCRIPTION,
        papers=papers,
        hypotheses=hypotheses,
        brief_version=brief_version,
    )
    brief_path = save_brief(brief_version, brief_content)
    console.print(f"Brief saved: {brief_path}")

    console.print("""
[bold]Manual LLM pass instructions:[/]
1. Open outputs/context_briefs/ and copy the brief
2. Paste into Claude with MANUAL PROMPT A (grounded)
3. Paste into Gemini with MANUAL PROMPT A (grounded)
4. Paste into ChatGPT with MANUAL PROMPT A (grounded)
5. Paste MANUAL PROMPT B (free-range) into all three without the brief
6. Save each output to data/manual_inputs/{source}_{datetime}.txt
7. Run: python -m core.manual_import --source manual_claude --file <path>
8. Repeat for all 6 outputs
9. Then run: python -m core.orchestrator --stage consolidate
""")


async def stage_consolidate(brief_version: str):
    """Merges all hypotheses, flags consensus, deduplicates."""
    console.print("[bold green]Stage: Consolidate[/]")
    all_hyps = load_hypotheses()
    console.print(f"Total hypotheses across all passes: {len(all_hyps)}")
    clusters = await cluster_and_flag(all_hyps)

    # Save cluster output
    out = Path("data/consensus") / f"{brief_version}_clusters.json"
    out.write_text(clusters.model_dump_json(indent=2))

    consensus_count = sum(1 for c in clusters.clusters if c.consensus_flag)
    console.print(f"[green]Clusters: {len(clusters.clusters)}, Consensus-flagged: {consensus_count}[/]")
    return clusters


async def stage_enumerate(brief_version: str):
    """Generates 2-3 variants per surviving hypothesis."""
    console.print("[bold green]Stage: Enumerate[/]")
    hypotheses = load_hypotheses(brief_version)
    variants = await enumerate_variants(hypotheses)
    console.print(f"[green]Generated {len(variants)} variants[/]")
    return variants


async def stage_redteam(brief_version: str):
    """Adversarial review. Kills weak ideas."""
    console.print("[bold green]Stage: Red Team[/]")
    hypotheses = load_hypotheses()
    console.print(f"Reviewing {len(hypotheses)} hypotheses...")
    survivors, reviews = await red_team(hypotheses, BOTTLENECK)
    console.print(f"[green]Survivors: {len(survivors)}[/]")
    return survivors, reviews


async def stage_rank(brief_version: str):
    """Ranks survivors by composite score."""
    console.print("[bold green]Stage: Rank[/]")
    survivors, reviews = await stage_redteam(brief_version)
    clusters_path = Path("data/consensus") / f"{brief_version}_clusters.json"

    from core.schemas import ConsensusClusters
    if clusters_path.exists():
        clusters = ConsensusClusters.model_validate_json(clusters_path.read_text())
    else:
        # No clusters yet — create empty
        clusters = ConsensusClusters(clusters=[])

    ranks = await rank_hypotheses(survivors, reviews, clusters)
    out = Path("data") / f"ranks_{brief_version}.json"
    out.write_text(ranks.model_dump_json(indent=2))
    console.print(f"[green]Ranked {len(ranks.ranks)} hypotheses[/]")
    return survivors, reviews, ranks


async def stage_spec(brief_version: str):
    """Writes experiment specs for top N ranked hypotheses."""
    console.print("[bold green]Stage: Spec Writing[/]")
    survivors, reviews, ranks = await stage_rank(brief_version)

    # Sort survivors by rank order
    rank_order = {r.hypothesis_id: r.recommended_order for r in ranks.ranks}
    survivors.sort(key=lambda h: rank_order.get(h.id, 999))

    specs = await write_specs_for_ranked(survivors, reviews, PIPELINE_DESCRIPTION, TARGET_SPEC_COUNT)
    console.print(f"[bold green]Done. {len(specs)} specs written to data/specs/[/]")

    # Print ranked spec list
    for spec in specs:
        order = rank_order.get(spec.hypothesis_id, "?")
        console.print(f"  #{order} {spec.branch_name}")


async def run_all():
    brief_version = new_brief_version()
    console.print(f"[bold]Starting full pipeline. Brief version: {brief_version}[/]")

    await stage_generate(brief_version)
    await stage_manual_pause(brief_version)

    console.print("[yellow]Waiting for manual imports... Press Enter when done.[/]")
    input()

    await stage_consolidate(brief_version)
    await stage_enumerate(brief_version)
    await stage_spec(brief_version)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _summarize_method_families(papers) -> str:
    from collections import Counter
    # Naive grouping by keywords — replace with Architect synthesis pass if needed
    families = Counter()
    for p in papers:
        text = (p.method + " " + p.core_idea).lower()
        if "surrogate" in text or "docking" in text:
            families["docking_surrogates"] += 1
        if "multi-fidelity" in text or "multifidelity" in text:
            families["multi_fidelity"] += 1
        if "few-shot" in text or "meta-learn" in text or "transfer" in text:
            families["low_data_ml"] += 1
        if "ligand" in text and "structure" not in text:
            families["ligand_only"] += 1
        if "graph" in text or "3d" in text or "equivariant" in text:
            families["structural_dl"] += 1

    return "\n".join(f"- {k}: {v} papers" for k, v in families.most_common())


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "generate", "consolidate", "enumerate",
                                 "redteam", "rank", "spec", "manual-pause"])
    parser.add_argument("--brief-version", default=None)
    args = parser.parse_args()

    bv = args.brief_version or latest_brief_version() or new_brief_version()

    stage_map = {
        "generate": lambda: stage_generate(bv),
        "manual-pause": lambda: stage_manual_pause(bv),
        "consolidate": lambda: stage_consolidate(bv),
        "enumerate": lambda: stage_enumerate(bv),
        "redteam": lambda: stage_redteam(bv),
        "rank": lambda: stage_rank(bv),
        "spec": lambda: stage_spec(bv),
        "all": run_all,
    }

    asyncio.run(stage_map[args.stage]())
```

---

## Step 17 — Brief Compiler (`core/brief.py`)

Generates the compact context brief for manual LLM passes.

```python
# core/brief.py
from __future__ import annotations
from core.schemas import PaperRecord, Hypothesis
from core.memory import graveyard_summary
from pathlib import Path


def compile_brief(
    bottleneck: str,
    pipeline: str,
    papers: list[PaperRecord],
    hypotheses: list[Hypothesis],
    brief_version: str,
) -> str:
    """
    Generates the compact context brief for manual LLM passes.
    Target: 10k-30k tokens. Dense. Structured.
    """
    from collections import defaultdict

    # Group papers by method family (inferred from core_idea keywords)
    families: dict[str, list[PaperRecord]] = defaultdict(list)
    for p in papers:
        text = (p.method + " " + p.core_idea).lower()
        if "surrogate" in text or "docking" in text:
            families["Docking Surrogates"].append(p)
        elif "multi-fidelity" in text:
            families["Multi-Fidelity Pipelines"].append(p)
        elif "few-shot" in text or "meta-learn" in text:
            families["Low-Data ML"].append(p)
        elif "graph" in text or "equivariant" in text:
            families["Structural DL"].append(p)
        else:
            families["Other"].append(p)

    lines = [
        f"# RESEARCH BRIEF",
        f"**Version:** {brief_version}",
        "",
        "---",
        "",
        "## SECTION 1: Problem Definition",
        pipeline,
        "",
        "## SECTION 2: Bottleneck",
        bottleneck,
        "",
        "## SECTION 3: Method Families from Literature",
    ]

    for family, fpapers in families.items():
        lines.append(f"\n### {family}")
        for p in fpapers:
            lines.append(f"**{p.title}**")
            lines.append(f"- Core idea: {p.core_idea}")
            lines.append(f"- Possible transfer: {p.possible_transfer}")
            lines.append(f"- Failure modes: {p.failure_modes}")
            lines.append("")

    lines += [
        "## SECTION 4: Hypotheses Already in System",
        "(Do not regenerate these. They are already being tested or were killed.)",
        "",
    ]
    for h in hypotheses:
        lines.append(f"- **{h.name}** ({h.source.value}): {h.hypothesis[:120]}...")

    lines += [
        "",
        "## SECTION 5: Graveyard (Killed Ideas — Do Not Regenerate)",
        graveyard_summary(),
        "",
        "## SECTION 6: Open Directions",
        "Generate ideas that combine approaches across families, apply underexplored ML techniques,",
        "or borrow from adjacent domains (NLP, vision, robotics) not yet represented above.",
    ]

    brief = "\n".join(lines)

    # Also save to outputs/context_briefs/
    out = Path("outputs/context_briefs") / f"{brief_version}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(brief)

    return brief
```

---

## Step 18 — Prompt Files

Create all prompt files. Copy verbatim from the spec document.

```bash
# Create each prompt file
touch prompts/scout.txt
touch prompts/librarian.txt
touch prompts/architect_grounded.txt
touch prompts/architect_free_range.txt
touch prompts/enumerator.txt
touch prompts/consolidator.txt
touch prompts/adversary.txt
touch prompts/ranker.txt
touch prompts/spec_writer.txt
touch prompts/manual_grounded.txt
touch prompts/manual_free_range.txt
```

Fill each file with the corresponding prompt from the spec. The system prompts (PROMPT 1–9) go into the agent `prompts/` files. MANUAL PROMPT A → `manual_grounded.txt`. MANUAL PROMPT B → `manual_free_range.txt`.

---

## Step 19 — Paper Retrieval Utilities (`core/retrieval.py`)

Utility functions for pulling papers from Semantic Scholar and arXiv. Called by the orchestrator during the paper expansion stage.

```python
# core/retrieval.py
from __future__ import annotations
import httpx
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential


SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"
ARXIV_BASE = "https://export.arxiv.org/api/query"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def semantic_scholar_search(query: str, limit: int = 5) -> list[dict]:
    """
    Returns list of paper dicts with title, abstract, year, externalIds.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{SEMANTIC_SCHOLAR_BASE}/paper/search",
            params={
                "query": query,
                "limit": limit,
                "fields": "title,abstract,year,authors,externalIds,tldr",
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def arxiv_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Returns list of paper dicts with title, summary (abstract).
    """
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            ARXIV_BASE,
            params={
                "search_query": f"all:{query}",
                "max_results": max_results,
                "sortBy": "relevance",
            },
        )
        resp.raise_for_status()

    # Parse Atom XML
    import xml.etree.ElementTree as ET
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        papers.append({
            "title": entry.findtext("atom:title", "", ns).strip(),
            "abstract": entry.findtext("atom:summary", "", ns).strip(),
            "id": entry.findtext("atom:id", "", ns).strip(),
        })
    return papers


async def fetch_papers_for_queries(
    queries: list[str],
    papers_per_query: int = 5,
) -> list[str]:
    """
    Runs all queries against both sources. Returns list of raw text strings
    (title + abstract) suitable for Librarian ingestion.
    De-duplicates by title.
    """
    seen_titles: set[str] = set()
    texts: list[str] = []

    async def fetch_one(query: str):
        results = []
        try:
            ss = await semantic_scholar_search(query, papers_per_query)
            results.extend(ss)
        except Exception as e:
            print(f"[Retrieval] Semantic Scholar failed for '{query}': {e}")

        try:
            ax = await arxiv_search(query, papers_per_query)
            results.extend(ax)
        except Exception as e:
            print(f"[Retrieval] arXiv failed for '{query}': {e}")

        return results

    all_results = await asyncio.gather(*[fetch_one(q) for q in queries])

    for result_list in all_results:
        for paper in result_list:
            title = paper.get("title", "").strip()
            if title in seen_titles:
                continue
            seen_titles.add(title)
            abstract = paper.get("abstract") or paper.get("summary", "")
            text = f"TITLE: {title}\n\nABSTRACT:\n{abstract}"
            texts.append(text)

    print(f"[Retrieval] Fetched {len(texts)} unique papers from {len(queries)} queries")
    return texts
```

---

## Step 20 — Running the System

### First run (full pipeline)

```bash
# 1. Activate environment
source .venv/bin/activate

# 2. Set up your context in core/orchestrator.py
#    Edit BOTTLENECK and PIPELINE_DESCRIPTION strings at the top

# 3. Run full pipeline (will pause for manual LLM pass)
python -m core.orchestrator --stage all
```

### Run individual stages

```bash
# Generate hypotheses only (grounded + free-range)
python -m core.orchestrator --stage generate

# After manual imports, run consolidation
python -m core.orchestrator --stage consolidate --brief-version v20250328_01

# Run red-team only
python -m core.orchestrator --stage redteam --brief-version v20250328_01

# Write specs for ranked survivors
python -m core.orchestrator --stage spec --brief-version v20250328_01
```

### Import manual LLM outputs

```bash
# After running manual LLM passes:
mkdir -p data/manual_inputs

# Save each LLM output to a file, then import:
python -m core.manual_import --source manual_claude --file data/manual_inputs/claude_grounded.txt
python -m core.manual_import --source manual_gemini --file data/manual_inputs/gemini_grounded.txt
python -m core.manual_import --source manual_gpt4 --file data/manual_inputs/gpt4_grounded.txt
python -m core.manual_import --source manual_claude --file data/manual_inputs/claude_free.txt
python -m core.manual_import --source manual_gemini --file data/manual_inputs/gemini_free.txt
python -m core.manual_import --source manual_gpt4 --file data/manual_inputs/gpt4_free.txt
```

### Start Codex branches from specs

```bash
# After specs are written, create git branches:
python -c "
from core.memory import load_specs
import subprocess

specs = load_specs()
for spec in specs:
    subprocess.run(['git', 'checkout', '-b', spec.branch_name, 'main'])
    print(f'Created branch: {spec.branch_name}')
    subprocess.run(['git', 'checkout', 'main'])
"
```

---

## Step 21 — Iteration Loop

After the first run produces specs and Codex branches are active:

1. Monitor Codex agent progress on branches
2. As ideas fail diagnostics or experiments, record them:

```bash
# Add failed hypothesis to graveyard manually if killed outside the automated pipeline
python -c "
from core.schemas import KilledIdea, HypothesisSource, KillStage
from core.memory import add_to_graveyard
killed = KilledIdea(
    id='hyp_<id>',
    name='<name>',
    hypothesis='<brief description>',
    source=HypothesisSource.GROUNDED,
    brief_version='v20250328_01',
    kill_reason='Diagnostic showed no binding signal in ligand-only features',
    fatal_flaws=['Data does not contain signal hypothesis requires'],
    killed_at_stage=KillStage.MANUAL_REVIEW,
)
add_to_graveyard(killed)
print('Added to graveyard')
"
```

3. Add newly found papers to `data/manual_inputs/` or run retrieval again
4. Bump brief version: `new_brief_version()` returns next version automatically
5. Re-run `--stage generate` with new brief version — graveyard is automatically fed to Architect

---

## Appendix A — Model String Reference

Exact model strings as of current availability. Update if endpoints change.

| Use | Model String | Client |
|---|---|---|
| Groq fast (Scout, Enumerator, Ranker, Consolidator) | `llama-3.3-70b-versatile` | `GroqModel` |
| Groq Qwen3 (Enumerator, Ranker) | `qwen-qwq-32b` | `GroqModel` |
| gpt-oss-120b (Architect, Adversary, SpecWriter) | `gpt-oss-120b` | `OpenAIModel` with custom base_url |
| Kimi K2 (Librarian) | `kimi-k2` | `OpenAIModel` with Moonshot base_url |

Adjust model strings in each agent file if the Groq or OSS endpoint uses different identifiers.

---

## Appendix B — LangGraph Scaffold (Alternative Framework)

If you prefer LangGraph for checkpoint resumption and DAG visualization, here is the equivalent scaffold. Drop this into `core/graph.py` and adapt each node to call the same agent functions from Steps 7–16.

```python
# core/graph.py (LangGraph alternative)
from __future__ import annotations
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
import operator


class PipelineState(TypedDict):
    brief_version: str
    papers: list
    hypotheses: Annotated[list, operator.add]   # accumulates across nodes
    reviews: list
    clusters: dict
    ranks: list
    specs: list
    manual_done: bool


def build_graph():
    builder = StateGraph(PipelineState)

    # Add nodes — each calls the corresponding agent function
    builder.add_node("generate", generate_node)
    builder.add_node("manual_pause", manual_pause_node)
    builder.add_node("consolidate", consolidate_node)
    builder.add_node("enumerate", enumerate_node)
    builder.add_node("redteam", redteam_node)
    builder.add_node("rank", rank_node)
    builder.add_node("spec", spec_node)

    # Linear flow
    builder.set_entry_point("generate")
    builder.add_edge("generate", "manual_pause")
    builder.add_conditional_edges(
        "manual_pause",
        lambda state: "consolidate" if state["manual_done"] else "manual_pause",
    )
    builder.add_edge("consolidate", "enumerate")
    builder.add_edge("enumerate", "redteam")

    # Conditional: if too few survivors, re-run generate
    builder.add_conditional_edges(
        "redteam",
        lambda state: "rank" if len(state["hypotheses"]) >= 5 else "generate",
    )
    builder.add_edge("rank", "spec")
    builder.add_edge("spec", END)

    # SqliteSaver enables crash recovery — resume from last checkpoint
    memory = SqliteSaver.from_conn_string("outputs/pipeline_state.db")
    return builder.compile(checkpointer=memory)
```

Key difference from PydanticAI: LangGraph gives you the conditional re-run edge (`if <5 survivors → regenerate`) and `SqliteSaver` for resuming crashed runs. PydanticAI gives you tighter schema validation per agent with less boilerplate. Use PydanticAI for the agents themselves; use LangGraph only if you need the DAG features.

---

## Common Failure Modes and Fixes

**LLM returns non-JSON despite structured prompt**
PydanticAI retries automatically. If it keeps failing, add `model_settings={"response_format": {"type": "json_object"}}` to the `Agent` constructor if the endpoint supports it.

**Groq rate limit hits during long runs**
The `GroqPool` handles failover. Check `outputs/logs/token_usage.jsonl` to see which keys are near the 200k/day limit. If all keys are exhausted, the run will raise `RuntimeError` — add a sleep and retry loop around `groq_pool.chat()` calls.

**Consolidator produces too many clusters (over-splits)**
Reduce the number of top-level clusters by adding to the consolidator prompt: "Aim for 5–10 clusters maximum. Prefer larger, coarser clusters over fine-grained splits."

**Adversary kills fewer than 40% of ideas**
The prompt instructs it to kill aggressively. If it's being too lenient, add to the adversary prompt: "If you have passed more than 60% of ideas, re-review the weakest ones and kill more aggressively. Your job is to kill ideas, not to be polite."

**Manual import parser misses hypotheses from ChatGPT output**
ChatGPT sometimes formats with markdown tables or inconsistent numbering. If `manual_import.py` misses ideas, run it with a more explicit parse prompt that handles tables. Alternatively, lightly reformat the ChatGPT output before importing.
