# Research Hypothesis Generation System
### Spec v1.0

---

## System Overview

### Goal

Turn your current pipeline, its bottleneck, and relevant literature into:

- **7–10 high-quality, attackable research hypotheses**
- **Concrete experiment specs** (one per hypothesis)
- **Filtered via adversarial review and multi-model consensus**
- **Ranked for parallel Codex execution**

### Core Problem

Current potency prediction model requires molecular docking at inference time. Docking is compute-intensive and does not scale beyond ~200k molecules. Environment is low-data (limited labeled examples). Need to either eliminate docking from inference or dramatically reduce its cost while preserving predictive performance.

---

## Core Loop

```
INGEST → RETRIEVE → SYNTHESIZE → GENERATE (grounded + free-range)
       → ENUMERATE → MERGE → RED-TEAM → RANK → SPEC (7–10 outputs)
```

Key additions over naive agentic systems:
- **Free-range generation pass** runs alongside the grounded pass — Architect runs twice with different prompts
- **Merge + dedup step** before red-teaming collapses near-identical ideas across passes and manual LLM inputs
- **Rank stage** after red-team — ideas get a priority score so you know which branches to give to Codex first
- **Failure memory** fed back into each new generation cycle (Improvement #1)
- **Consensus flagging** across all LLM passes (Improvement #3)
- **Versioned context briefs** so every hypothesis is traceable to the brief that generated it (Improvement #5)

---

## Architecture

### Components

**1. Knowledge Layer**

Stores:
- Pipeline context
- Structured paper records
- Generated hypotheses (grounded + free-range)
- Reviews
- Ranked specs
- Killed ideas + kill reasons (failure memory)
- Brief versions

**2. Agent Layer**

| Role | Purpose | Mode |
|---|---|---|
| Librarian | Paper ingestion + summarization | Grounded |
| Fast Scout | Search query generation | Grounded |
| Architect A | Main idea generator | Grounded (uses paper context) |
| Architect B | Main idea generator | Free-range (training knowledge only) |
| Enumerator | Variant generation | Both |
| Adversary | Red-team critique | Grounded |
| Ranker | Priority scoring | Grounded |
| Consolidator | Cross-LLM consensus clustering | Grounded |

**3. Orchestrator**

Deterministic Python controller:
- Runs stages in order
- Manages memory and brief versioning
- Rotates Groq API keys
- Enforces schemas (nothing enters the pipeline as free text)
- Logs everything including kills

**4. External Loop (Manual)**

You paste the compact context brief into Claude, Gemini, and ChatGPT. Two prompt variants each — grounded and free-range. Outputs feed back into the system for red-teaming and consensus flagging.

---

## Model Assignments

| Role | Model | Notes |
|---|---|---|
| Librarian | kimi-k2 | Cheap, good extraction |
| Fast Scout | groq/compound | Fast query expansion |
| Architect A | gpt-oss-120b | Grounded — reads paper context |
| Architect B | gpt-oss-120b | Free-range — training knowledge only |
| Enumerator | qwen3-32b | Variant generation, both modes |
| Adversary | gpt-oss-120b | Different system prompt from Architect |
| Ranker | qwen3-32b | Lightweight scoring pass |
| Consolidator | qwen3-32b | Cross-pass consensus clustering |

---

## Token Strategy (Groq Rotation)

You have ~600k tokens/day across accounts.

```python
ACCOUNTS = [key1, key2, key3]

def get_client():
    return round_robin(ACCOUNTS)
```

Track tokens used per account. Failover on rate limit. Log per-account usage to `outputs/logs/token_usage.jsonl`.

---

## Data Schemas

### PaperRecord

```json
{
  "title": "",
  "problem": "",
  "method": "",
  "inputs": "",
  "outputs": "",
  "training_data": "",
  "inference_cost": "low | medium | high | unknown",
  "core_idea": "",
  "relevance_to_project": "",
  "possible_transfer": "",
  "failure_modes": "",
  "citations": []
}
```

### Hypothesis

```json
{
  "id": "",
  "name": "",
  "hypothesis": "",
  "source": "grounded | free_range | manual_claude | manual_gemini | manual_gpt4",
  "brief_version": "",
  "method_family": "",
  "how_it_replaces_or_reduces_docking": "",
  "why_it_should_work_here": "",
  "data_requirements": "",
  "expected_speedup": "",
  "risk_level": "low | medium | high",
  "novelty": "incremental | moderate | high",
  "minimal_prototype": "",
  "killer_experiment": "",
  "kill_criteria": "",
  "paper_refs": []
}
```

`source` field tracks which pass generated each idea. `brief_version` ties each hypothesis to the exact context brief used (see Improvement #5).

### KilledIdea (Failure Memory)

```json
{
  "id": "",
  "name": "",
  "hypothesis": "",
  "source": "",
  "brief_version": "",
  "kill_reason": "",
  "fatal_flaws": [],
  "killed_at_stage": "red_team | manual_review | consolidation",
  "date": ""
}
```

Every killed idea is written to `data/graveyard/`. Fed back to Architect on subsequent runs as a "do not regenerate" list. This prevents the system from rediscovering the same bad ideas across sessions (Improvement #1).

### Review

```json
{
  "hypothesis_id": "",
  "fatal_flaws": [],
  "hidden_assumptions": [],
  "data_mismatch": [],
  "benchmark_risks": [],
  "complexity_vs_gain": "",
  "verdict": "kill | revise | proceed",
  "revision_direction": ""
}
```

### RankRecord

```json
{
  "hypothesis_id": "",
  "feasibility_score": 0,
  "novelty_score": 0,
  "speedup_potential_score": 0,
  "data_risk_score": 0,
  "composite_score": 0,
  "recommended_order": 0,
  "rationale": "",
  "consensus_flag": false
}
```

`consensus_flag: true` means this idea appeared in 2+ independent LLM passes — highest-confidence directions (Improvement #3).

### ExperimentSpec

```json
{
  "hypothesis_id": "",
  "goal": "",
  "approach": "",
  "model_changes": "",
  "data_pipeline": "",
  "training_plan": "",
  "evaluation_metrics": "",
  "baseline": "",
  "expected_outcome": "",
  "failure_modes": "",
  "time_estimate": "",
  "branch_name": "",
  "codex_instructions": ""
}
```

---

## Brief Versioning (Improvement #5)

Every time you add papers or run a new generation cycle, the brief gets a version number. Format: `v{date}_{increment}` (e.g., `v20250328_01`).

- Hypotheses record `brief_version` at generation time
- Brief versions are stored in `data/briefs/`
- Killed ideas also record `brief_version` so you can trace what context led to bad ideas
- When you find a new key paper, bump the brief version before re-running Architect

This matters at week 3 when you want to know if an idea was generated before or after a critical paper was added.

---

## Pipeline Stages

### Stage 1 — Ingest Context
- Your pipeline description
- Model architecture details
- Bottleneck definition
- Generate initial brief (v{date}_01)

### Stage 2 — Paper Expansion
- Scout generates 20–25 queries
- Retrieve from Semantic Scholar + arXiv
- Librarian creates `PaperRecord` for each paper
- Cluster into method families

### Stage 3 — Synthesis
- Architect synthesizes method families from paper records
- Outputs: family names, transferable patterns, key constraints

### Stage 4 — Dual Hypothesis Generation
- Architect A (grounded): generates 8–10 hypotheses using paper context + graveyard as "avoid" list
- Architect B (free-range): generates 5–7 hypotheses from training knowledge alone
- Total: ~15 hypotheses before enumeration

### Stage 5 — Manual LLM Passes
- You generate compact brief from current state
- Paste into Claude, Gemini, ChatGPT — grounded prompt each
- Paste bottleneck statement into Claude, Gemini, ChatGPT — free-range prompt each
- 6 total passes, ~5–7 ideas each → ~30–40 raw ideas from manual step
- Parse outputs into Hypothesis schema with `source: manual_*`

### Stage 6 — Merge + Dedup
- Consolidator clusters all ~45–55 raw hypotheses by method family
- Flags semantic duplicates (same core idea, different wording)
- Flags consensus ideas: appeared in 2+ independent sources → `consensus_flag: true`
- Output: deduplicated set with provenance preserved

### Stage 7 — Enumeration
- Enumerator generates 2–3 variants per surviving hypothesis
- Variants must be genuinely distinct (not "use more data")

### Stage 8 — Red Team
- Adversary reviews all candidates
- Target: kill 40%+ aggressively
- Verdicts: kill / revise / proceed
- All kills written to graveyard

### Stage 9 — Rank
- Ranker scores surviving hypotheses
- Composite: feasibility 35%, speedup 35%, data risk (inverted) 20%, novelty 10%
- Consensus-flagged ideas get +0.5 composite bonus
- Output: ordered list, target 7–10 specs

### Stage 10 — Spec Writing
- Spec writer produces implementation-ready spec for each ranked hypothesis
- Includes `codex_instructions` field for immediate Codex handoff
- Includes `branch_name` suggestion for your repo

---

## Paper Discovery System

### Step 1 — Query Generation (Fast Scout)

Input: your bottleneck + pipeline description  
Output: 20–25 targeted queries

Coverage requirements:
- Docking surrogates and approximation methods
- Multi-fidelity ML pipelines
- Low-data ML: few-shot, meta-learning, transfer learning
- Ligand-only potency prediction
- Structure-based deep learning without explicit docking
- Active learning for molecular property prediction
- Physics-informed neural networks for binding
- Uncertainty quantification in drug discovery ML
- Distillation from expensive oracles
- Cascade/screening pipeline approaches

### Step 2 — Retrieval

Sources:
- Semantic Scholar (API available)
- arXiv (scrape or manual paste initially)

### Step 3 — Librarian Processing

Each paper → `PaperRecord`. Strict schema, no free-form summaries.

### Step 4 — Clustering

Group into:
- Docking surrogates
- Multi-fidelity pipelines
- Ligand-only models
- Structural DL models
- Low-data ML techniques

---

## Context Brief Format (for Manual LLM Passes)

The compact brief is the artifact you paste into frontier models. Keep it 10k–30k tokens max. Dense. Structured. Versioned.

```
BRIEF VERSION: v{date}_{increment}

SECTION 1: Problem definition
  - Your pipeline architecture (brief)
  - What the model predicts
  - Why docking is currently required
  - Scale at which the bottleneck hits

SECTION 2: Current bottleneck
  - Why docking is needed (low-data regime, structural signal)
  - Why it's expensive (compute, wall time)
  - What you've already tried or considered
  - Constraints: low data, must preserve structure signal, inference-time only

SECTION 3: Method families from literature
  - Summary of each approach (5–10 lines each)
  - Key papers per family (title + 1 sentence)

SECTION 4: Hypotheses already in system
  - Names only + one-line description
  - Killed ideas + why (graveyard summary)
  - Goal: prevent LLM from regenerating what's already been considered

SECTION 5: Open directions
  - Gaps identified by synthesis stage
  - Combinations not yet explored
```

---

## All Prompts

---

### PROMPT 1 — Fast Scout (Query Generation)

```
ROLE: You are a scientific search specialist.

TASK: Generate 20–25 targeted search queries to find papers relevant to solving the 
following ML bottleneck.

BOTTLENECK:
{bottleneck_description}

CURRENT SYSTEM:
{pipeline_description}

CONSTRAINTS:
- Low data environment (limited labeled examples)
- Must preserve structural signal from molecular data
- Need to reduce or eliminate dependence on docking for inference

OUTPUT FORMAT: Return only a JSON array of query strings. No explanation.

COVERAGE REQUIREMENTS:
- Docking surrogates and approximation methods
- Multi-fidelity ML pipelines
- Low-data ML: few-shot, meta-learning, transfer learning
- Ligand-only potency prediction
- Structure-based deep learning without explicit docking
- Active learning for molecular property prediction
- Physics-informed neural networks for binding
- Uncertainty quantification in drug discovery ML
- Distillation from expensive oracles
- Cascade/screening pipeline approaches
```

---

### PROMPT 2 — Librarian (Paper Extraction)

```
ROLE: You are a precise scientific knowledge extractor.

TASK: Read the following paper and extract a structured record.

PAPER TEXT:
{paper_text}

PROJECT CONTEXT:
{bottleneck_description}

Return a JSON object matching this exact schema:
{
  "title": "",
  "problem": "",
  "method": "",
  "inputs": "",
  "outputs": "",
  "training_data": "",
  "inference_cost": "low | medium | high | unknown",
  "core_idea": "",
  "relevance_to_project": "",
  "possible_transfer": "",
  "failure_modes": "",
  "citations": []
}

Rules:
- Be specific. No vague summaries.
- "possible_transfer" = concrete ideas for how this method could apply to our bottleneck
- "failure_modes" = known weaknesses from the paper or obvious ones
- If unsure about inference_cost, use "unknown"
- Return only the JSON. No markdown.
```

---

### PROMPT 3 — Architect A (Grounded Hypothesis Generation)

```
ROLE: You are a senior ML research architect specializing in drug discovery pipelines.

TASK: Generate 8–10 high-quality research hypotheses to solve the bottleneck described below. 
You MUST ground each hypothesis in at least one of the provided paper records.

BOTTLENECK:
{bottleneck_description}

PIPELINE:
{pipeline_description}

PAPER RECORDS:
{paper_records_json}

METHOD FAMILIES IDENTIFIED:
{method_families}

GRAVEYARD — IDEAS ALREADY KILLED (do not regenerate these):
{graveyard_summary}

For each hypothesis, return a JSON object:
{
  "id": "hyp_{uuid}",
  "name": "",
  "hypothesis": "",
  "source": "grounded",
  "brief_version": "{brief_version}",
  "method_family": "",
  "how_it_replaces_or_reduces_docking": "",
  "why_it_should_work_here": "",
  "data_requirements": "",
  "expected_speedup": "estimate with reasoning",
  "risk_level": "low | medium | high",
  "novelty": "incremental | moderate | high",
  "minimal_prototype": "",
  "killer_experiment": "",
  "kill_criteria": "",
  "paper_refs": ["title1", "title2"]
}

Rules:
- Be specific. "Use a neural network" is not a hypothesis.
- expected_speedup must include a rough order-of-magnitude estimate and your reasoning.
- killer_experiment = the single fastest experiment that would tell you if this works or not.
- kill_criteria = what result would cause you to immediately abandon this idea.
- Do not generate ideas that appear in the GRAVEYARD section.
- Return a JSON array of hypothesis objects. No markdown.
```

---

### PROMPT 4 — Architect B (Free-Range Hypothesis Generation)

```
ROLE: You are a senior ML research architect specializing in drug discovery and molecular ML.

TASK: Generate 5–7 research hypotheses to solve the bottleneck below.

IMPORTANT: Do NOT try to recall specific papers. Reason purely from your understanding of 
ML techniques, drug discovery, and molecular modeling. The goal is to surface ideas 
that might NOT appear in a standard literature search — techniques from adjacent fields, 
underexplored combinations, or first-principles approaches.

BOTTLENECK:
{bottleneck_description}

PIPELINE (brief):
{pipeline_description}

GRAVEYARD — IDEAS ALREADY KILLED (do not regenerate these):
{graveyard_summary}

Think about:
- Techniques from adjacent fields (NLP, vision, robotics) that could transfer
- Architectural changes that fundamentally change the compute profile
- Data augmentation or synthetic data strategies for low-data regimes
- Self-supervised or contrastive approaches that could reduce label dependence
- Approximation or distillation approaches not yet standard in this domain
- Combinations of techniques that are novel in this problem space

For each hypothesis:
{
  "id": "hyp_{uuid}",
  "name": "",
  "hypothesis": "",
  "source": "free_range",
  "brief_version": "{brief_version}",
  "method_family": "",
  "how_it_replaces_or_reduces_docking": "",
  "why_it_should_work_here": "",
  "data_requirements": "",
  "expected_speedup": "estimate with reasoning",
  "risk_level": "low | medium | high",
  "novelty": "incremental | moderate | high",
  "minimal_prototype": "",
  "killer_experiment": "",
  "kill_criteria": "",
  "paper_refs": []
}

Return a JSON array. Paper refs are not required in free-range mode.
No markdown.
```

---

### PROMPT 5 — Enumerator (Variant Generation)

```
ROLE: You are a creative ML research variant generator.

TASK: For each hypothesis below, generate 2–3 concrete variants that explore different 
implementation strategies, data regimes, or architectural choices.

HYPOTHESES:
{hypotheses_json}

For each variant, produce a new hypothesis object with the SAME schema.
- Generate a new unique id: "hyp_{uuid}"
- Set "source" to "{parent_hypothesis_name}_variant"
- Preserve "brief_version" from parent
- Be concrete. A variant is not "use more data" — it is a specific architectural or 
  methodological change.
- Variants should be genuinely different from each other and from the parent.
- Do not add paper_refs unless you are confident they are accurate.

Return a flat JSON array of all variant hypothesis objects. No markdown.
```

---

### PROMPT 6 — Consolidator (Cross-Pass Consensus Clustering)

```
ROLE: You are a research synthesis specialist.

TASK: You have received hypotheses from multiple independent sources. Your job is to:
1. Cluster semantically similar ideas by method family
2. Flag duplicates (same core idea, different wording) — keep the best-written version
3. Flag consensus ideas: any idea that appears in 2 or more independent sources

ALL HYPOTHESES (from all passes):
{all_hypotheses_json}

SOURCE LABELS:
- grounded: generated by system from papers
- free_range: generated by system without papers
- manual_claude: from Claude manual pass
- manual_gemini: from Gemini manual pass
- manual_gpt4: from ChatGPT manual pass
- *_variant: enumeration variants

For each cluster, return:
{
  "cluster_id": "",
  "method_family": "",
  "representative_hypothesis_id": "",
  "member_ids": [],
  "sources_represented": [],
  "consensus_flag": true/false,
  "consensus_rationale": ""
}

consensus_flag = true if 2+ DIFFERENT source types appear in member_ids.
Return a JSON array of cluster objects. No markdown.
```

---

### PROMPT 7 — Adversary (Red-Team)

```
ROLE: You are a ruthless scientific adversary. Your job is to kill weak ideas.

TASK: Review each hypothesis below and produce a structured critique.
Be aggressive. Most of these should not survive.

BOTTLENECK:
{bottleneck_description}

HARD CONSTRAINTS (violations are automatic kills):
- Must work at inference time without docking
- Must function in a low-data environment
- Must integrate with existing pipeline without full retraining

HYPOTHESES:
{hypotheses_json}

For each hypothesis, return:
{
  "hypothesis_id": "",
  "fatal_flaws": [],
  "hidden_assumptions": [],
  "data_mismatch": [],
  "benchmark_risks": [],
  "complexity_vs_gain": "",
  "verdict": "kill | revise | proceed",
  "revision_direction": "if verdict is revise, what specifically needs to change — one sentence"
}

Verdict criteria:
- kill: fundamental blocker that cannot be addressed without rethinking the approach
- revise: fixable problem with a clear path
- proceed: ready for spec writing

Aim to kill at least 40% of ideas. If you are not killing ideas, you are not doing your job.
All kills will be written to the failure memory graveyard.
Return a JSON array of review objects. No markdown.
```

---

### PROMPT 8 — Ranker

```
ROLE: You are a research prioritization specialist.

TASK: Given the surviving hypotheses and their reviews, rank them for implementation priority.

SURVIVING HYPOTHESES:
{surviving_hypotheses_json}

REVIEWS:
{reviews_json}

CONSENSUS FLAGS:
{consensus_flags_json}

CONSTRAINTS:
- Team will run ideas in parallel across ~10 Codex coding agents
- Want to maximize probability of finding a solution within 2–4 weeks
- Low-data constraint is hard and non-negotiable
- Fast iteration is more valuable than theoretical novelty
- Consensus-flagged ideas are higher confidence

For each hypothesis, return:
{
  "hypothesis_id": "",
  "feasibility_score": 1-10,
  "novelty_score": 1-10,
  "speedup_potential_score": 1-10,
  "data_risk_score": 1-10,
  "composite_score": 1-10,
  "consensus_flag": true/false,
  "recommended_order": 1-N,
  "rationale": ""
}

Composite weights: feasibility 35%, speedup 35%, data_risk_inverted 20%, novelty 10%.
Consensus-flagged ideas: add 0.5 to composite before final ranking.
Target output: 7–10 ranked ideas.
Return a JSON array sorted by recommended_order ascending. No markdown.
```

---

### PROMPT 9 — Spec Writer

```
ROLE: You are a senior ML engineer writing an implementation spec for a coding agent.

TASK: Write a complete, implementation-ready experiment spec for the hypothesis below.

HYPOTHESIS:
{hypothesis_json}

REVIEW:
{review_json}

PIPELINE CONTEXT:
{pipeline_description}

Return:
{
  "hypothesis_id": "",
  "goal": "",
  "approach": "step-by-step, concrete — numbered list of actions",
  "model_changes": "exact architectural changes needed",
  "data_pipeline": "what data, how to get it, how to preprocess",
  "training_plan": "loss function, optimizer, schedule, batch size guidance",
  "evaluation_metrics": "primary and secondary metrics with target thresholds",
  "baseline": "what you compare against",
  "expected_outcome": "quantitative target",
  "failure_modes": "what to watch for during training and eval",
  "time_estimate": "realistic estimate for a competent engineer",
  "branch_name": "git-branch-name-suggestion",
  "codex_instructions": "3–5 sentences a coding agent can act on immediately to start implementing this"
}

The codex_instructions field must be self-contained — assume the agent has access to 
the repo but has not read this spec. Include: what to build, what to measure, what 
success looks like.

Return only the JSON object. No markdown.
```

---

## Manual LLM Prompts

These are the prompts you paste into Claude, Gemini, and ChatGPT. Use both variants for each model. Six total passes.

---

### MANUAL PROMPT A — Grounded
*Paste this with the compact context brief attached*

```
You are a senior ML research scientist advising on a drug discovery pipeline bottleneck.

CONTEXT BRIEF:
[PASTE COMPACT RESEARCH BRIEF HERE — include brief version number]

Your task:
1. Generate 5–8 novel research hypotheses to solve this bottleneck.
2. Draw primarily from the method families and papers described in the brief.
3. Combine or extend ideas in ways that are NOT already listed as existing hypotheses 
   in the brief. Do not regenerate killed ideas listed in the graveyard section.
4. For each hypothesis, provide:
   - Name
   - Core idea (2–3 sentences, technically specific)
   - Why it should work in a low-data regime
   - Estimated speedup vs. docking (order of magnitude, with reasoning)
   - Biggest risk
   - Minimal experiment to validate (what you would run in 2 weeks)

Focus on ideas that are: (a) feasible within 1–2 months, (b) do not require massive 
new data collection, and (c) could plausibly replace or eliminate docking at inference time.

Format: numbered list, one hypothesis per item. Be specific and technical.
```

---

### MANUAL PROMPT B — Free-Range
*Paste this without the brief — only give the bottleneck statement*

```
You are a senior ML research scientist. I have a drug discovery ML pipeline with a 
specific bottleneck I need to solve.

BOTTLENECK:
My current potency prediction model requires molecular docking at inference time.
Docking is compute-intensive and does not scale beyond ~200k molecules.
The environment is low-data (limited labeled examples).
I need to either eliminate docking from inference or dramatically reduce its cost 
while preserving predictive performance.

Do NOT focus on papers I might already know. Instead, reason from first principles 
and from techniques in adjacent ML domains that are underrepresented in drug 
discovery literature.

Generate 5–7 research hypotheses. For each:
- Name
- Core idea (technically specific)
- Why it applies to a low-data molecular ML problem
- What makes it non-obvious or underexplored in this domain
- Biggest risk
- What a 2-week prototype would look like

Be technically specific. No hand-waving. Treat this as a real research problem.
```

---

### Why Two Prompts Per Model

The grounded prompt anchors the LLM to your actual literature — good for surfacing combinations and extensions you missed. The free-range prompt forces the model to go off-script — good for ideas that won't appear in a standard literature review because they come from NLP, vision, robotics, or other fields.

Running both on all three frontier models gives you 6 independent passes, which is enough volume (30–40 ideas) to funnel down to 7–10 quality specs after dedup, consensus flagging, red-teaming, and ranking.

---

## Improvements Built Into This System

### Improvement #1 — Failure Memory

Every killed idea is stored in `data/graveyard/` with its kill reason and brief version. The Architect prompts (both grounded and free-range) receive a graveyard summary at generation time as an explicit "do not regenerate these" list.

Without this, the system rediscovers the same bad ideas every session. With it, each cycle builds on the accumulated knowledge of what has already failed.

### Improvement #3 — Consensus Flagging

After all 6 manual LLM passes plus the two automated Architect passes, the Consolidator clusters outputs by method family and flags any idea that appears in 2+ independent source types as `consensus_flag: true`. These are your highest-confidence directions because multiple independent systems with different prompts and training reached the same conclusion.

Consensus-flagged ideas get a +0.5 composite score bonus in the Ranker. They should be prioritized for the first Codex branches you launch.

### Improvement #5 — Versioned Context Briefs

Every brief is stored with a version number (`v{date}_{increment}`). Every hypothesis records which brief version it was generated from. Every killed idea also records its brief version.

This matters at week 3 when you want to know: "Was this idea generated before or after we found the key paper on multi-fidelity surrogates?" If the answer is before, that idea may need to be regenerated or revisited with updated context.

---

## Design Rules

**Rule 1 — Force grounding.** Every grounded hypothesis must reference at least one paper record. Free-range hypotheses are exempt by design.

**Rule 2 — Kill aggressively.** Adversary targets 40%+ kill rate. Most ideas should die. The goal is 7–10 specs, not 30.

**Rule 3 — No free-form outputs.** Everything structured. Nothing enters the pipeline as plain text — all agent outputs are validated against schema before proceeding.

**Rule 4 — Limit parallel branches.** Max 10 Codex branches running simultaneously. Ranked order tells you which to launch first.

**Rule 5 — Track all failures.** Killed ideas are not discarded — they are stored, versioned, and fed back as constraints. A good graveyard is a competitive advantage.

**Rule 6 — Diagnostic-first in Codex.** Each coding agent runs a short diagnostic before full implementation: does the data actually have the signal this hypothesis requires? This can catch dead ideas in hours, not weeks.

---

## File Structure

```
research_system/
│
├── data/
│   ├── papers/
│   ├── hypotheses/
│   ├── reviews/
│   ├── specs/
│   ├── graveyard/           ← killed ideas + kill reasons
│   ├── briefs/              ← versioned context briefs
│   └── consensus/           ← consolidator cluster outputs
│
├── agents/
│   ├── librarian.py
│   ├── architect.py         ← runs both grounded and free-range
│   ├── adversary.py
│   ├── enumerator.py
│   ├── scout.py
│   ├── ranker.py
│   └── consolidator.py      ← cross-pass consensus
│
├── core/
│   ├── orchestrator.py
│   ├── memory.py            ← graveyard read/write + brief versioning
│   └── groq_client.py       ← key rotation + failover
│
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
│   ├── manual_grounded.txt  ← paste into Claude/Gemini/GPT
│   └── manual_free_range.txt
│
└── outputs/
    ├── context_briefs/
    └── logs/
        └── token_usage.jsonl
```

---

## What This System Actually Gives You

Not a random idea generator. A structured research accelerator that:

- Surfaces non-obvious approaches from literature
- Imports ideas from adjacent ML domains via free-range passes
- Prevents idea recycling through failure memory
- Amplifies confidence in directions that multiple frontier models independently reach
- Produces implementation-ready specs with Codex-ready instructions
- Gives you a ranked queue so you always know what to launch next

The power is the tight loop: **literature → hypothesis (grounded + free-range) → consensus → adversarial filtering → ranked spec → parallel Codex execution → back into the loop with failure memory updated.**
