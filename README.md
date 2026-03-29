![header](imgs/header.png)


# ARES

Project ARES: Automated Research Engine for Specification is a structured research acceleration system designed to turn high-level pipeline descriptions, identified bottlenecks, and relevant scientific literature into high-quality, attackable research hypotheses and concrete experiment specifications. Engineered as a solution for low-data, high-compute machine learning environments, ARES utilizes a distinct "Core Loop" architecture that orchestrates multiple, specialized LLM agents in parallel and sequential stages. The system ingests context, retrieves and synthesizes literature, generates hypotheses through both grounded and "free-range" passes, red-teams and ranks the outputs, and ultimately produces implementation-ready specifications.

Unlike conventional, naive agentic systems, ARES integrates several improvements to enhance research quality and efficiency. A core feature is Failure Memory, where every killed idea is stored in a "graveyard" with its kill reason and versioned context, then fed back into subsequent generation cycles to prevent the redundant exploration of failed concepts. The engine also features Consensus Flagging, which automatically prioritizes high-confidence directions that independently emerge across multiple distinct LLM passes. Furthermore, by enforcing a deterministic Python orchestration layer with rigorous data schemas and automated token rotation, ARES ensures traceability, reproducibility, and structured output suitable for immediate parallel execution by Codex agents.

## Quick Start

1. Create a virtual environment and install dependencies.
2. Copy `.env.example` to `.env` and fill in your Groq keys.
3. Edit:
   - `data/context/bottleneck.txt`
   - `data/context/pipeline_description.txt`
4. Run:

```bash
python -m core.orchestrator --stage manual-papers
python -m core.orchestrator --stage generate
python -m core.orchestrator --stage manual-pause

python -m core.manual_import --source manual_claude --file data/manual_inputs/claude_grounded.txt
python -m core.manual_import --source manual_gemini --file data/manual_inputs/gemini_grounded.txt
python -m core.manual_import --source manual_gpt4 --file data/manual_inputs/gpt4_grounded.txt
python -m core.manual_import --source manual_claude --file data/manual_inputs/claude_free.txt
python -m core.manual_import --source manual_gemini --file data/manual_inputs/gemini_free.txt
python -m core.manual_import --source manual_gpt4 --file data/manual_inputs/gpt4_free.txt

python -m core.orchestrator --stage consolidate
python -m core.orchestrator --stage enumerate
python -m core.orchestrator --stage rank
python -m core.orchestrator --stage spec
```

## Notes

- Groq key rotation is built in.
- All model roles are configured as Groq model names, including `gpt-oss-120b` and `kimi-k2` when available through Groq.
- To ingest local PDFs first, put them in `data/manual_papers/` and run `python -m core.orchestrator --stage manual-papers`.
- High-signal extracted text is cached in `data/manual_papers_extracted/`.
- Retrieval works without a Semantic Scholar key by falling back to OpenAlex and arXiv.
- LLM responses are cached locally in `outputs/cache/llm/` to avoid repeat token usage on reruns.
- Per-model per-key request/token budgets are tracked in `outputs/logs/rate_state.json`.
- Manual Claude/Gemini/ChatGPT outputs can be imported with `python -m core.manual_import`.
