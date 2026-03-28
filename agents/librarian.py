from __future__ import annotations

import re

from core.config import PROMPTS_DIR
from core.llm import OversizeRequestError, llm_registry
from core.memory import save_paper
from core.schemas import PaperRecord


PROMPT = (PROMPTS_DIR / "librarian.txt").read_text(encoding="utf-8")


def _compress_text(paper_text: str, max_chars: int) -> str:
    title_match = re.search(r"(?im)^title:\s*(.+)$", paper_text)
    title_line = f"TITLE: {title_match.group(1).strip()}" if title_match else ""

    sections = re.split(r"\n\s*\n", paper_text)
    kept: list[str] = []
    for section in sections:
        clean = section.strip()
        if not clean:
            continue
        lower = clean.lower()
        if lower.startswith("title:"):
            continue
        if any(
            marker in lower
            for marker in [
                "abstract:",
                "method highlights:",
                "result highlights:",
                "discussion / conclusion highlights:",
                "introduction highlights:",
                "highlighted excerpts:",
            ]
        ):
            kept.append(clean)

    body = "\n\n".join(kept) if kept else paper_text
    if len(body) > max_chars:
        paragraphs = [block.strip() for block in re.split(r"\n\s*\n", body) if block.strip()]
        shortened: list[str] = []
        current = 0
        for paragraph in paragraphs:
            if current + len(paragraph) + 2 > max_chars:
                remaining = max_chars - current
                if remaining > 200:
                    shortened.append(paragraph[:remaining].rsplit(" ", 1)[0] + " ...")
                break
            shortened.append(paragraph)
            current += len(paragraph) + 2
        body = "\n\n".join(shortened)
    return f"{title_line}\n\n{body}".strip()


async def process_paper(paper_text: str, bottleneck: str) -> PaperRecord:
    attempts = [
        paper_text,
        _compress_text(paper_text, 18000),
        _compress_text(paper_text, 12000),
        _compress_text(paper_text, 8000),
        _compress_text(paper_text, 5000),
    ]

    last_error: Exception | None = None
    seen_payloads: set[str] = set()
    for candidate in attempts:
        if candidate in seen_payloads:
            continue
        seen_payloads.add(candidate)
        try:
            result = await llm_registry.complete_structured(
                role="librarian",
                system_prompt=PROMPT,
                user_prompt=f"PAPER TEXT:\n{candidate}\n\nPROJECT CONTEXT:\n{bottleneck}",
                response_model=PaperRecord,
                temperature=0.1,
                max_tokens=1200,
            )
            save_paper(result)
            return result
        except OversizeRequestError as exc:
            last_error = exc
            continue
        except Exception as exc:
            message = str(exc).lower()
            last_error = exc
            if "request too large" in message or "413" in message or "tokens per minute" in message:
                continue
            raise

    raise RuntimeError(f"Librarian could not fit the paper into model limits after compression: {last_error}")


async def process_papers_batch(papers: list[str], bottleneck: str) -> list[PaperRecord]:
    records: list[PaperRecord] = []
    for index, paper in enumerate(papers, start=1):
        try:
            record = await process_paper(paper, bottleneck)
            records.append(record)
            print(f"[Librarian] {index}/{len(papers)} {record.title}")
        except Exception as exc:
            print(f"[Librarian] Failed paper {index}: {exc}")
    return records
