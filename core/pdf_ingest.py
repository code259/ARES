from __future__ import annotations

import argparse
import asyncio
import re
from pathlib import Path

from pypdf import PdfReader

from agents.librarian import process_paper
from core.memory import read_bottleneck
from core.schemas import PaperRecord


MANUAL_PAPERS_DIR = Path("data/manual_papers")
EXTRACTED_DIR = Path("data/manual_papers_extracted")
FAILED_DIR = Path("data/manual_papers_failed")
EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
FAILED_DIR.mkdir(parents=True, exist_ok=True)

SECTION_PATTERNS = {
    "abstract": re.compile(r"(?im)^\s*abstract\s*$"),
    "introduction": re.compile(r"(?im)^\s*(\d+\.?\s+)?introduction\s*$"),
    "methods": re.compile(r"(?im)^\s*(\d+\.?\s+)?(method|methods|approach|methodology|model)\s*$"),
    "results": re.compile(r"(?im)^\s*(\d+\.?\s+)?(results|experiments|evaluation)\s*$"),
    "discussion": re.compile(r"(?im)^\s*(\d+\.?\s+)?(discussion|conclusion|conclusions)\s*$"),
    "references": re.compile(r"(?im)^\s*(references|bibliography)\s*$"),
}

KEYWORDS = {
    "docking": 6,
    "dock": 5,
    "binding": 5,
    "affinity": 5,
    "potency": 5,
    "surrogate": 6,
    "proxy": 4,
    "inference": 5,
    "extrapolation": 6,
    "ood": 5,
    "out-of-distribution": 6,
    "low-data": 6,
    "few-shot": 5,
    "transfer": 4,
    "meta-learning": 5,
    "distillation": 5,
    "contrastive": 4,
    "retrieval": 4,
    "uncertainty": 4,
    "gaussian process": 4,
    "protein": 3,
    "ligand": 4,
    "molecular": 4,
    "molecule": 4,
    "graph": 3,
    "equivariant": 4,
    "diffusion": 3,
    "screening": 4,
    "ranking": 3,
    "generalization": 5,
    "scaffold": 5,
}


def _normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _strip_references(text: str) -> str:
    match = SECTION_PATTERNS["references"].search(text)
    if match:
        return text[: match.start()].strip()
    return text


def _guess_title(reader: PdfReader, raw_text: str, fallback_name: str) -> str:
    metadata_title = getattr(reader.metadata, "title", None) if reader.metadata else None
    if metadata_title and metadata_title.strip():
        return metadata_title.strip()

    for line in raw_text.splitlines()[:20]:
        line = line.strip()
        if len(line) < 12:
            continue
        if len(line) > 220:
            continue
        if re.fullmatch(r"[\d\W_]+", line):
            continue
        if line.lower() in {"abstract", "introduction"}:
            continue
        return line
    return Path(fallback_name).stem


def _extract_section(text: str, name: str) -> str:
    pattern = SECTION_PATTERNS[name]
    match = pattern.search(text)
    if not match:
        return ""
    start = match.end()

    next_matches = []
    for key, other_pattern in SECTION_PATTERNS.items():
        if key == name:
            continue
        candidate = other_pattern.search(text, start)
        if candidate:
            next_matches.append(candidate.start())
    end = min(next_matches) if next_matches else len(text)
    return text[start:end].strip()


def _paragraphs(text: str) -> list[str]:
    raw_blocks = re.split(r"\n\s*\n", text)
    blocks = []
    for block in raw_blocks:
        clean = " ".join(line.strip() for line in block.splitlines() if line.strip())
        if len(clean) < 80:
            continue
        if clean.lower().startswith("figure ") or clean.lower().startswith("table "):
            continue
        blocks.append(clean)
    return blocks


def _score_paragraph(paragraph: str) -> int:
    score = 0
    lower = paragraph.lower()
    for keyword, weight in KEYWORDS.items():
        if keyword in lower:
            score += weight
    if any(term in lower for term in ["we propose", "we present", "our method", "we introduce"]):
        score += 3
    if any(term in lower for term in ["results", "improves", "outperforms", "achieves"]):
        score += 2
    if any(term in lower for term in ["dataset", "benchmark", "training", "inference"]):
        score += 2
    return score


def _top_paragraphs(text: str, limit: int = 8) -> list[str]:
    scored = [(paragraph, _score_paragraph(paragraph)) for paragraph in _paragraphs(text)]
    scored = [item for item in scored if item[1] > 0]
    scored.sort(key=lambda item: item[1], reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for paragraph, _score in scored:
        key = paragraph[:140]
        if key in seen:
            continue
        seen.add(key)
        selected.append(paragraph)
        if len(selected) >= limit:
            break
    return selected


def extract_high_signal_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages[:20]:
        pages.append(page.extract_text() or "")
    raw_text = _normalize_text("\n\n".join(pages))
    trimmed = _strip_references(raw_text)
    title = _guess_title(reader, trimmed, pdf_path.name)

    abstract = _extract_section(trimmed, "abstract")
    introduction = _extract_section(trimmed, "introduction")
    methods = _extract_section(trimmed, "methods")
    results = _extract_section(trimmed, "results")
    discussion = _extract_section(trimmed, "discussion")

    chosen_sections: list[str] = []
    if abstract:
        chosen_sections.append("ABSTRACT:\n" + "\n".join(_paragraphs(abstract)[:3]))
    if introduction:
        chosen_sections.append("INTRODUCTION HIGHLIGHTS:\n" + "\n".join(_top_paragraphs(introduction, limit=3)))
    if methods:
        chosen_sections.append("METHOD HIGHLIGHTS:\n" + "\n".join(_top_paragraphs(methods, limit=3)))
    if results:
        chosen_sections.append("RESULT HIGHLIGHTS:\n" + "\n".join(_top_paragraphs(results, limit=3)))
    if discussion:
        chosen_sections.append("DISCUSSION / CONCLUSION HIGHLIGHTS:\n" + "\n".join(_top_paragraphs(discussion, limit=3)))

    if not chosen_sections:
        chosen_sections.append("HIGHLIGHTED EXCERPTS:\n" + "\n".join(_top_paragraphs(trimmed, limit=10)))

    return f"TITLE: {title}\n\n" + "\n\n".join(section for section in chosen_sections if section.strip())


def cache_extraction(pdf_path: Path, extracted_text: str) -> Path:
    out_path = EXTRACTED_DIR / f"{pdf_path.stem}.txt"
    out_path.write_text(extracted_text, encoding="utf-8")
    return out_path


def cache_failure(pdf_path: Path, extracted_text: str, error: str) -> Path:
    out_path = FAILED_DIR / f"{pdf_path.stem}.txt"
    out_path.write_text(
        f"ERROR:\n{error}\n\nEXTRACTED_TEXT_SENT_TO_LIBRARIAN:\n\n{extracted_text}",
        encoding="utf-8",
    )
    return out_path


async def ingest_manual_pdfs(
    pdf_dir: Path = MANUAL_PAPERS_DIR,
    bottleneck: str | None = None,
) -> list[PaperRecord]:
    bottleneck_text = bottleneck or read_bottleneck()
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    records: list[PaperRecord] = []

    for pdf_path in pdf_paths:
        extracted = extract_high_signal_text(pdf_path)
        cache_extraction(pdf_path, extracted)
        try:
            record = await process_paper(extracted, bottleneck_text)
            records.append(record)
            print(f"[PDFIngest] Ingested {pdf_path.name} -> {record.title}")
        except Exception as exc:
            cache_failure(pdf_path, extracted, str(exc))
            print(f"[PDFIngest] Failed {pdf_path.name}: {exc}")

    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(MANUAL_PAPERS_DIR))
    args = parser.parse_args()
    records = asyncio.run(ingest_manual_pdfs(Path(args.dir)))
    print(f"Ingested {len(records)} PDFs.")


if __name__ == "__main__":
    main()
