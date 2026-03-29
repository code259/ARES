from __future__ import annotations

import re


KEYWORDS = {
    "docking": 8,
    "inference": 7,
    "potency": 7,
    "pIC50": 7,
    "extrapolation": 8,
    "out-of-distribution": 8,
    "ood": 6,
    "scaffold": 7,
    "rmse": 7,
    "gaussian process": 6,
    "ransac": 5,
    "xgboost": 4,
    "delta model": 7,
    "score_mean": 7,
    "dist_to_walker_a": 7,
    "hbond_counts": 6,
    "low-data": 8,
    "few-shot": 5,
    "surrogate": 7,
    "distillation": 6,
    "contrastive": 5,
    "ligand": 4,
    "protein": 4,
    "recA".lower(): 5,
    "constraints": 6,
    "must": 4,
}

SECTION_BOOSTS = {
    "executive summary": 20,
    "system overview": 18,
    "model architecture": 18,
    "training constraints": 16,
    "future research directions": 14,
    "research search space": 14,
    "non-negotiable project constraints": 22,
}


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _blocks(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", _normalize(text)) if block.strip()]


def _score(block: str) -> int:
    lower = block.lower()
    score = 0
    for heading, boost in SECTION_BOOSTS.items():
        if heading in lower:
            score += boost
    for keyword, weight in KEYWORDS.items():
        if keyword in lower:
            score += weight
    if lower.startswith("#") or lower.startswith("##"):
        score += 6
    if any(term in lower for term in ["result", "constraint", "bottleneck", "current", "final architecture"]):
        score += 4
    return score


def compact_context_text(text: str, max_chars: int = 5000) -> str:
    blocks = _blocks(text)
    if len(text) <= max_chars:
        return _normalize(text)

    chosen: list[str] = []
    used = 0

    # Always preserve the title + first two substantive blocks for identity and high-level framing.
    for block in blocks[:3]:
        if used + len(block) + 2 > max_chars:
            break
        chosen.append(block)
        used += len(block) + 2

    ranked = sorted(
        ((block, _score(block)) for block in blocks[3:]),
        key=lambda item: item[1],
        reverse=True,
    )

    seen = set(chosen)
    for block, score in ranked:
        if score <= 0 or block in seen:
            continue
        remaining = max_chars - used
        if remaining <= 200:
            break
        if len(block) + 2 <= remaining:
            chosen.append(block)
            seen.add(block)
            used += len(block) + 2
            continue
        truncated = block[:remaining].rsplit(" ", 1)[0].strip()
        if len(truncated) >= 150:
            chosen.append(truncated + " ...")
            used = max_chars
            break

    return "\n\n".join(chosen).strip()
