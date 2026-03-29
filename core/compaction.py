from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from core.schemas import ChunkManifest, EvidenceRecord, PaperRecord


def infer_method_family(paper: PaperRecord) -> str:
    text = f"{paper.method} {paper.core_idea} {paper.possible_transfer}".lower()
    if "surrogate" in text or "docking" in text:
        return "docking_surrogates"
    if "multi-fidelity" in text or "multifidelity" in text:
        return "multi_fidelity"
    if "few-shot" in text or "meta" in text or "transfer" in text:
        return "low_data_ml"
    if "ligand" in text:
        return "ligand_only"
    if "graph" in text or "3d" in text or "equivariant" in text or "structure" in text:
        return "structural_dl"
    return "other"


def compact_evidence_records(papers: list[PaperRecord]) -> list[EvidenceRecord]:
    return [
        EvidenceRecord(
            title=paper.title,
            method_family=infer_method_family(paper),
            core_idea=paper.core_idea,
            relevance_to_project=paper.relevance_to_project,
            possible_transfer=paper.possible_transfer,
            failure_modes=paper.failure_modes,
        )
        for paper in papers
    ]


def summarize_method_families_from_evidence(evidence_records: list[EvidenceRecord]) -> str:
    families: dict[str, int] = defaultdict(int)
    for evidence in evidence_records:
        families[evidence.method_family] += 1
    if not families:
        return "- No paper families yet."
    return "\n".join(f"- {family}: {count} papers" for family, count in sorted(families.items()))


def serialize_evidence(evidence_records: list[EvidenceRecord]) -> str:
    compact = [
        {
            "title": evidence.title,
            "family": evidence.method_family,
            "core_idea": evidence.core_idea,
            "relevance": evidence.relevance_to_project,
            "transfer": evidence.possible_transfer,
            "failure_modes": evidence.failure_modes,
        }
        for evidence in evidence_records
    ]
    return json.dumps(compact, separators=(",", ":"), sort_keys=True)


def estimate_tokens(*parts: str, max_tokens: int = 0) -> int:
    text = "".join(parts)
    return max(1, len(text) // 4) + max_tokens


def chunk_evidence_records(
    *,
    role: str,
    brief_version: str,
    model: str,
    evidence_records: list[EvidenceRecord],
    base_context_parts: list[str],
    max_completion_tokens: int,
    safe_input_tokens: int,
) -> list[tuple[ChunkManifest, list[EvidenceRecord]]]:
    grouped: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for evidence in evidence_records:
        grouped[evidence.method_family].append(evidence)

    chunks: list[tuple[ChunkManifest, list[EvidenceRecord]]] = []
    chunk_counter = 1
    base_estimate = estimate_tokens(*base_context_parts, max_tokens=max_completion_tokens)

    for family, family_records in sorted(grouped.items()):
        current: list[EvidenceRecord] = []
        for record in family_records:
            candidate = current + [record]
            serialized = serialize_evidence(candidate)
            estimate = estimate_tokens(*base_context_parts, serialized, max_tokens=max_completion_tokens)
            if current and estimate > safe_input_tokens:
                payload = serialize_evidence(current)
                payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
                manifest = ChunkManifest(
                    chunk_id=f"chunk_{chunk_counter:03d}",
                    role=role,
                    brief_version=brief_version,
                    model=model,
                    estimated_tokens=estimate_tokens(*base_context_parts, payload, max_tokens=max_completion_tokens),
                    item_count=len(current),
                    payload_hash=payload_hash,
                    metadata={"method_family": family},
                )
                chunks.append((manifest, current))
                chunk_counter += 1
                current = [record]
            else:
                current = candidate

        if current:
            payload = serialize_evidence(current)
            payload_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            manifest = ChunkManifest(
                chunk_id=f"chunk_{chunk_counter:03d}",
                role=role,
                brief_version=brief_version,
                model=model,
                estimated_tokens=max(base_estimate, estimate_tokens(*base_context_parts, payload, max_tokens=max_completion_tokens)),
                item_count=len(current),
                payload_hash=payload_hash,
                metadata={"method_family": family},
            )
            chunks.append((manifest, current))
            chunk_counter += 1

    return chunks


def chunk_serialized_items(
    *,
    role: str,
    brief_version: str,
    model: str,
    items: list,
    serialize_item,
    base_context_parts: list[str],
    max_completion_tokens: int,
    safe_input_tokens: int,
    metadata_factory=None,
) -> list[tuple[ChunkManifest, list]]:
    chunks: list[tuple[ChunkManifest, list]] = []
    current: list = []
    chunk_counter = 1

    for item in items:
        candidate = current + [item]
        serialized = "[" + ",".join(serialize_item(value) for value in candidate) + "]"
        estimate = estimate_tokens(*base_context_parts, serialized, max_tokens=max_completion_tokens)
        if current and estimate > safe_input_tokens:
            payload = "[" + ",".join(serialize_item(value) for value in current) + "]"
            metadata = metadata_factory(current) if metadata_factory else {}
            manifest = ChunkManifest(
                chunk_id=f"chunk_{chunk_counter:03d}",
                role=role,
                brief_version=brief_version,
                model=model,
                estimated_tokens=estimate_tokens(*base_context_parts, payload, max_tokens=max_completion_tokens),
                item_count=len(current),
                payload_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                metadata=metadata,
            )
            chunks.append((manifest, current))
            chunk_counter += 1
            current = [item]
        else:
            current = candidate

    if current:
        payload = "[" + ",".join(serialize_item(value) for value in current) + "]"
        metadata = metadata_factory(current) if metadata_factory else {}
        manifest = ChunkManifest(
            chunk_id=f"chunk_{chunk_counter:03d}",
            role=role,
            brief_version=brief_version,
            model=model,
            estimated_tokens=estimate_tokens(*base_context_parts, payload, max_tokens=max_completion_tokens),
            item_count=len(current),
            payload_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            metadata=metadata,
        )
        chunks.append((manifest, current))

    return chunks
