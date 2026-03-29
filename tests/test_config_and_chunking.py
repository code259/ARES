from __future__ import annotations

import os
import unittest

from core.compaction import chunk_evidence_records, compact_evidence_records
from core.config import SAFE_PROMPT_TOKENS, gemini_api_key, groq_keys, role_endpoint, role_fallback_endpoints
from core.schemas import InferenceCost, PaperRecord


class ConfigAndChunkingTests(unittest.TestCase):
    def test_groq_keys_supports_four_keys(self) -> None:
        original = {key: os.environ.get(key) for key in ["GROQ_KEY_1", "GROQ_KEY_2", "GROQ_KEY_3", "GROQ_KEY_4"]}
        try:
            os.environ["GROQ_KEY_1"] = "k1"
            os.environ["GROQ_KEY_2"] = "k2"
            os.environ["GROQ_KEY_3"] = "k3"
            os.environ["GROQ_KEY_4"] = "k4"
            self.assertEqual(groq_keys(), ["k1", "k2", "k3", "k4"])
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_grounded_chunking_splits_large_evidence(self) -> None:
        papers = [
            PaperRecord(
                title=f"Paper {index}",
                problem="Docking bottleneck",
                method="Docking surrogate with graph model",
                inputs="SMILES",
                outputs="Affinity",
                training_data="Small active set",
                inference_cost=InferenceCost.MEDIUM,
                core_idea=("Use structural surrogate signals " * 40).strip(),
                relevance_to_project=("Preserve extrapolation " * 30).strip(),
                possible_transfer=("Feed score proxy into delta model " * 30).strip(),
                failure_modes=("May overfit low-data regime " * 20).strip(),
                citations=[],
            )
            for index in range(18)
        ]
        evidence = compact_evidence_records(papers)
        endpoint = role_endpoint("architect")
        chunks = chunk_evidence_records(
            role="architect",
            brief_version="vtest",
            model="openai/gpt-oss-120b",
            evidence_records=evidence,
            base_context_parts=["BOTTLENECK:", "x" * 500, "PIPELINE:", "y" * 500],
            max_completion_tokens=1500,
            safe_input_tokens=SAFE_PROMPT_TOKENS["openai/gpt-oss-120b"],
        )
        self.assertGreater(len(chunks), 1)
        self.assertEqual(sum(manifest.item_count for manifest, _chunk in chunks), len(evidence))

    def test_gemini_key_and_role_routing(self) -> None:
        original = {
            key: os.environ.get(key)
            for key in [
                "GEMINI_API_KEY",
                "ARCHITECT_MODEL",
                "ARCHITECT_FALLBACK_MODEL",
            ]
        }
        try:
            os.environ["GEMINI_API_KEY"] = "gem-key"
            os.environ.pop("ARCHITECT_MODEL", None)
            os.environ.pop("ARCHITECT_FALLBACK_MODEL", None)
            self.assertEqual(gemini_api_key(), "gem-key")
            endpoint = role_endpoint("architect")
            fallbacks = role_fallback_endpoints("architect")
            self.assertEqual(endpoint.provider, "gemini")
            self.assertEqual(endpoint.model, "gemini-3-flash-preview")
            self.assertEqual([item.model for item in fallbacks[:3]], ["gemini-3-flash-preview", "gemini-2.5-flash", "openai/gpt-oss-120b"])
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
