from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import core.memory as memory
from core.schemas import InferenceCost, PaperRecord


class MultiRunMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.original_dirs = {
            name: getattr(memory, name)
            for name in [
                "BRIEFS",
                "GRAVEYARD",
                "HYPOTHESES",
                "REVIEWS",
                "SPECS",
                "PAPERS",
                "CONSENSUS",
                "RANKS",
                "MANUAL_INPUTS",
                "CONTEXT",
                "STATE",
                "MANIFESTS",
                "RUNS",
            ]
        }
        for name in self.original_dirs:
            path = root / name.lower()
            path.mkdir(parents=True, exist_ok=True)
            setattr(memory, name, path)
        (memory.CONTEXT / "bottleneck.txt").write_text("bottleneck", encoding="utf-8")
        (memory.CONTEXT / "pipeline_description.txt").write_text("pipeline", encoding="utf-8")

    def tearDown(self) -> None:
        for name, value in self.original_dirs.items():
            setattr(memory, name, value)
        self.tempdir.cleanup()

    @staticmethod
    def _paper(title: str) -> PaperRecord:
        return PaperRecord(
            title=title,
            problem="Docking bottleneck",
            method="Method",
            inputs="Inputs",
            outputs="Outputs",
            training_data="Data",
            inference_cost=InferenceCost.MEDIUM,
            core_idea="Idea",
            relevance_to_project="Relevant",
            possible_transfer="Transfer",
            failure_modes="Failure",
            citations=[],
        )

    def test_run_manifest_scopes_papers(self) -> None:
        alpha = self._paper("Alpha")
        beta = self._paper("Beta")
        memory.save_papers([alpha, beta])
        memory.create_run_manifest("v20260328_01", paper_ids=[memory.paper_storage_key(alpha)])
        memory.create_run_manifest("v20260328_02", paper_ids=[memory.paper_storage_key(beta)])

        self.assertEqual([paper.title for paper in memory.load_papers("v20260328_01")], ["Alpha"])
        self.assertEqual([paper.title for paper in memory.load_papers("v20260328_02")], ["Beta"])

    def test_legacy_brief_bootstraps_manifest_without_data_loss(self) -> None:
        alpha = self._paper("Alpha")
        memory.save_paper(alpha)
        memory.save_brief("v20260328_01", "legacy brief")
        manifest_path = memory.RUNS / "v20260328_01.json"
        manifest_path.unlink()

        manifest = memory.load_run_manifest("v20260328_01")

        self.assertIsNotNone(manifest)
        assert manifest is not None
        self.assertEqual(manifest.status, "legacy_imported")
        self.assertEqual(manifest.paper_ids, [memory.paper_storage_key(alpha)])

    def test_new_brief_version_counts_existing_run_manifests(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        memory.create_run_manifest(f"v{today}_01")
        memory.create_run_manifest(f"v{today}_02")

        self.assertTrue(memory.new_brief_version().endswith("_03"))


if __name__ == "__main__":
    unittest.main()
