from __future__ import annotations

import unittest

from core.brief import write_manual_packet
from core.schemas import Hypothesis, InferenceCost, NoveltyLevel, PaperRecord, RiskLevel


class ManualPacketTests(unittest.TestCase):
    def test_write_manual_packet_creates_staged_files(self) -> None:
        papers = [
            PaperRecord(
                title="Paper A",
                problem="Docking bottleneck",
                method="Graph surrogate",
                inputs="Inputs",
                outputs="Outputs",
                training_data="Data",
                inference_cost=InferenceCost.MEDIUM,
                core_idea="Core idea",
                relevance_to_project="Relevant",
                possible_transfer="Transfer",
                failure_modes="Failure",
                citations=[],
            ),
        ]
        hypotheses = [
            Hypothesis(
                name="Hypothesis A",
                hypothesis="A grounded hypothesis",
                source="grounded",
                brief_version="vpacket",
                method_family="family",
                how_it_replaces_or_reduces_docking="Reduce docking",
                why_it_should_work_here="Why",
                data_requirements="Data",
                expected_speedup="Speed",
                risk_level=RiskLevel.MEDIUM,
                novelty=NoveltyLevel.MODERATE,
                minimal_prototype="Prototype",
                killer_experiment="Experiment",
                kill_criteria="Criteria",
                paper_refs=[],
            ),
        ]

        packet_dir = write_manual_packet(
            bottleneck="bottleneck",
            pipeline="pipeline",
            papers=papers,
            hypotheses=hypotheses,
            brief_version="vpacket",
        )

        expected = [
            "README.md",
            "problem_context.md",
            "literature_digest.md",
            "hypothesis_inventory.md",
            "graveyard.md",
            "manual_grounded_stage1_context.txt",
            "manual_grounded_stage2_brainstorm.txt",
            "manual_grounded_stage3_finalize.txt",
        ]
        for filename in expected:
            self.assertTrue((packet_dir / filename).exists(), filename)


if __name__ == "__main__":
    unittest.main()
