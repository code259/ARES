from __future__ import annotations

import unittest

from core.brief import compile_brief
from core.schemas import Hypothesis, InferenceCost, NoveltyLevel, PaperRecord, RiskLevel


class BriefCompactionTests(unittest.TestCase):
    def test_compile_brief_limits_papers_and_hypotheses(self) -> None:
        papers = [
            PaperRecord(
                title=f"Paper {index}",
                problem="Docking bottleneck",
                method="Graph surrogate",
                inputs="Inputs",
                outputs="Outputs",
                training_data="Data",
                inference_cost=InferenceCost.MEDIUM,
                core_idea=("core idea " * 30).strip(),
                relevance_to_project=("relevance " * 30).strip(),
                possible_transfer=("transfer " * 30).strip(),
                failure_modes="failure",
                citations=[],
            )
            for index in range(10)
        ]
        hypotheses = [
            Hypothesis(
                name=f"Hypothesis {index}",
                hypothesis=("idea " * 50).strip(),
                source="grounded",
                brief_version="vtest",
                method_family="family",
                how_it_replaces_or_reduces_docking="reduce docking",
                why_it_should_work_here="why",
                data_requirements="data",
                expected_speedup="speed",
                risk_level=RiskLevel.MEDIUM,
                novelty=NoveltyLevel.MODERATE,
                minimal_prototype="prototype",
                killer_experiment="experiment",
                kill_criteria="kill",
                paper_refs=[],
            )
            for index in range(30)
        ]

        brief = compile_brief(
            bottleneck="bottleneck",
            pipeline="pipeline",
            papers=papers,
            hypotheses=hypotheses,
            brief_version="vtest",
        )

        self.assertIn("... 6 more papers in this family omitted for brevity.", brief)
        self.assertIn("... 10 more hypotheses omitted for brevity.", brief)


if __name__ == "__main__":
    unittest.main()
