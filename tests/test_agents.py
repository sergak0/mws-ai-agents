from pathlib import Path

from kaggle_multi_agent.agents import CriticAgent, ModelerAgent, ResearchAgent
from kaggle_multi_agent.contracts import BenchmarkResult, DatasetProfile, ToolPlan
from kaggle_multi_agent.rag import LocalKnowledgeBase


def test_agents_produce_structured_outputs(tmp_path: Path) -> None:
    source_dir = tmp_path / "curated"
    source_dir.mkdir()
    (source_dir / "notes.md").write_text(
        "Use reflection loops. Compare offline public and private metrics.",
        encoding="utf-8",
    )
    kb = LocalKnowledgeBase.build(source_dir)
    profile = DatasetProfile(
        train_rows=10,
        test_rows=4,
        feature_columns=["feature"],
        target_column="target",
        target_min=0,
        target_max=365,
        target_unique=5,
    )
    research_agent = ResearchAgent(kb)
    brief = research_agent.run(iteration=0, dataset_profile=profile, history=[])
    modeler_agent = ModelerAgent()
    spec = modeler_agent.run(iteration=0, research_brief=brief)
    ensemble_spec = modeler_agent.run(iteration=2, research_brief=brief)
    critic_agent = CriticAgent()
    critique = critic_agent.run(
        current=BenchmarkResult(
            experiment_name="baseline",
            holdout_rmse=10.0,
            holdout_mae=8.0,
            offline_public_rmse=9.0,
            offline_private_rmse=8.5,
        ),
        best=None,
        iteration=0,
        max_iterations=3,
    )
    assert brief.retrieved_chunks
    assert isinstance(spec, ToolPlan)
    assert isinstance(ensemble_spec, ToolPlan)
    assert any(step.name == "set_seed_ensemble" for step in ensemble_spec.steps)
    assert critique.decision in {"accept", "retry", "stop"}
