from pathlib import Path

from kaggle_multi_agent.contracts import BenchmarkResult
from kaggle_multi_agent.registry import ExperimentRegistry


def test_registry_persists_results_and_returns_best_run(tmp_path: Path) -> None:
    registry = ExperimentRegistry(tmp_path / "registry.jsonl")
    first = BenchmarkResult(experiment_name="baseline", holdout_rmse=10.0, holdout_mae=8.0)
    second = BenchmarkResult(experiment_name="improved", holdout_rmse=9.0, holdout_mae=7.0)
    registry.append(first)
    registry.append(second)
    loaded = registry.load()
    best = registry.best()
    assert len(loaded) == 2
    assert best is not None
    assert best.experiment_name == "improved"
