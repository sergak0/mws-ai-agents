import pytest

from kaggle_multi_agent.contracts import (
    BenchmarkResult,
    ExperimentSpec,
    ModelSpec,
    ToolCall,
    ToolPlan,
)


def test_experiment_spec_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        ExperimentSpec(
            name="baseline",
            feature_pack="default",
            model=ModelSpec(family="lightgbm", objective="regression", params={}),
            prediction_min=10,
            prediction_max=10,
        )


def test_benchmark_result_accepts_non_negative_metrics() -> None:
    result = BenchmarkResult(
        experiment_name="baseline",
        holdout_rmse=10.5,
        holdout_mae=7.2,
        offline_public_rmse=9.8,
        offline_private_rmse=9.4,
    )
    assert result.offline_private_rmse == 9.4


def test_experiment_spec_rejects_non_positive_seed_ensemble_size() -> None:
    with pytest.raises(ValueError):
        ExperimentSpec(
            name="ensemble",
            feature_pack="encoded_default",
            model=ModelSpec(family="lightgbm", objective="regression", params={}),
            seed_ensemble_size=0,
        )


def test_tool_plan_requires_at_least_one_step() -> None:
    with pytest.raises(ValueError):
        ToolPlan(steps=[])


def test_tool_call_requires_name() -> None:
    with pytest.raises(ValueError):
        ToolCall(name="", arguments={})
