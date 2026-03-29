import json

from kaggle_multi_agent.contracts import BenchmarkResult, CritiqueResult, ExperimentSpec, ModelSpec
from kaggle_multi_agent.tracking import JsonlTracker


def test_jsonl_tracker_persists_iteration_events(tmp_path) -> None:
    tracker = JsonlTracker(tmp_path / "events.jsonl")
    tracker.log_iteration(
        iteration=1,
        best_so_far=False,
        spec=ExperimentSpec(
            name="iteration_01",
            feature_pack="encoded_default",
            model=ModelSpec(
                family="lightgbm",
                objective="regression",
                params={"n_estimators": 10},
            ),
        ),
        result=BenchmarkResult(
            experiment_name="iteration_01",
            holdout_rmse=1.0,
            holdout_mae=0.8,
            offline_public_rmse=1.1,
            offline_private_rmse=1.2,
        ),
        critique=CritiqueResult(
            decision="retry",
            summary="continue",
            actions=["tune features"],
        ),
        tool_trace=["select_feature_pack", "select_model_preset"],
        guardrail_events=["removed suspicious line"],
    )
    payload = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").strip())
    assert payload["iteration"] == 1
    assert payload["tool_trace"] == ["select_feature_pack", "select_model_preset"]
    assert payload["guardrail_events"] == ["removed suspicious line"]
