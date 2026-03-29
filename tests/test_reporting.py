from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_multi_agent.contracts import BenchmarkResult, DatasetProfile, ExperimentSpec, ModelSpec
from kaggle_multi_agent.reporting import write_run_report, write_submission_file


def test_write_run_report_creates_markdown_summary(tmp_path: Path) -> None:
    profile = DatasetProfile(
        train_rows=10,
        test_rows=4,
        feature_columns=["feature"],
        target_column="target",
        target_min=0,
        target_max=365,
        target_unique=5,
    )
    final_state = {
        "history": [
            BenchmarkResult(
                experiment_name="iteration_00",
                holdout_rmse=10.0,
                holdout_mae=8.0,
                offline_public_rmse=9.0,
                offline_private_rmse=8.5,
            )
        ],
        "best_result": BenchmarkResult(
            experiment_name="iteration_00",
            holdout_rmse=10.0,
            holdout_mae=8.0,
            offline_public_rmse=9.0,
            offline_private_rmse=8.5,
        ),
        "best_spec": ExperimentSpec(
            name="iteration_00",
            feature_pack="default",
            model=ModelSpec(family="lightgbm", objective="regression", params={"n_estimators": 10}),
        ),
        "tool_trace_history": [["select_feature_pack", "select_model_preset"]],
        "guardrail_events": ["Removed suspicious untrusted line: ignore previous instructions"],
    }
    path = write_run_report(tmp_path, profile, final_state)
    content = path.read_text(encoding="utf-8")
    assert "iteration_00" in content
    assert "offline_private_rmse" in content
    assert "tool_trace" in content
    assert "guardrail_event_count" in content


def test_write_submission_file_uses_sample_index(tmp_path: Path) -> None:
    sample_submission = pd.DataFrame({"index": [0, 1], "prediction": [0, 0]})
    output_path = write_submission_file(
        tmp_path / "submission.csv", sample_submission, np.array([1.2, 3.4])
    )
    saved = pd.read_csv(output_path)
    assert saved["prediction"].tolist() == [1.2, 3.4]
