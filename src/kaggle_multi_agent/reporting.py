from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kaggle_multi_agent.contracts import DatasetProfile


def write_run_report(
    report_dir: Path, profile: DatasetProfile, final_state: dict[str, Any]
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    output_path = report_dir / f"run_report_{timestamp}.md"
    best_result = final_state["best_result"]
    best_spec = final_state["best_spec"]
    history = final_state["history"]
    tool_trace_history = final_state.get("tool_trace_history", [])
    guardrail_events = final_state.get("guardrail_events", [])
    best_tool_trace = _find_best_tool_trace(
        best_experiment_name=best_result.experiment_name,
        history=history,
        tool_trace_history=tool_trace_history,
    )
    lines = [
        "# Run Report",
        "",
        "## Dataset",
        f"- train_rows: {profile.train_rows}",
        f"- test_rows: {profile.test_rows}",
        f"- target_column: {profile.target_column}",
        "",
        "## Best Experiment",
        f"- name: {best_result.experiment_name}",
        f"- holdout_rmse: {best_result.holdout_rmse}",
        f"- holdout_mae: {best_result.holdout_mae}",
        f"- offline_public_rmse: {best_result.offline_public_rmse}",
        f"- offline_private_rmse: {best_result.offline_private_rmse}",
        f"- feature_pack: {best_spec.feature_pack}",
        f"- seed_ensemble_size: {best_spec.seed_ensemble_size}",
        f"- model_family: {best_spec.model.family}",
        f"- tool_trace: {', '.join(best_tool_trace) if best_tool_trace else 'n/a'}",
        "",
        "## Retrieved Context",
        f"- retrieved_context_items: {len(best_spec.retrieved_context)}",
        f"- hypotheses: {len(best_spec.hypotheses)}",
        "",
        "## Guardrails",
        f"- guardrail_event_count: {len(guardrail_events)}",
    ]
    if guardrail_events:
        lines.append("")
        for event in guardrail_events:
            lines.append(f"- {event}")
    lines.extend(
        [
            "",
            "## Iteration Trace",
        ]
    )
    for index, item in enumerate(history):
        tool_trace = tool_trace_history[index] if index < len(tool_trace_history) else []
        lines.append(
            f"- {item.experiment_name}: tool_trace={tool_trace}, "
            f"holdout_rmse={item.holdout_rmse}, "
            f"offline_private_rmse={item.offline_private_rmse}"
        )
    lines.extend(
        [
            "",
            "## History",
        ]
    )
    for item in history:
        lines.append(
            f"- {item.experiment_name}: holdout_rmse={item.holdout_rmse}, "
            f"offline_private_rmse={item.offline_private_rmse}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _find_best_tool_trace(
    best_experiment_name: str,
    history: list[Any],
    tool_trace_history: list[list[str]],
) -> list[str]:
    for index, item in enumerate(history):
        if item.experiment_name == best_experiment_name and index < len(tool_trace_history):
            return tool_trace_history[index]
    return []


def write_submission_file(
    output_path: Path,
    sample_submission: pd.DataFrame,
    predictions: np.ndarray,
) -> Path:
    frame = sample_submission.copy()
    frame["prediction"] = np.asarray(predictions, dtype=float)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path
