import json
from pathlib import Path

from kaggle_multi_agent.contracts import BenchmarkResult, CritiqueResult, ExperimentSpec


class JsonlTracker:
    def __init__(self, path: Path) -> None:
        self.path = path

    def log_iteration(
        self,
        iteration: int,
        best_so_far: bool,
        spec: ExperimentSpec,
        result: BenchmarkResult,
        critique: CritiqueResult,
        tool_trace: list[str],
        guardrail_events: list[str],
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "iteration": iteration,
            "best_so_far": best_so_far,
            "spec": spec.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
            "critique": critique.model_dump(mode="json"),
            "tool_trace": tool_trace,
            "guardrail_events": guardrail_events,
        }
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload))
            file.write("\n")

    def finish(self) -> None:
        return None


class WandbTracker:
    def __init__(
        self,
        project: str,
        entity: str | None,
        run_name: str,
        config: dict[str, str | int | float | bool],
    ) -> None:
        import wandb

        self._run = wandb.init(project=project, entity=entity, name=run_name, config=config)

    def log_iteration(
        self,
        iteration: int,
        best_so_far: bool,
        spec: ExperimentSpec,
        result: BenchmarkResult,
        critique: CritiqueResult,
        tool_trace: list[str],
        guardrail_events: list[str],
    ) -> None:
        self._run.log(
            {
                "iteration": iteration,
                "best_so_far": best_so_far,
                "holdout_rmse": result.holdout_rmse,
                "holdout_mae": result.holdout_mae,
                "offline_public_rmse": result.offline_public_rmse,
                "offline_private_rmse": result.offline_private_rmse,
                "feature_pack": spec.feature_pack,
                "seed_ensemble_size": spec.seed_ensemble_size,
                "tool_trace": tool_trace,
                "guardrail_event_count": len(guardrail_events),
                "critique_decision": critique.decision,
            }
        )

    def finish(self) -> None:
        self._run.finish()


def build_trackers(
    base_path: Path,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str | None,
    run_name: str,
) -> list[JsonlTracker | WandbTracker]:
    trackers: list[JsonlTracker | WandbTracker] = [
        JsonlTracker(base_path / "iteration_events.jsonl")
    ]
    if wandb_enabled:
        try:
            trackers.append(
                WandbTracker(
                    project=wandb_project,
                    entity=wandb_entity,
                    run_name=run_name,
                    config={},
                )
            )
        except ModuleNotFoundError:
            pass
    return trackers
