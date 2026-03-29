import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer

from kaggle_multi_agent.data import (
    download_competition_bundle,
    load_competition_frames,
    load_kaggle_credentials,
)
from kaggle_multi_agent.graph import run_agent_loop
from kaggle_multi_agent.llm import OpenSourceLLM
from kaggle_multi_agent.modeling import build_baseline_spec, run_experiment
from kaggle_multi_agent.profiling import build_dataset_profile
from kaggle_multi_agent.rag import LocalKnowledgeBase
from kaggle_multi_agent.registry import ExperimentRegistry
from kaggle_multi_agent.reporting import write_run_report, write_submission_file
from kaggle_multi_agent.settings import get_settings
from kaggle_multi_agent.tracking import build_trackers

app = typer.Typer(name="kaggle-multi-agent", add_completion=False, no_args_is_help=True)


@app.command("show-settings")
def show_settings() -> None:
    settings = get_settings()
    typer.echo(settings.model_dump_json(indent=2, exclude_none=True))


@app.command("download-data")
def download_data(
    data_dir: Annotated[Path | None, typer.Option()] = None,
    credentials_path: Annotated[Path, typer.Option()] = Path("kaggle.json"),
    competition: Annotated[str | None, typer.Option()] = None,
) -> None:
    settings = get_settings()
    username, key = load_kaggle_credentials(credentials_path)
    target_dir = data_dir or settings.data_dir
    files = download_competition_bundle(
        data_dir=target_dir,
        competition=competition or settings.kaggle_competition,
        username=username,
        key=key,
    )
    typer.echo(json.dumps([str(path) for path in files], indent=2))


@app.command("profile-data")
def profile_data(
    data_dir: Annotated[Path | None, typer.Option()] = None,
    output_path: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = get_settings()
    frames = load_competition_frames(data_dir or settings.data_dir)
    profile = build_dataset_profile(frames.train, frames.test, target_column=settings.target_column)
    destination = output_path or settings.reports_dir / "data_profile.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(str(destination))


@app.command("build-kb")
def build_kb(
    source_dir: Annotated[Path | None, typer.Option()] = None,
    output_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = get_settings()
    kb = LocalKnowledgeBase.build(source_dir or settings.knowledge_base_dir / "curated")
    destination = output_dir or settings.knowledge_base_dir / "index"
    kb.save(destination)
    typer.echo(str(destination))


@app.command("benchmark")
def benchmark(
    data_dir: Annotated[Path | None, typer.Option()] = None,
    output_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    settings = get_settings()
    frames = load_competition_frames(data_dir or settings.data_dir)
    run_dir = _build_run_dir(output_dir or settings.artifacts_dir / "benchmark")
    profile = build_dataset_profile(frames.train, frames.test, target_column=settings.target_column)
    spec = build_baseline_spec()
    run = run_experiment(
        spec=spec,
        frames=frames,
        target_column=settings.target_column,
        random_seed=settings.random_seed,
    )
    registry = ExperimentRegistry(run_dir / "registry.jsonl")
    registry.append(run.result)
    write_submission_file(run_dir / "submission.csv", frames.sample_submission, run.predictions)
    report_path = write_run_report(
        run_dir,
        profile,
        {"history": [run.result], "best_result": run.result, "best_spec": spec},
    )
    typer.echo(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "report_path": str(report_path),
                "holdout_rmse": run.result.holdout_rmse,
            },
            indent=2,
        )
    )


@app.command("run-agent")
def run_agent(
    data_dir: Annotated[Path | None, typer.Option()] = None,
    knowledge_source_dir: Annotated[Path | None, typer.Option()] = None,
    output_dir: Annotated[Path | None, typer.Option()] = None,
    max_iterations: Annotated[int, typer.Option()] = 3,
) -> None:
    settings = get_settings()
    frames = load_competition_frames(data_dir or settings.data_dir)
    run_dir = _build_run_dir(output_dir or settings.artifacts_dir / "runs")
    profile = build_dataset_profile(frames.train, frames.test, target_column=settings.target_column)
    kb = LocalKnowledgeBase.build(knowledge_source_dir or settings.knowledge_base_dir / "curated")
    kb.save(run_dir / "knowledge_index")
    planner_llm = None
    critic_llm = None
    if settings.llm_provider != "mock":
        planner_llm = OpenSourceLLM(provider=settings.llm_provider, model=settings.planner_model)
        critic_llm = OpenSourceLLM(provider=settings.llm_provider, model=settings.critic_model)
    trackers = build_trackers(
        base_path=run_dir,
        wandb_enabled=settings.wandb_enabled,
        wandb_project=settings.wandb_project,
        wandb_entity=settings.wandb_entity,
        run_name=run_dir.name,
    )
    final_state = run_agent_loop(
        frames=frames,
        knowledge_base=kb,
        target_column=settings.target_column,
        max_iterations=max_iterations,
        random_seed=settings.random_seed,
        planner_llm=planner_llm,
        critic_llm=critic_llm,
        trackers=trackers,
    )
    registry = ExperimentRegistry(run_dir / "registry.jsonl")
    for result in final_state["history"]:
        registry.append(result)
    write_submission_file(
        run_dir / "submission.csv", frames.sample_submission, final_state["best_predictions"]
    )
    report_path = write_run_report(run_dir, profile, final_state)
    for tracker in trackers:
        tracker.finish()
    typer.echo(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "report_path": str(report_path),
                "best_experiment": final_state["best_result"].experiment_name,
                "offline_private_rmse": final_state["best_result"].offline_private_rmse,
            },
            indent=2,
        )
    )


def main() -> None:
    app()


def _build_run_dir(base_dir: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


if __name__ == "__main__":
    main()
