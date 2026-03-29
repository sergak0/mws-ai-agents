import numpy as np
import pandas as pd

from kaggle_multi_agent.contracts import ExperimentSpec, ModelSpec
from kaggle_multi_agent.data import CompetitionFrames
from kaggle_multi_agent.modeling import build_baseline_spec, run_experiment


def test_run_experiment_returns_metrics_and_test_predictions() -> None:
    rows = 40
    train = pd.DataFrame(
        {
            "name": [f"name-{idx % 5}" for idx in range(rows)],
            "_id": list(range(rows)),
            "host_name": [f"host-{idx % 4}" for idx in range(rows)],
            "location_cluster": ["A" if idx % 2 == 0 else "B" for idx in range(rows)],
            "location": [f"loc-{idx % 3}" for idx in range(rows)],
            "lat": np.linspace(1.0, 4.0, rows),
            "lon": np.linspace(10.0, 12.0, rows),
            "type_house": ["flat" if idx % 2 == 0 else "room" for idx in range(rows)],
            "sum": np.linspace(50, 150, rows),
            "min_days": [1 + idx % 5 for idx in range(rows)],
            "amt_reviews": [idx % 7 for idx in range(rows)],
            "last_dt": ["2019-01-02"] * rows,
            "avg_reviews": np.linspace(0.1, 3.0, rows),
            "total_host": [1 + idx % 3 for idx in range(rows)],
            "target": [idx % 30 for idx in range(rows)],
        }
    )
    test = train.drop(columns=["target"]).head(8).copy()
    solution = pd.DataFrame(
        {
            "index": list(range(len(test))),
            "prediction": [idx % 10 for idx in range(len(test))],
            "Usage": ["Public"] * 4 + ["Private"] * 4,
        }
    )
    sample = pd.DataFrame({"index": list(range(len(test))), "prediction": [0] * len(test)})
    spec = ExperimentSpec(
        name="baseline",
        feature_pack="default",
        model=ModelSpec(
            family="lightgbm",
            objective="regression",
            params={"n_estimators": 20, "learning_rate": 0.1, "num_leaves": 15},
        ),
    )
    frames = CompetitionFrames(
        train=train,
        test=test,
        sample_submission=sample,
        solution=solution,
    )
    run = run_experiment(spec, frames, target_column="target", random_seed=42)
    assert run.result.holdout_rmse >= 0
    assert len(run.predictions) == len(test)
    assert run.result.offline_public_rmse is not None


def test_run_experiment_supports_encoded_feature_pack() -> None:
    rows = 40
    train = pd.DataFrame(
        {
            "name": [f"name-{idx % 5}" for idx in range(rows)],
            "_id": list(range(rows)),
            "host_name": [f"host-{idx % 4}" for idx in range(rows)],
            "location_cluster": ["A" if idx % 2 == 0 else "B" for idx in range(rows)],
            "location": [f"loc-{idx % 3}" for idx in range(rows)],
            "lat": np.linspace(1.0, 4.0, rows),
            "lon": np.linspace(10.0, 12.0, rows),
            "type_house": ["flat" if idx % 2 == 0 else "room" for idx in range(rows)],
            "sum": np.linspace(50, 150, rows),
            "min_days": [1 + idx % 5 for idx in range(rows)],
            "amt_reviews": [idx % 7 for idx in range(rows)],
            "last_dt": ["2019-01-02"] * rows,
            "avg_reviews": np.linspace(0.1, 3.0, rows),
            "total_host": [1 + idx % 3 for idx in range(rows)],
            "target": [idx % 30 for idx in range(rows)],
        }
    )
    test = train.drop(columns=["target"]).head(8).copy()
    solution = pd.DataFrame(
        {
            "index": list(range(len(test))),
            "prediction": [idx % 10 for idx in range(len(test))],
            "Usage": ["Public"] * 4 + ["Private"] * 4,
        }
    )
    sample = pd.DataFrame({"index": list(range(len(test))), "prediction": [0] * len(test)})
    spec = ExperimentSpec(
        name="encoded",
        feature_pack="encoded_default",
        model=ModelSpec(
            family="lightgbm",
            objective="regression",
            params={"n_estimators": 20, "learning_rate": 0.1, "num_leaves": 15},
        ),
    )
    frames = CompetitionFrames(
        train=train,
        test=test,
        sample_submission=sample,
        solution=solution,
    )
    run = run_experiment(spec, frames, target_column="target", random_seed=42)
    assert run.result.holdout_rmse >= 0
    assert len(run.predictions) == len(test)
    assert run.feature_importances


def test_build_baseline_spec_uses_encoded_feature_pack() -> None:
    spec = build_baseline_spec()
    assert spec.feature_pack == "encoded_default"


def test_run_experiment_supports_seed_ensemble() -> None:
    rows = 40
    train = pd.DataFrame(
        {
            "name": [f"name-{idx % 5}" for idx in range(rows)],
            "_id": list(range(rows)),
            "host_name": [f"host-{idx % 4}" for idx in range(rows)],
            "location_cluster": ["A" if idx % 2 == 0 else "B" for idx in range(rows)],
            "location": [f"loc-{idx % 3}" for idx in range(rows)],
            "lat": np.linspace(1.0, 4.0, rows),
            "lon": np.linspace(10.0, 12.0, rows),
            "type_house": ["flat" if idx % 2 == 0 else "room" for idx in range(rows)],
            "sum": np.linspace(50, 150, rows),
            "min_days": [1 + idx % 5 for idx in range(rows)],
            "amt_reviews": [idx % 7 for idx in range(rows)],
            "last_dt": ["2019-01-02"] * rows,
            "avg_reviews": np.linspace(0.1, 3.0, rows),
            "total_host": [1 + idx % 3 for idx in range(rows)],
            "target": [idx % 30 for idx in range(rows)],
        }
    )
    test = train.drop(columns=["target"]).head(8).copy()
    solution = pd.DataFrame(
        {
            "index": list(range(len(test))),
            "prediction": [idx % 10 for idx in range(len(test))],
            "Usage": ["Public"] * 4 + ["Private"] * 4,
        }
    )
    sample = pd.DataFrame({"index": list(range(len(test))), "prediction": [0] * len(test)})
    spec = ExperimentSpec(
        name="ensemble",
        feature_pack="encoded_default",
        model=ModelSpec(
            family="lightgbm",
            objective="regression",
            params={"n_estimators": 20, "learning_rate": 0.1, "num_leaves": 15},
        ),
        seed_ensemble_size=3,
    )
    frames = CompetitionFrames(
        train=train,
        test=test,
        sample_submission=sample,
        solution=solution,
    )
    run = run_experiment(spec, frames, target_column="target", random_seed=42)
    assert run.result.holdout_rmse >= 0
    assert len(run.predictions) == len(test)
