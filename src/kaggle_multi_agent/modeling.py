from dataclasses import dataclass

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split

from kaggle_multi_agent.contracts import BenchmarkResult, ExperimentSpec, ModelSpec
from kaggle_multi_agent.data import CompetitionFrames
from kaggle_multi_agent.features import build_feature_bundle
from kaggle_multi_agent.metrics import (
    clip_predictions,
    compute_offline_metrics,
    compute_regression_metrics,
)


@dataclass(frozen=True)
class ExperimentRun:
    spec: ExperimentSpec
    result: BenchmarkResult
    predictions: np.ndarray
    feature_importances: dict[str, float]


def build_baseline_spec(name: str = "baseline") -> ExperimentSpec:
    return ExperimentSpec(
        name=name,
        feature_pack="encoded_default",
        model=ModelSpec(
            family="lightgbm",
            objective="regression",
            params={
                "n_estimators": 500,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
            },
        ),
    )


def run_experiment(
    spec: ExperimentSpec,
    frames: CompetitionFrames,
    target_column: str,
    random_seed: int,
) -> ExperimentRun:
    bundle = build_feature_bundle(
        frames.train,
        frames.test,
        target_column=target_column,
        feature_pack=spec.feature_pack,
    )
    ensemble_seeds = _resolve_ensemble_seeds(random_seed, spec.seed_ensemble_size)
    holdout_metrics_history: list[dict[str, float]] = []
    prediction_history: list[np.ndarray] = []
    importance_history: list[np.ndarray] = []

    for ensemble_seed in ensemble_seeds:
        X_train, X_valid, y_train, y_valid = train_test_split(
            bundle.train_features,
            bundle.target,
            test_size=0.2,
            random_state=ensemble_seed,
        )
        model = _build_model(spec, random_seed=ensemble_seed)
        _fit_model(model, X_train, y_train, bundle.categorical_columns)
        holdout_predictions = clip_predictions(
            model.predict(X_valid),
            lower=spec.prediction_min,
            upper=spec.prediction_max,
        )
        holdout_metrics_history.append(
            compute_regression_metrics(y_valid.to_numpy(), holdout_predictions)
        )

        final_model = _build_model(spec, random_seed=ensemble_seed)
        _fit_model(final_model, bundle.train_features, bundle.target, bundle.categorical_columns)
        prediction_history.append(
            clip_predictions(
                final_model.predict(bundle.test_features),
                lower=spec.prediction_min,
                upper=spec.prediction_max,
            )
        )
        importance_history.append(final_model.feature_importances_.astype(float))

    holdout_metrics = {
        "rmse": float(np.mean([item["rmse"] for item in holdout_metrics_history])),
        "mae": float(np.mean([item["mae"] for item in holdout_metrics_history])),
    }
    predictions = clip_predictions(
        np.mean(prediction_history, axis=0),
        lower=spec.prediction_min,
        upper=spec.prediction_max,
    )
    offline_metrics = compute_offline_metrics(frames.solution, predictions)
    result = BenchmarkResult(
        experiment_name=spec.name,
        holdout_rmse=holdout_metrics["rmse"],
        holdout_mae=holdout_metrics["mae"],
        offline_public_rmse=offline_metrics["public_rmse"],
        offline_private_rmse=offline_metrics["private_rmse"],
    )
    mean_importances = np.mean(importance_history, axis=0)
    feature_importances = {
        column: float(value)
        for column, value in zip(
            bundle.train_features.columns, mean_importances, strict=True
        )
    }
    return ExperimentRun(
        spec=spec,
        result=result,
        predictions=predictions,
        feature_importances=feature_importances,
    )


def _build_model(spec: ExperimentSpec, random_seed: int) -> LGBMRegressor:
    params = {"objective": "regression", "random_state": random_seed, "verbose": -1}
    params.update(spec.model.params)
    return LGBMRegressor(**params)


def _fit_model(
    model: LGBMRegressor,
    features,
    target,
    categorical_columns: list[str],
) -> None:
    if categorical_columns:
        model.fit(features, target, categorical_feature=categorical_columns)
        return
    model.fit(features, target)


def _resolve_ensemble_seeds(random_seed: int, ensemble_size: int) -> list[int]:
    seeds = [random_seed]
    offsets = (-29, 35, 59, 89, 113)
    for offset in offsets:
        candidate = max(1, random_seed + offset)
        if candidate not in seeds:
            seeds.append(candidate)
        if len(seeds) >= ensemble_size:
            return seeds[:ensemble_size]
    while len(seeds) < ensemble_size:
        seeds.append(seeds[-1] + 17)
    return seeds[:ensemble_size]
