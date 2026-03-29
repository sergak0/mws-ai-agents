import numpy as np
import pandas as pd

from kaggle_multi_agent.metrics import (
    clip_predictions,
    compute_offline_metrics,
    compute_regression_metrics,
)


def test_clip_predictions_enforces_bounds() -> None:
    clipped = clip_predictions(np.array([-1.0, 10.0, 400.0]), lower=0, upper=365)
    assert clipped.tolist() == [0.0, 10.0, 365.0]


def test_compute_regression_metrics_returns_rmse_and_mae() -> None:
    metrics = compute_regression_metrics(np.array([0, 1]), np.array([0, 3]))
    assert round(metrics["rmse"], 6) == round((2.0**0.5), 6)
    assert metrics["mae"] == 1.0


def test_compute_offline_metrics_splits_public_and_private() -> None:
    solution = pd.DataFrame(
        {
            "index": [0, 1, 2, 3],
            "prediction": [0, 1, 2, 3],
            "Usage": ["Public", "Public", "Private", "Private"],
        }
    )
    metrics = compute_offline_metrics(solution, np.array([0, 2, 2, 5]))
    assert metrics["public_rmse"] >= 0
    assert metrics["private_rmse"] >= 0

