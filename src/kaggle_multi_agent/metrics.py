import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def clip_predictions(predictions: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip(np.asarray(predictions, dtype=float), lower, upper)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true_values = np.asarray(y_true, dtype=float)
    predicted_values = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(mean_squared_error(true_values, predicted_values) ** 0.5),
        "mae": float(mean_absolute_error(true_values, predicted_values)),
    }


def compute_offline_metrics(solution_df: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
    predicted_values = np.asarray(predictions, dtype=float)
    if len(solution_df) != len(predicted_values):
        raise ValueError("Predictions length must match solution length")
    public_mask = solution_df["Usage"].eq("Public")
    private_mask = solution_df["Usage"].eq("Private")
    public_metrics = compute_regression_metrics(
        solution_df.loc[public_mask, "prediction"].to_numpy(),
        predicted_values[public_mask.to_numpy()],
    )
    private_metrics = compute_regression_metrics(
        solution_df.loc[private_mask, "prediction"].to_numpy(),
        predicted_values[private_mask.to_numpy()],
    )
    return {
        "public_rmse": public_metrics["rmse"],
        "public_mae": public_metrics["mae"],
        "private_rmse": private_metrics["rmse"],
        "private_mae": private_metrics["mae"],
    }
