import pandas as pd

from kaggle_multi_agent.contracts import DatasetProfile


def build_dataset_profile(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
) -> DatasetProfile:
    feature_columns = [column for column in train_df.columns if column != target_column]
    target = train_df[target_column]
    return DatasetProfile(
        train_rows=len(train_df),
        test_rows=len(test_df),
        feature_columns=feature_columns,
        target_column=target_column,
        target_min=float(target.min()),
        target_max=float(target.max()),
        target_unique=int(target.nunique(dropna=True)),
    )
