from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureBundle:
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    target: pd.Series
    categorical_columns: list[str]


def build_feature_bundle(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str,
    feature_pack: str = "default",
) -> FeatureBundle:
    train_features = train_df.drop(columns=[target_column]).copy()
    test_features = test_df.copy()
    target = train_df[target_column].copy()

    train_features = _engineer_features(train_features)
    test_features = _engineer_features(test_features)
    if feature_pack in {"encoded_geo", "encoded_geo_interactions"}:
        train_features = _apply_geo_features(train_features)
        test_features = _apply_geo_features(test_features)
    if feature_pack == "encoded_geo_interactions":
        train_features = _apply_interaction_features(train_features)
        test_features = _apply_interaction_features(test_features)

    categorical_columns = [
        column
        for column in train_features.columns
        if str(train_features[column].dtype) == "category"
    ]

    aligned_test = test_features.reindex(columns=train_features.columns)
    if feature_pack in {"encoded_default", "encoded_geo", "encoded_geo_interactions"}:
        train_features, aligned_test = _encode_categorical_columns(
            train_features,
            aligned_test,
            categorical_columns,
        )
        if feature_pack in {"encoded_geo", "encoded_geo_interactions"}:
            train_features, aligned_test = _add_frequency_features(train_features, aligned_test)
        categorical_columns = []
    elif feature_pack != "default":
        raise ValueError(f"Unsupported feature pack: {feature_pack}")
    return FeatureBundle(
        train_features=train_features,
        test_features=aligned_test,
        target=target,
        categorical_columns=categorical_columns,
    )


def _engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    dataset = frame.copy()
    dataset["name"] = dataset.get("name", "").fillna("").astype(str)
    dataset["host_name"] = dataset.get("host_name", "").fillna("").astype(str)
    dataset["name_len"] = dataset["name"].str.len()
    dataset["host_name_len"] = dataset["host_name"].str.len()
    dataset["has_last_dt"] = dataset["last_dt"].notna().astype(int)
    last_dt = pd.to_datetime(dataset["last_dt"], errors="coerce")
    dataset["last_dt_year"] = last_dt.dt.year.fillna(0).astype(int)
    dataset["last_dt_month"] = last_dt.dt.month.fillna(0).astype(int)
    dataset["last_dt_day"] = last_dt.dt.day.fillna(0).astype(int)
    dataset["last_dt_dow"] = last_dt.dt.dayofweek.fillna(0).astype(int)
    dataset["sum_log1p"] = np.log1p(dataset["sum"].fillna(0).astype(float))
    dataset["reviews_log1p"] = np.log1p(dataset["amt_reviews"].fillna(0).astype(float))
    dataset["price_per_review"] = dataset["sum"].fillna(0).astype(float) / (
        dataset["amt_reviews"].fillna(0).astype(float) + 1.0
    )
    dataset["avg_reviews"] = dataset["avg_reviews"].fillna(0.0)
    dataset["last_dt"] = dataset["last_dt"].fillna("")

    categorical_columns = [
        column
        for column in ["name", "host_name", "location_cluster", "location", "type_house", "last_dt"]
        if column in dataset.columns
    ]
    for column in categorical_columns:
        dataset[column] = dataset[column].fillna("__missing__").astype("category")

    return dataset


def _apply_geo_features(frame: pd.DataFrame) -> pd.DataFrame:
    dataset = frame.copy()
    lat = dataset["lat"].fillna(0.0).astype(float)
    lon = dataset["lon"].fillna(0.0).astype(float)
    dataset["geo_bin_1"] = (
        np.floor(lat * 10).astype(int).astype(str)
        + "_"
        + np.floor(lon * 10).astype(int).astype(str)
    ).astype("category")
    dataset["geo_bin_2"] = (
        np.floor(lat * 100).astype(int).astype(str)
        + "_"
        + np.floor(lon * 100).astype(int).astype(str)
    ).astype("category")
    return dataset


def _apply_interaction_features(frame: pd.DataFrame) -> pd.DataFrame:
    dataset = frame.copy()
    dataset["avg_reviews_missing"] = dataset["avg_reviews"].fillna(0).eq(0).astype(int)
    dataset["review_density"] = dataset["amt_reviews"].fillna(0).astype(float) / (
        dataset["total_host"].fillna(0).astype(float) + 1.0
    )
    dataset["sum_per_min_day"] = dataset["sum"].fillna(0).astype(float) / (
        dataset["min_days"].fillna(0).astype(float) + 1.0
    )
    dataset["sum_x_reviews"] = dataset["sum"].fillna(0).astype(float) * np.log1p(
        dataset["amt_reviews"].fillna(0).astype(float)
    )
    return dataset


def _add_frequency_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    updated_train = train_features.copy()
    updated_test = test_features.copy()
    for column in [
        column for column in ("geo_bin_1", "geo_bin_2") if column in updated_train.columns
    ]:
        value_counts = pd.concat(
            [updated_train[column], updated_test[column]],
            ignore_index=True,
        ).value_counts()
        updated_train[f"{column}_freq"] = updated_train[column].map(value_counts).astype(float)
        updated_test[f"{column}_freq"] = updated_test[column].map(value_counts).astype(float)
    return updated_train, updated_test


def _encode_categorical_columns(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    encoded_train = train_features.copy()
    encoded_test = test_features.copy()
    for column in categorical_columns:
        combined = pd.concat(
            [
                encoded_train[column].astype(str),
                encoded_test[column].astype(str),
            ],
            axis=0,
            ignore_index=True,
        )
        categories = pd.Categorical(combined)
        encoded_train[column] = categories.codes[: len(encoded_train)]
        encoded_test[column] = categories.codes[len(encoded_train) :]
    return encoded_train, encoded_test
