import pandas as pd

from kaggle_multi_agent.features import build_feature_bundle


def test_build_feature_bundle_creates_engineered_columns() -> None:
    train = pd.DataFrame(
        {
            "name": ["alpha home"],
            "_id": [1],
            "host_name": ["alice"],
            "location_cluster": ["A"],
            "location": ["loc-1"],
            "lat": [1.0],
            "lon": [2.0],
            "type_house": ["flat"],
            "sum": [100],
            "min_days": [2],
            "amt_reviews": [0],
            "last_dt": ["2019-01-02"],
            "avg_reviews": [0.0],
            "total_host": [1],
            "target": [3],
        }
    )
    test = train.drop(columns=["target"]).copy()
    bundle = build_feature_bundle(train, test, target_column="target")
    assert "name_len" in bundle.train_features.columns
    assert "last_dt_month" in bundle.train_features.columns
    assert "target" not in bundle.train_features.columns
    assert bundle.categorical_columns


def test_build_feature_bundle_can_encode_categorical_columns() -> None:
    train = pd.DataFrame(
        {
            "name": ["alpha home", "beta home"],
            "_id": [1, 2],
            "host_name": ["alice", "bob"],
            "location_cluster": ["A", "B"],
            "location": ["loc-1", "loc-2"],
            "lat": [1.0, 2.0],
            "lon": [2.0, 3.0],
            "type_house": ["flat", "room"],
            "sum": [100, 120],
            "min_days": [2, 3],
            "amt_reviews": [0, 2],
            "last_dt": ["2019-01-02", "2019-02-03"],
            "avg_reviews": [0.0, 1.0],
            "total_host": [1, 2],
            "target": [3, 5],
        }
    )
    test = pd.DataFrame(
        {
            "name": ["gamma home"],
            "_id": [3],
            "host_name": ["carol"],
            "location_cluster": ["C"],
            "location": ["loc-3"],
            "lat": [4.0],
            "lon": [5.0],
            "type_house": ["suite"],
            "sum": [130],
            "min_days": [4],
            "amt_reviews": [1],
            "last_dt": ["2019-03-04"],
            "avg_reviews": [2.0],
            "total_host": [3],
        }
    )
    bundle = build_feature_bundle(
        train,
        test,
        target_column="target",
        feature_pack="encoded_default",
    )
    assert not bundle.categorical_columns
    assert pd.api.types.is_integer_dtype(bundle.train_features["name"])
    assert pd.api.types.is_integer_dtype(bundle.test_features["host_name"])
