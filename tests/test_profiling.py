
import pandas as pd

from kaggle_multi_agent.profiling import build_dataset_profile


def test_build_dataset_profile_detects_target_range() -> None:
    train = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            "category": ["a", "b", "a"],
            "target": [0, 10, 365],
        }
    )
    test = pd.DataFrame({"feature": [4, 5], "category": ["a", "c"]})
    profile = build_dataset_profile(train, test, target_column="target")
    assert profile.train_rows == 3
    assert profile.test_rows == 2
    assert profile.target_min == 0
    assert profile.target_max == 365
    assert profile.target_unique == 3

