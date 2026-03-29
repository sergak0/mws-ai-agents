from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from kaggle_multi_agent.cli import app

runner = CliRunner()


def _write_competition_files(base: Path) -> None:
    pd.DataFrame(
        {
            "name": ["a", "b", "c", "d"],
            "_id": [1, 2, 3, 4],
            "host_name": ["u1", "u2", "u3", "u4"],
            "location_cluster": ["A", "A", "B", "B"],
            "location": ["x", "y", "x", "z"],
            "lat": [1.0, 2.0, 3.0, 4.0],
            "lon": [10.0, 11.0, 12.0, 13.0],
            "type_house": ["flat", "room", "flat", "room"],
            "sum": [10, 20, 30, 40],
            "min_days": [1, 2, 3, 4],
            "amt_reviews": [0, 1, 2, 3],
            "last_dt": ["2019-01-01"] * 4,
            "avg_reviews": [0.1, 0.2, 0.3, 0.4],
            "total_host": [1, 1, 2, 2],
            "target": [0, 1, 2, 3],
        }
    ).to_csv(base / "train.csv", index=False)
    pd.DataFrame(
        {
            "name": ["e", "f"],
            "_id": [5, 6],
            "host_name": ["u5", "u6"],
            "location_cluster": ["A", "B"],
            "location": ["x", "z"],
            "lat": [5.0, 6.0],
            "lon": [14.0, 15.0],
            "type_house": ["flat", "room"],
            "sum": [50, 60],
            "min_days": [2, 5],
            "amt_reviews": [1, 0],
            "last_dt": ["2019-01-01"] * 2,
            "avg_reviews": [0.5, 0.6],
            "total_host": [2, 1],
        }
    ).to_csv(base / "test.csv", index=False)
    pd.DataFrame({"index": [0, 1], "prediction": [0, 0]}).to_csv(
        base / "sample_submition.csv", index=False
    )
    pd.DataFrame(
        {"index": [0, 1], "prediction": [1, 2], "Usage": ["Public", "Private"]}
    ).to_csv(base / "solution.csv", index=False)


def test_profile_data_command_writes_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_competition_files(data_dir)
    output_path = tmp_path / "profile.json"
    result = runner.invoke(
        app,
        [
            "profile-data",
            "--data-dir",
            str(data_dir),
            "--output-path",
            str(output_path),
        ],
    )
    assert result.exit_code == 0
    assert output_path.exists()


def test_build_kb_command_creates_index(tmp_path: Path) -> None:
    source_dir = tmp_path / "curated"
    source_dir.mkdir()
    (source_dir / "notes.md").write_text("offline metrics and reflection loops", encoding="utf-8")
    output_dir = tmp_path / "index"
    result = runner.invoke(
        app,
        [
            "build-kb",
            "--source-dir",
            str(source_dir),
            "--output-dir",
            str(output_dir),
        ],
    )
    assert result.exit_code == 0
    assert (output_dir / "vectorizer.joblib").exists()

