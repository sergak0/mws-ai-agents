from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from kaggle_multi_agent.cli import app

runner = CliRunner()


def _write_competition_files(base: Path) -> None:
    rows = 20
    pd.DataFrame(
        {
            "name": [f"name-{idx}" for idx in range(rows)],
            "_id": list(range(rows)),
            "host_name": [f"host-{idx % 4}" for idx in range(rows)],
            "location_cluster": ["A" if idx % 2 == 0 else "B" for idx in range(rows)],
            "location": [f"loc-{idx % 3}" for idx in range(rows)],
            "lat": [float(idx) for idx in range(rows)],
            "lon": [10.0 + float(idx) for idx in range(rows)],
            "type_house": ["flat" if idx % 2 == 0 else "room" for idx in range(rows)],
            "sum": [50 + idx for idx in range(rows)],
            "min_days": [1 + idx % 5 for idx in range(rows)],
            "amt_reviews": [idx % 7 for idx in range(rows)],
            "last_dt": ["2019-01-02"] * rows,
            "avg_reviews": [0.1 + idx * 0.05 for idx in range(rows)],
            "total_host": [1 + idx % 3 for idx in range(rows)],
            "target": [idx % 12 for idx in range(rows)],
        }
    ).to_csv(base / "train.csv", index=False)
    pd.DataFrame(
        {
            "name": ["a", "b", "c", "d"],
            "_id": [100, 101, 102, 103],
            "host_name": ["u1", "u2", "u3", "u4"],
            "location_cluster": ["A", "B", "A", "B"],
            "location": ["loc-0", "loc-1", "loc-2", "loc-0"],
            "lat": [1.0, 2.0, 3.0, 4.0],
            "lon": [11.0, 12.0, 13.0, 14.0],
            "type_house": ["flat", "room", "flat", "room"],
            "sum": [60, 70, 80, 90],
            "min_days": [2, 3, 4, 5],
            "amt_reviews": [1, 2, 3, 4],
            "last_dt": ["2019-01-02"] * 4,
            "avg_reviews": [0.2, 0.4, 0.6, 0.8],
            "total_host": [1, 2, 1, 2],
        }
    ).to_csv(base / "test.csv", index=False)
    pd.DataFrame({"index": [0, 1, 2, 3], "prediction": [0, 0, 0, 0]}).to_csv(
        base / "sample_submition.csv", index=False
    )
    pd.DataFrame(
        {
            "index": [0, 1, 2, 3],
            "prediction": [1, 2, 3, 4],
            "Usage": ["Public", "Public", "Private", "Private"],
        }
    ).to_csv(base / "solution.csv", index=False)


def test_run_agent_command_generates_submission_and_report(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    kb_dir = tmp_path / "knowledge"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    kb_dir.mkdir()
    _write_competition_files(data_dir)
    (kb_dir / "notes.md").write_text(
        "Use offline public and private metrics. Reflection loops improve agents.",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "run-agent",
            "--data-dir",
            str(data_dir),
            "--knowledge-source-dir",
            str(kb_dir),
            "--output-dir",
            str(output_dir),
            "--max-iterations",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert list(output_dir.rglob("submission.csv"))
    assert list(output_dir.rglob("run_report_*.md"))

