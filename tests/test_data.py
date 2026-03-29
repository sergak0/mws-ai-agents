import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from kaggle_multi_agent.data import (
    REQUIRED_COMPETITION_FILES,
    download_competition_bundle,
    load_competition_frames,
    load_kaggle_credentials,
    validate_competition_files,
)


def _write_competition_files(base: Path) -> None:
    pd.DataFrame({"feature": [1, 2], "target": [0, 1]}).to_csv(base / "train.csv", index=False)
    pd.DataFrame({"feature": [3, 4]}).to_csv(base / "test.csv", index=False)
    pd.DataFrame({"index": [0, 1], "prediction": [0, 0]}).to_csv(
        base / "sample_submition.csv", index=False
    )
    pd.DataFrame({"index": [0, 1], "prediction": [0, 1], "Usage": ["Public", "Private"]}).to_csv(
        base / "solution.csv", index=False
    )


def test_validate_competition_files_requires_expected_bundle(tmp_path: Path) -> None:
    _write_competition_files(tmp_path)
    resolved = validate_competition_files(tmp_path)
    assert set(resolved) == set(REQUIRED_COMPETITION_FILES)


def test_validate_competition_files_raises_for_missing_input(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_competition_files(tmp_path)


def test_load_competition_frames_reads_expected_shapes(tmp_path: Path) -> None:
    _write_competition_files(tmp_path)
    frames = load_competition_frames(tmp_path)
    assert list(frames.train.columns) == ["feature", "target"]
    assert list(frames.solution.columns) == ["index", "prediction", "Usage"]


def test_load_kaggle_credentials_reads_json_file(tmp_path: Path) -> None:
    credentials_path = tmp_path / "kaggle.json"
    credentials_path.write_text(json.dumps({"username": "demo", "key": "secret"}), encoding="utf-8")
    credentials = load_kaggle_credentials(credentials_path)
    assert credentials == ("demo", "secret")


def test_download_competition_bundle_extracts_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("train.csv", "feature,target\n1,0\n")
        zf.writestr("test.csv", "feature\n2\n")
        zf.writestr("sample_submition.csv", "index,prediction\n0,0\n")
        zf.writestr("solution.csv", "index,prediction,Usage\n0,0,Public\n")

    class Response:
        status_code = 200
        content = archive.getvalue()

        def raise_for_status(self) -> None:
            return None

    def fake_get(*args, **kwargs):
        return Response()

    monkeypatch.setattr("kaggle_multi_agent.data.requests.get", fake_get)
    extracted = download_competition_bundle(
        data_dir=tmp_path,
        competition="demo-competition",
        username="demo",
        key="secret",
    )
    assert sorted(path.name for path in extracted) == sorted(REQUIRED_COMPETITION_FILES)
