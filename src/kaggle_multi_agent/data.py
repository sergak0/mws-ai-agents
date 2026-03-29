import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

REQUIRED_COMPETITION_FILES = (
    "train.csv",
    "test.csv",
    "sample_submition.csv",
    "solution.csv",
)


@dataclass(frozen=True)
class CompetitionFrames:
    train: pd.DataFrame
    test: pd.DataFrame
    sample_submission: pd.DataFrame
    solution: pd.DataFrame


def validate_competition_files(data_dir: Path) -> dict[str, Path]:
    resolved = {name: data_dir / name for name in REQUIRED_COMPETITION_FILES}
    missing = [name for name, path in resolved.items() if not path.exists()]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise FileNotFoundError(f"Missing competition files: {missing_str}")
    return resolved


def load_competition_frames(data_dir: Path) -> CompetitionFrames:
    files = validate_competition_files(data_dir)
    return CompetitionFrames(
        train=pd.read_csv(files["train.csv"]),
        test=pd.read_csv(files["test.csv"]),
        sample_submission=pd.read_csv(files["sample_submition.csv"]),
        solution=pd.read_csv(files["solution.csv"]),
    )


def load_kaggle_credentials(credentials_path: Path) -> tuple[str, str]:
    payload = json.loads(credentials_path.read_text(encoding="utf-8"))
    return payload["username"], payload["key"]


def download_competition_bundle(
    data_dir: Path,
    competition: str,
    username: str,
    key: str,
) -> list[Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.kaggle.com/api/v1/competitions/data/download-all/{competition}"
    response = requests.get(url, auth=HTTPBasicAuth(username, key), timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        archive.extractall(data_dir)
    files = validate_competition_files(data_dir)
    return [files[name] for name in REQUIRED_COMPETITION_FILES]

