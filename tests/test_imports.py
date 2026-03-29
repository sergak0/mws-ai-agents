from kaggle_multi_agent import __version__
from kaggle_multi_agent.cli import app


def test_package_version_is_exposed() -> None:
    assert __version__ == "0.1.0"


def test_cli_app_exists() -> None:
    assert app.info.name == "kaggle-multi-agent"
