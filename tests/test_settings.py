from pathlib import Path

from kaggle_multi_agent.settings import Settings


def test_settings_resolve_workspace_paths() -> None:
    settings = Settings(project_root=Path("/tmp/demo-root"))
    assert settings.data_dir == Path("/tmp/demo-root/data")
    assert settings.artifacts_dir == Path("/tmp/demo-root/artifacts")
    assert settings.reports_dir == Path("/tmp/demo-root/reports")
    assert settings.knowledge_base_dir == Path("/tmp/demo-root/knowledge_base")


def test_settings_allow_provider_override() -> None:
    settings = Settings(llm_provider="openrouter")
    assert settings.llm_provider == "openrouter"


def test_settings_allow_wandb_configuration() -> None:
    settings = Settings(wandb_enabled=True, wandb_project="demo-project")
    assert settings.wandb_enabled is True
    assert settings.wandb_project == "demo-project"
