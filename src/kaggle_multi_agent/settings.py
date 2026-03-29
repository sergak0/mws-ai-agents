from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="KMA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path | None = None
    artifacts_dir: Path | None = None
    reports_dir: Path | None = None
    knowledge_base_dir: Path | None = None
    kaggle_competition: str = "mws-ai-agents-2026"
    target_column: str = "target"
    task_type: Literal["bounded_regression"] = "bounded_regression"
    llm_provider: Literal["mock", "ollama", "openrouter"] = "mock"
    planner_model: str = "qwen/qwen2.5-coder-14b-instruct"
    critic_model: str = "qwen/qwen2.5-14b-instruct"
    max_iterations: int = 3
    prediction_min: int = 0
    prediction_max: int = 365
    random_seed: int = 42
    wandb_enabled: bool = False
    wandb_project: str = "kaggle-multi-agent"
    wandb_entity: str | None = None

    @model_validator(mode="after")
    def resolve_paths(self) -> "Settings":
        root = self.project_root
        if self.data_dir is None:
            object.__setattr__(self, "data_dir", root / "data")
        if self.artifacts_dir is None:
            object.__setattr__(self, "artifacts_dir", root / "artifacts")
        if self.reports_dir is None:
            object.__setattr__(self, "reports_dir", root / "reports")
        if self.knowledge_base_dir is None:
            object.__setattr__(self, "knowledge_base_dir", root / "knowledge_base")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
