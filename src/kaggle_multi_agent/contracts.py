from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DatasetProfile(BaseModel):
    train_rows: int = Field(gt=0)
    test_rows: int = Field(gt=0)
    feature_columns: list[str]
    target_column: str
    target_min: float
    target_max: float
    target_unique: int = Field(gt=1)


class ModelSpec(BaseModel):
    family: Literal["lightgbm"]
    objective: Literal["regression"]
    params: dict[str, int | float | str | bool]


class ExperimentSpec(BaseModel):
    name: str = Field(min_length=3)
    feature_pack: str = Field(min_length=1)
    model: ModelSpec
    use_cross_validation: bool = False
    seed_ensemble_size: int = Field(default=1, ge=1)
    prediction_min: int = 0
    prediction_max: int = 365
    retrieved_context: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_bounds(self) -> "ExperimentSpec":
        if self.prediction_min >= self.prediction_max:
            raise ValueError("prediction_min must be lower than prediction_max")
        return self


class ResearchBrief(BaseModel):
    retrieved_chunks: list[str] = Field(default_factory=list)
    feature_ideas: list[str] = Field(default_factory=list)
    model_ideas: list[str] = Field(default_factory=list)
    guardrail_events: list[str] = Field(default_factory=list)


class ToolCall(BaseModel):
    name: str = Field(min_length=1)
    arguments: dict[str, str | int | float | bool] = Field(default_factory=dict)


class ToolPlan(BaseModel):
    steps: list[ToolCall] = Field(min_length=1)


class BenchmarkResult(BaseModel):
    experiment_name: str
    holdout_rmse: float = Field(ge=0)
    holdout_mae: float = Field(ge=0)
    offline_public_rmse: float | None = Field(default=None, ge=0)
    offline_private_rmse: float | None = Field(default=None, ge=0)
    is_best: bool = False


class CritiqueResult(BaseModel):
    decision: Literal["accept", "retry", "stop"]
    summary: str = Field(min_length=1)
    actions: list[str] = Field(default_factory=list)
