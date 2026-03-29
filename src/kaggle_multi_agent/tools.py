from dataclasses import dataclass

from kaggle_multi_agent.contracts import ExperimentSpec, ModelSpec, ToolPlan


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    allowed_arguments: dict[str, tuple[str | int, ...]]


def build_tool_catalog() -> dict[str, ToolDefinition]:
    return {
        "select_feature_pack": ToolDefinition(
            name="select_feature_pack",
            description="Select one of the supported feature packs.",
            allowed_arguments={
                "feature_pack": (
                    "encoded_default",
                    "encoded_geo",
                    "encoded_geo_interactions",
                )
            },
        ),
        "select_model_preset": ToolDefinition(
            name="select_model_preset",
            description="Select one of the supported LightGBM presets.",
            allowed_arguments={
                "preset": (
                    "balanced_lgbm",
                    "wide_lgbm",
                    "deep_lgbm",
                )
            },
        ),
        "set_seed_ensemble": ToolDefinition(
            name="set_seed_ensemble",
            description="Set the number of seeds used in prediction averaging.",
            allowed_arguments={"size": (1, 3)},
        ),
    }


def execute_tool_plan(
    iteration: int,
    tool_plan: ToolPlan,
    tool_catalog: dict[str, ToolDefinition],
) -> ExperimentSpec:
    feature_pack = "encoded_default"
    model_preset = "balanced_lgbm"
    seed_ensemble_size = 1
    for step in tool_plan.steps:
        if step.name not in tool_catalog:
            raise ValueError(f"Unknown tool: {step.name}")
        if step.name == "select_feature_pack":
            feature_pack = str(step.arguments["feature_pack"])
        elif step.name == "select_model_preset":
            model_preset = str(step.arguments["preset"])
        elif step.name == "set_seed_ensemble":
            seed_ensemble_size = int(step.arguments["size"])
    return ExperimentSpec(
        name=f"iteration_{iteration:02d}",
        feature_pack=feature_pack,
        model=ModelSpec(
            family="lightgbm",
            objective="regression",
            params=_model_preset_to_params(model_preset),
        ),
        seed_ensemble_size=seed_ensemble_size,
    )


def _model_preset_to_params(preset: str) -> dict[str, int | float | str | bool]:
    presets = {
        "balanced_lgbm": {
            "n_estimators": 500,
            "learning_rate": 0.03,
            "num_leaves": 63,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
        "wide_lgbm": {
            "n_estimators": 700,
            "learning_rate": 0.025,
            "num_leaves": 95,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 0.5,
        },
        "deep_lgbm": {
            "n_estimators": 900,
            "learning_rate": 0.02,
            "num_leaves": 127,
            "subsample": 0.8,
            "colsample_bytree": 0.75,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
        },
    }
    return presets[preset]
