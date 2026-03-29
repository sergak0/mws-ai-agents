from kaggle_multi_agent.contracts import ToolCall, ToolPlan
from kaggle_multi_agent.tools import build_tool_catalog, execute_tool_plan


def test_execute_tool_plan_builds_experiment_spec() -> None:
    tool_plan = ToolPlan(
        steps=[
            ToolCall(name="select_feature_pack", arguments={"feature_pack": "encoded_geo"}),
            ToolCall(name="select_model_preset", arguments={"preset": "wide_lgbm"}),
            ToolCall(name="set_seed_ensemble", arguments={"size": 3}),
        ]
    )
    spec = execute_tool_plan(
        iteration=2,
        tool_plan=tool_plan,
        tool_catalog=build_tool_catalog(),
    )
    assert spec.feature_pack == "encoded_geo"
    assert spec.seed_ensemble_size == 3
    assert spec.model.params["num_leaves"] == 95


def test_execute_tool_plan_rejects_unknown_tool() -> None:
    tool_plan = ToolPlan(steps=[ToolCall(name="not_a_tool", arguments={})])
    try:
        execute_tool_plan(
            iteration=0,
            tool_plan=tool_plan,
            tool_catalog=build_tool_catalog(),
        )
    except ValueError as exc:
        assert "Unknown tool" in str(exc)
    else:
        raise AssertionError("Expected unknown tool to raise ValueError")
