from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from kaggle_multi_agent.agents import CriticAgent, ModelerAgent, ResearchAgent
from kaggle_multi_agent.contracts import (
    BenchmarkResult,
    CritiqueResult,
    DatasetProfile,
    ExperimentSpec,
    ToolPlan,
)
from kaggle_multi_agent.data import CompetitionFrames
from kaggle_multi_agent.guardrails import validate_tool_plan
from kaggle_multi_agent.modeling import run_experiment
from kaggle_multi_agent.profiling import build_dataset_profile
from kaggle_multi_agent.rag import LocalKnowledgeBase
from kaggle_multi_agent.tools import build_tool_catalog, execute_tool_plan


class AgentState(TypedDict, total=False):
    frames: CompetitionFrames
    knowledge_base: LocalKnowledgeBase
    dataset_profile: DatasetProfile
    research_brief: Any
    current_tool_plan: ToolPlan
    current_tool_trace: list[str]
    current_spec: ExperimentSpec
    current_result: BenchmarkResult
    current_predictions: Any
    best_spec: ExperimentSpec
    best_result: BenchmarkResult
    best_predictions: Any
    critique: CritiqueResult
    history: list[BenchmarkResult]
    tool_trace_history: list[list[str]]
    guardrail_events: list[str]
    iteration: int
    max_iterations: int
    target_column: str
    random_seed: int
    planner_llm: Any
    critic_llm: Any
    trackers: list[Any]


def run_agent_loop(
    frames: CompetitionFrames,
    knowledge_base: LocalKnowledgeBase,
    target_column: str,
    max_iterations: int,
    random_seed: int,
    planner_llm: Any = None,
    critic_llm: Any = None,
    trackers: list[Any] | None = None,
) -> AgentState:
    graph = _build_graph()
    initial_state: AgentState = {
        "frames": frames,
        "knowledge_base": knowledge_base,
        "history": [],
        "tool_trace_history": [],
        "guardrail_events": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "target_column": target_column,
        "random_seed": random_seed,
        "planner_llm": planner_llm,
        "critic_llm": critic_llm,
        "trackers": trackers or [],
    }
    return graph.invoke(initial_state)


def _build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("research", _research_node)
    builder.add_node("model", _model_node)
    builder.add_node("tools", _tool_node)
    builder.add_node("benchmark", _benchmark_node)
    builder.add_node("critique", _critique_node)
    builder.add_edge(START, "research")
    builder.add_edge("research", "model")
    builder.add_edge("model", "tools")
    builder.add_edge("tools", "benchmark")
    builder.add_edge("benchmark", "critique")
    builder.add_conditional_edges("critique", _route_after_critique)
    return builder.compile()


def _research_node(state: AgentState) -> AgentState:
    dataset_profile = state.get("dataset_profile") or build_dataset_profile(
        state["frames"].train,
        state["frames"].test,
        target_column=state["target_column"],
    )
    agent = ResearchAgent(state["knowledge_base"])
    brief = agent.run(
        iteration=state["iteration"],
        dataset_profile=dataset_profile,
        history=state.get("history", []),
    )
    return {
        "dataset_profile": dataset_profile,
        "research_brief": brief,
        "guardrail_events": [*state.get("guardrail_events", []), *brief.guardrail_events],
    }


def _model_node(state: AgentState) -> AgentState:
    agent = ModelerAgent(llm=state.get("planner_llm"))
    tool_plan = agent.run(iteration=state["iteration"], research_brief=state["research_brief"])
    return {"current_tool_plan": tool_plan}


def _tool_node(state: AgentState) -> AgentState:
    tool_catalog = build_tool_catalog()
    validate_tool_plan(state["current_tool_plan"], tool_catalog)
    spec = execute_tool_plan(
        iteration=state["iteration"],
        tool_plan=state["current_tool_plan"],
        tool_catalog=tool_catalog,
    )
    spec = spec.model_copy(
        update={
            "retrieved_context": state["research_brief"].retrieved_chunks,
            "hypotheses": (
                state["research_brief"].feature_ideas + state["research_brief"].model_ideas
            ),
        }
    )
    tool_trace = [step.name for step in state["current_tool_plan"].steps]
    return {"current_spec": spec, "current_tool_trace": tool_trace}


def _benchmark_node(state: AgentState) -> AgentState:
    run = run_experiment(
        spec=state["current_spec"],
        frames=state["frames"],
        target_column=state["target_column"],
        random_seed=state["random_seed"],
    )
    return {"current_result": run.result, "current_predictions": run.predictions}


def _critique_node(state: AgentState) -> AgentState:
    critic = CriticAgent(llm=state.get("critic_llm"))
    critique = critic.run(
        current=state["current_result"],
        best=state.get("best_result"),
        iteration=state["iteration"],
        max_iterations=state["max_iterations"],
    )
    history = [*state.get("history", []), state["current_result"]]
    tool_trace_history = [*state.get("tool_trace_history", []), state["current_tool_trace"]]
    best_result = state.get("best_result")
    best_spec = state.get("best_spec")
    best_predictions = state.get("best_predictions")
    is_best = best_result is None or _score(state["current_result"]) < _score(best_result)
    if is_best:
        best_result = state["current_result"]
        best_spec = state["current_spec"]
        best_predictions = state["current_predictions"]
    for tracker in state.get("trackers", []):
        tracker.log_iteration(
            iteration=state["iteration"],
            best_so_far=is_best,
            spec=state["current_spec"],
            result=state["current_result"],
            critique=critique,
            tool_trace=state["current_tool_trace"],
            guardrail_events=state.get("guardrail_events", []),
        )
    return {
        "critique": critique,
        "history": history,
        "tool_trace_history": tool_trace_history,
        "best_result": best_result,
        "best_spec": best_spec,
        "best_predictions": best_predictions,
        "iteration": state["iteration"] + 1,
    }


def _route_after_critique(state: AgentState) -> Literal["research", END]:
    if state["critique"].decision == "stop":
        return END
    if state["iteration"] >= state["max_iterations"]:
        return END
    return "research"


def _score(result: BenchmarkResult) -> float:
    if result.offline_private_rmse is not None:
        return result.offline_private_rmse
    return result.holdout_rmse
