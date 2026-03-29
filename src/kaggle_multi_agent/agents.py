from kaggle_multi_agent.contracts import (
    BenchmarkResult,
    CritiqueResult,
    DatasetProfile,
    ResearchBrief,
    ToolCall,
    ToolPlan,
)
from kaggle_multi_agent.guardrails import sanitize_untrusted_text
from kaggle_multi_agent.llm import OpenSourceLLM
from kaggle_multi_agent.rag import LocalKnowledgeBase


class ResearchAgent:
    def __init__(self, knowledge_base: LocalKnowledgeBase) -> None:
        self.knowledge_base = knowledge_base

    def run(
        self,
        iteration: int,
        dataset_profile: DatasetProfile,
        history: list[BenchmarkResult],
    ) -> ResearchBrief:
        query = (
            f"bounded regression kaggle feature engineering offline metrics iteration {iteration} "
            f"target range {dataset_profile.target_min} {dataset_profile.target_max}"
        )
        if history:
            query = f"{query} previous rmse {history[-1].holdout_rmse}"
        results = self.knowledge_base.search(query, k=3)
        sanitized_chunks = []
        guardrail_events = []
        for result in results:
            sanitized = sanitize_untrusted_text(result.text)
            if sanitized.text:
                sanitized_chunks.append(sanitized.text)
            guardrail_events.extend(sanitized.events)
        return ResearchBrief(
            retrieved_chunks=sanitized_chunks,
            feature_ideas=[
                "date parts",
                "text length signals",
                "review rate normalization",
                "consistent categorical encoding",
                "geo bucketing",
                "interaction features",
            ],
            model_ideas=[
                "numeric categorical encoding",
                "higher-capacity lightgbm",
                "seed ensembling for stable leaderboard gains",
            ],
            guardrail_events=guardrail_events,
        )


class ModelerAgent:
    def __init__(self, llm: OpenSourceLLM | None = None) -> None:
        self.llm = llm

    def run(
        self,
        iteration: int,
        research_brief: ResearchBrief,
    ) -> ToolPlan:
        if self.llm is not None and self.llm.is_available():
            try:
                return self.llm.generate_structured(
                    schema=ToolPlan,
                    system_prompt=(
                        "Return only JSON for a tool plan for bounded regression. "
                        "Use only these tools in order when needed: "
                        "select_feature_pack(feature_pack), "
                        "select_model_preset(preset), "
                        "set_seed_ensemble(size). "
                        "Allowed feature packs: encoded_default, encoded_geo, "
                        "encoded_geo_interactions. "
                        "Allowed presets: balanced_lgbm, wide_lgbm, deep_lgbm. "
                        "Allowed seed ensemble sizes: 1 or 3."
                    ),
                    user_prompt=(
                        f"Iteration: {iteration}\n"
                        f"Retrieved context: {research_brief.retrieved_chunks}\n"
                        f"Feature ideas: {research_brief.feature_ideas}\n"
                        f"Model ideas: {research_brief.model_ideas}\n"
                    ),
                )
            except Exception:
                pass
        return _fallback_tool_plan(iteration)


class CriticAgent:
    def __init__(self, llm: OpenSourceLLM | None = None) -> None:
        self.llm = llm

    def run(
        self,
        current: BenchmarkResult,
        best: BenchmarkResult | None,
        iteration: int,
        max_iterations: int,
    ) -> CritiqueResult:
        if self.llm is not None and self.llm.is_available():
            try:
                return self.llm.generate_structured(
                    schema=CritiqueResult,
                    system_prompt=(
                        "Return only JSON for a critique decision. "
                        "Valid decisions are accept, retry, or stop."
                    ),
                    user_prompt=(
                        f"Iteration: {iteration}\n"
                        f"Max iterations: {max_iterations}\n"
                        f"Current result: {current.model_dump_json()}\n"
                        f"Best result: {best.model_dump_json() if best else 'null'}\n"
                    ),
                )
            except Exception:
                pass
        if iteration + 1 >= max_iterations:
            return CritiqueResult(
                decision="stop",
                summary="Iteration budget exhausted",
                actions=[],
            )
        if best is None:
            return CritiqueResult(
                decision="retry",
                summary="Establish baseline and continue search",
                actions=["increase model capacity", "compare offline private score"],
            )
        if _score(current) < _score(best):
            return CritiqueResult(
                decision="retry",
                summary="New best candidate found, continue exploration",
                actions=["refine learning rate", "preserve useful features"],
            )
        return CritiqueResult(
            decision="retry",
            summary="Candidate did not improve best score",
            actions=["adjust num_leaves", "preserve robust feature pack"],
        )


def _score(result: BenchmarkResult) -> float:
    if result.offline_private_rmse is not None:
        return result.offline_private_rmse
    return result.holdout_rmse


def _fallback_tool_plan(iteration: int) -> ToolPlan:
    presets = [
        ("encoded_default", "balanced_lgbm", 1),
        ("encoded_geo", "wide_lgbm", 1),
        ("encoded_geo_interactions", "deep_lgbm", 3),
    ]
    feature_pack, preset, seed_size = presets[min(iteration, len(presets) - 1)]
    return ToolPlan(
        steps=[
            ToolCall(name="select_feature_pack", arguments={"feature_pack": feature_pack}),
            ToolCall(name="select_model_preset", arguments={"preset": preset}),
            ToolCall(name="set_seed_ensemble", arguments={"size": seed_size}),
        ]
    )
