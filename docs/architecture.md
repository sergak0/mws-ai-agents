# Architecture

## Overview

This repository implements a tool-driven multi-agent system for `mws-ai-agents-2026`.

The important distinction is this: the agents do not generate the entire ML solution code at runtime. The solution pipeline is already implemented in Python. The agents operate above it, choosing which typed tool sequence to try next, retrieving relevant context, reading metrics, and deciding whether to continue iterating.

The architecture is intentionally conservative:

- deterministic data loading, feature engineering, training, and reporting live in code
- agents handle planning, retrieval, tool selection, and critique
- the loop is fully automatic once started
- all intermediate state is stored locally for reproducibility

## Core Components

### Data Layer

- [src/kaggle_multi_agent/data.py](../src/kaggle_multi_agent/data.py) loads Kaggle files and validates their presence
- [src/kaggle_multi_agent/profiling.py](../src/kaggle_multi_agent/profiling.py) summarizes train/test shape and target range

Inputs:

- `train.csv`
- `test.csv`
- `sample_submition.csv`
- `solution.csv`

Outputs:

- `CompetitionFrames`
- `DatasetProfile`

### Knowledge Base

- Curated markdown sources live in [knowledge_base/curated/course_notes.md](../knowledge_base/curated/course_notes.md) and [knowledge_base/curated/competition_notes.md](../knowledge_base/curated/competition_notes.md)
- The retriever in [src/kaggle_multi_agent/rag.py](../src/kaggle_multi_agent/rag.py) builds a local TF-IDF index over markdown chunks
- The built index is stored locally as joblib artifacts in `knowledge_base/index/`

Why this choice:

- simple
- fast
- reproducible
- local-first
- easy to inspect during defense

### Agents

#### `ResearchAgent`

File: [src/kaggle_multi_agent/agents.py](../src/kaggle_multi_agent/agents.py)

Responsibilities:

- forms a retrieval query from the current dataset profile and history
- searches the local knowledge base
- returns a compact research brief

Input:

- `DatasetProfile`
- recent `BenchmarkResult` history
- local knowledge base

Output:

- `ResearchBrief`
- retrieved markdown chunks
- feature ideas
- model ideas
- guardrail events from sanitized untrusted context

#### `ModelerAgent`

File: [src/kaggle_multi_agent/agents.py](../src/kaggle_multi_agent/agents.py)

Responsibilities:

- turns the research brief into a typed tool plan
- chooses feature pack, model preset, and seed ensemble size through allowed tool calls
- optionally delegates the choice to an open-source LLM through an OpenAI-compatible client

Input:

- `ResearchBrief`
- iteration number

Output:

- `ToolPlan`

Current behavior:

- the agent does not generate arbitrary Python code
- it selects among predefined tools and allowed arguments
- the tool layer validates the plan before execution
- the executor applies the compiled experiment spec

#### `CriticAgent`

File: [src/kaggle_multi_agent/agents.py](../src/kaggle_multi_agent/agents.py)

Responsibilities:

- compares the current run against the best-so-far result
- checks iteration budget
- decides whether to retry, stop, or accept

Input:

- current `BenchmarkResult`
- best `BenchmarkResult`
- iteration count and budget

Output:

- `CritiqueResult`

### Tool Layer

Files:

- [src/kaggle_multi_agent/tools.py](../src/kaggle_multi_agent/tools.py)
- [src/kaggle_multi_agent/guardrails.py](../src/kaggle_multi_agent/guardrails.py)
- [src/kaggle_multi_agent/contracts.py](../src/kaggle_multi_agent/contracts.py)

Responsibilities:

- define the set of tools that agents may invoke
- validate tool names and arguments
- compile a `ToolPlan` into an executable `ExperimentSpec`
- block unsupported actions before they reach the executor

Current tools:

- `select_feature_pack(feature_pack)`
- `select_model_preset(preset)`
- `set_seed_ensemble(size)`

This is the key architectural compromise of the project. The system is autonomous at the level of planning and tool use, but execution is constrained to a safe and auditable tool catalog.

### Message Contracts

The graph passes typed objects rather than raw text:

- `ResearchBrief` contains `retrieved_chunks`, `feature_ideas`, `model_ideas`, `guardrail_events`
- `ToolPlan` contains a list of `ToolCall(name, arguments)`
- `ExperimentSpec` contains the resolved feature pack, model config, ensemble size, retrieved context, and hypotheses
- `BenchmarkResult` contains holdout and offline metrics
- `CritiqueResult` contains `decision`, `summary`, and follow-up `actions`

These contracts live in [src/kaggle_multi_agent/contracts.py](../src/kaggle_multi_agent/contracts.py) and are validated with `pydantic`.

### Executor

The executor is the deterministic part of the system:

- [src/kaggle_multi_agent/features.py](../src/kaggle_multi_agent/features.py) builds the selected feature pack
- [src/kaggle_multi_agent/modeling.py](../src/kaggle_multi_agent/modeling.py) trains and evaluates the model
- [src/kaggle_multi_agent/metrics.py](../src/kaggle_multi_agent/metrics.py) computes holdout and offline metrics
- [src/kaggle_multi_agent/reporting.py](../src/kaggle_multi_agent/reporting.py) writes reports and submissions
- [src/kaggle_multi_agent/registry.py](../src/kaggle_multi_agent/registry.py) persists experiment history

This is the part that makes the project reproducible. The agents choose the config; the executor does the actual work.

### Orchestration Graph

File: [src/kaggle_multi_agent/graph.py](../src/kaggle_multi_agent/graph.py)

The LangGraph state machine passes typed state through:

1. `research`
2. `model`
3. `tools`
4. `benchmark`
5. `critique`

The loop repeats until the critic decides to stop or the iteration budget is exhausted.

## Models

The project supports open-source models through OpenAI-compatible APIs:

- `mock` for deterministic offline runs
- `ollama` for local open-source models
- `openrouter` for hosted open-source models

Implementation:

- [src/kaggle_multi_agent/llm.py](../src/kaggle_multi_agent/llm.py)

The wrapper uses the OpenAI Python client only as a compatibility layer. The architecture is still open-source model friendly because the actual endpoint can be Ollama or OpenRouter.

## Prompt Locations

- planner prompt: [src/kaggle_multi_agent/agents.py](../src/kaggle_multi_agent/agents.py)
- critic prompt: [src/kaggle_multi_agent/agents.py](../src/kaggle_multi_agent/agents.py)
- JSON extraction and OpenAI-compatible routing: [src/kaggle_multi_agent/llm.py](../src/kaggle_multi_agent/llm.py)

The prompts are intentionally short and structured so the agents output typed JSON rather than free-form text. The planner prompt explicitly lists the only allowed tools and argument values.

## Data Flow

![Architecture Overview](assets/architecture-overview.png)

Mermaid source diagrams: [diagrams.md](diagrams.md)

## Why This Design

- The agents can be explained and tested independently.
- The executor can be benchmarked without LLM randomness.
- The pipeline is safer than unconstrained codegen because all execution paths are predefined.
- The system is still flexible because the agents can change the feature pack, model preset, and ensemble strategy across iterations.
- The tool layer makes the autonomy claim more defensible because agent choices are explicit and serializable.

## What This Is Not

This repository is not a fully autonomous code-writing agent that invents arbitrary Python pipelines from scratch at runtime.

Instead, it is:

- a multi-agent planner and critic system
- a retrieval-augmented tool selector
- a reproducible ML executor with typed experiment specs
- a feedback loop over a fixed set of tools and pipeline building blocks

That tradeoff was chosen to keep the system auditable, stable, and easy to defend in a course setting.
