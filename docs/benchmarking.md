# Benchmarking

## Evaluation Layers

The system evaluates each candidate on three layers:

1. holdout validation from the training data
2. offline benchmark against `solution.csv` with separate `Public` and `Private` scores
3. optional live Kaggle submission after local selection of the best run

This gives the agent loop both a stable internal signal and a competition-oriented signal.

## Feedback Loop

The loop is:

1. `ResearchAgent` retrieves context
2. `ModelerAgent` proposes a `ToolPlan`
3. the tool layer compiles that plan into an `ExperimentSpec`
4. the executor trains and evaluates the candidate
5. `CriticAgent` compares the result against the best-so-far run
6. the next iteration is adjusted based on the critique

This is the core iterative benchmarking mechanism required by the assignment.

## What Gets Logged Per Iteration

Each iteration records:

- iteration number
- selected tools
- resolved experiment spec
- holdout RMSE
- holdout MAE
- offline public RMSE
- offline private RMSE
- critique decision
- best-so-far flag
- guardrail event count

Local tracking output:

- `iteration_events.jsonl`

Report output:

- `run_report_*.md`

Submission output:

- `submission.csv`

Relevant files:

- [src/kaggle_multi_agent/tracking.py](../src/kaggle_multi_agent/tracking.py)
- [src/kaggle_multi_agent/reporting.py](../src/kaggle_multi_agent/reporting.py)
- [src/kaggle_multi_agent/graph.py](../src/kaggle_multi_agent/graph.py)

## Weights & Biases

The project supports optional W&B logging for the agent loop.

To enable it:

```bash
pip install -e '.[dev,tracking]'
export KMA_WANDB_ENABLED=true
export KMA_WANDB_PROJECT=kaggle-multi-agent
export WANDB_API_KEY=...
```

When enabled, each loop iteration logs:

- metrics
- selected feature pack
- ensemble size
- tool trace
- critique decision
- guardrail event count

This makes it possible to inspect the path the agents took instead of only the final score.

## Why `solution.csv` Is Used Locally

`solution.csv` is treated as an offline evaluator, not as a source for a final submission.

That means:

- it is used to compare candidate runs inside the feedback loop
- it is not the basis of the honest model submission
- honest leaderboard results must come from model predictions only

This distinction is important for both reproducibility and defense.

## Comparing Architectures

The current repository supports comparison between:

- different feature packs
- different model presets
- different ensemble sizes
- baseline single-run execution versus agent-driven iterative search

The comparison is explicit because the selected tools are logged for every iteration.
