# Course Notes

## Core patterns

- Use explicit agent roles with narrow responsibilities.
- Prefer supervisor-worker or planner-critic-executor loops for controllable automation.
- Use reflection not only for code repair but also for metric-driven improvement.
- Validate every structured output with schemas.

## RAG guidance

- RAG should retrieve compact, relevant context instead of dumping full documents.
- Knowledge sources may include course notes, baseline repositories, past experiment reports, and model registries.
- Retrieval quality should be benchmarked, not assumed.

## Safety guidance

- Use input validation, guardrails, and monitoring.
- Keep execution inside bounded workspaces with controlled file access.
- Secrets must live in environment variables or local excluded files.

## Benchmarking guidance

- Compare baselines and ablations.
- Save experiment artifacts and logs per run.
- Track not only model score but also stability and reproducibility.

