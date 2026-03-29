# Security

## Threat Model

The project has two different trust zones:

- trusted code and configs inside the repository
- untrusted text and artifacts coming from Kaggle descriptions, local notes, retrieved chunks, or any external prompt source

The main security goal is to prevent untrusted text from steering the agents outside the intended workflow.

## Secret Handling

- Kaggle credentials are loaded from local excluded files such as `kaggle.json`
- API keys for OpenRouter or Ollama-compatible services are provided only through environment variables
- no secrets are committed into the repository
- the public client-delivery package excludes credentials, raw data, and local artifacts

Relevant files:

- [src/kaggle_multi_agent/settings.py](../src/kaggle_multi_agent/settings.py)
- [src/kaggle_multi_agent/data.py](../src/kaggle_multi_agent/data.py)

## Input Validation

Validation happens at system boundaries:

- Kaggle bundle validation checks that the expected files exist before execution
- typed contracts in [src/kaggle_multi_agent/contracts.py](../src/kaggle_multi_agent/contracts.py) validate dataset profiles, tool plans, experiment specs, and critique outputs
- prediction arrays are clipped to the configured target bounds before submission
- the agent loop is bounded by `max_iterations`

This keeps malformed inputs from silently entering the executor.

## Prompt-Injection Guardrails

The system treats retrieved text as untrusted.

Current defenses:

- retrieved markdown chunks are sanitized in [src/kaggle_multi_agent/guardrails.py](../src/kaggle_multi_agent/guardrails.py)
- suspicious lines are removed before they are passed into the agent context
- the current sanitizer blocks patterns such as:
  - `ignore previous instructions`
  - requests to reveal the system prompt
  - requests to leak secrets or credentials
  - instructions to execute shell, bash, python, or code

Flow:

1. `ResearchAgent` retrieves chunks from the local KB
2. each chunk is passed through `sanitize_untrusted_text`
3. sanitized content is kept
4. removed lines are recorded as guardrail events
5. only sanitized context is passed to `ModelerAgent`

This creates a basic but explicit trust boundary between retrieved content and agent reasoning.

## Tool Guardrails

Agents cannot call arbitrary runtime actions.

The tool layer enforces:

- fixed tool names from [src/kaggle_multi_agent/tools.py](../src/kaggle_multi_agent/tools.py)
- fixed argument names
- fixed allowed argument values
- rejection of unsupported tool plans before execution

Example:

- a modeler output that asks for an unknown tool is rejected
- a tool call with an unsupported feature pack is rejected
- a tool call with extra arguments is rejected

This is the main safety mechanism that keeps the system tool-driven instead of unconstrained code execution.

## Monitoring

The project includes lightweight monitoring rather than a full observability stack.

Per-iteration monitoring includes:

- tool trace
- metrics
- critique decision
- guardrail event count
- best-so-far flag

Outputs:

- local JSONL event log in `iteration_events.jsonl`
- run reports in markdown
- optional Weights & Biases tracking when enabled

Relevant files:

- [src/kaggle_multi_agent/tracking.py](../src/kaggle_multi_agent/tracking.py)
- [src/kaggle_multi_agent/reporting.py](../src/kaggle_multi_agent/reporting.py)

## Reliability Limits

What this version protects well:

- accidental malformed tool outputs
- simple prompt-injection patterns in retrieved text
- unsafe drift outside the approved tool catalog
- infinite loops from uncontrolled retries

What it does not claim:

- full sandboxed code generation
- formal red-teaming against advanced adversarial prompts
- network-level or host-level isolation

For a course project this is a defensible middle ground: the agents are autonomous within a narrow, auditable execution envelope.
