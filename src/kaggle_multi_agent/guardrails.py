import re
from dataclasses import dataclass

from kaggle_multi_agent.contracts import ToolPlan
from kaggle_multi_agent.tools import ToolDefinition

SUSPICIOUS_PATTERNS = (
    re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(the\s+)?system\s+prompt", re.IGNORECASE),
    re.compile(r"(leak|exfiltrate|print)\s+(secrets?|credentials?)", re.IGNORECASE),
    re.compile(r"execute\s+(shell|bash|python|code)", re.IGNORECASE),
)


@dataclass(frozen=True)
class SanitizedText:
    text: str
    events: list[str]
    was_modified: bool


def sanitize_untrusted_text(text: str) -> SanitizedText:
    sanitized_lines: list[str] = []
    events: list[str] = []
    for line in text.splitlines():
        if any(pattern.search(line) for pattern in SUSPICIOUS_PATTERNS):
            events.append(f"Removed suspicious untrusted line: {line.strip()[:80]}")
            continue
        sanitized_lines.append(line)
    sanitized_text = "\n".join(sanitized_lines).strip()
    return SanitizedText(
        text=sanitized_text,
        events=events,
        was_modified=sanitized_text != text.strip(),
    )


def validate_tool_plan(
    tool_plan: ToolPlan,
    tool_catalog: dict[str, ToolDefinition],
) -> None:
    for step in tool_plan.steps:
        if step.name not in tool_catalog:
            raise ValueError(f"Unknown tool: {step.name}")
        allowed_arguments = tool_catalog[step.name].allowed_arguments
        unexpected_arguments = set(step.arguments) - set(allowed_arguments)
        if unexpected_arguments:
            unexpected = ", ".join(sorted(unexpected_arguments))
            raise ValueError(f"Tool {step.name} received unexpected arguments: {unexpected}")
        for argument_name, argument_value in step.arguments.items():
            if argument_value not in allowed_arguments[argument_name]:
                raise ValueError(
                    f"Tool {step.name} received unsupported value for {argument_name}: "
                    f"{argument_value}"
                )
