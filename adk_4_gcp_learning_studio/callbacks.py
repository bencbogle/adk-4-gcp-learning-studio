"""Lifecycle callbacks for observing ADK execution."""

from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext


def log_tool_result(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: dict[str, Any],
) -> None:
    """Log a compact summary after a tool finishes.

    Returning ``None`` tells ADK to pass the original tool response through
    unchanged.
    """
    del args, tool_context  # This learning callback only observes the result.

    summary = f"status={tool_response.get('status', 'returned')}"
    sources = tool_response.get("sources")
    if isinstance(sources, list):
        summary += f" sources={len(sources)}"

    print(f"CALLBACK after_tool tool={tool.name} {summary}")


def log_model_response(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> None:
    """Log whether a model response contains text or a tool request."""
    del callback_context  # This learning callback only observes the response.

    parts = llm_response.content.parts if llm_response.content else []
    text_present = any(part.text for part in parts or [])
    tool_calls = [
        part.function_call.name
        for part in parts or []
        if part.function_call is not None and part.function_call.name
    ]
    print(
        "CALLBACK after_model "
        f"text={bool(text_present)} tool_calls={tool_calls or 'none'}"
    )
