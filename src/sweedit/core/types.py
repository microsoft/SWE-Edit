# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, TypedDict

from anthropic.types import Message, MessageParam
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam
from openai.types.responses import Response, ResponseInputParam

type InputParam = ResponseInputParam | MessageParam | ChatCompletionMessageParam
type AgentResponse = Response | Message | ChatCompletion


class AgentInput(TypedDict):
    """Prepared input for LLM call including messages and tools."""

    messages: list[InputParam]
    system: str | None
    tools: list[dict[str, Any]] | None


__all__ = [
    "AgentResponse",
    "AgentInput",
]
