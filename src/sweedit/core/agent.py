# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from sweedit.core.api_pool import get_api_config
from sweedit.core.context_manager import ContextManager
from sweedit.core.llm import LLM
from sweedit.tools import Tool
from sweedit.utils import generate_session_id

load_dotenv()

# API-specific configuration
_API_CONFIG = {
    "ANTHROPIC": {
        "output_attr": "content",
        "tool_type": "tool_use",
        "tool_args_attr": "input",
        "tool_id_attr": "id",
    },
    "OPENAI": {
        "output_attr": "output",
        "tool_type": "function_call",
        "tool_args_attr": "arguments",
        "tool_id_attr": "call_id",
    },
}


class AgentBase:
    """Base agent class that coordinates LLM, Context, and tool execution."""

    def __init__(
        self,
        system: str | None = None,
        tools: list[Tool] | None = None,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        max_iterations: int = 50,
        session_id: str | None = None,
        save_dir: str = ".traj/",
        enable_live_save: bool = True,
        rate_limit_delay: float = 0.0,
    ):
        # Get random API config from pool
        api_config = get_api_config()
        self.api_type = api_config["API_TYPE"]

        # Store the appropriate API configuration
        api_key = "ANTHROPIC" if self.api_type == "ANTHROPIC" else "OPENAI"
        self._api_config = _API_CONFIG[api_key]

        tools = tools or []
        self._tool_handlers: dict[str, Tool] = {tool.name: tool for tool in tools}

        self.max_iterations = max_iterations
        self.rate_limit_delay = rate_limit_delay
        self.session_id = session_id or generate_session_id()
        self.save_dir = Path(save_dir) / self.session_id

        self.llm = LLM(api_type=self.api_type, api_config=api_config, reasoning_effort=reasoning_effort)
        self.context = ContextManager(
            system=system,
            tools=tools,
            api_type=self.api_type,
            enable_live_save=enable_live_save,
            save_dir=self.save_dir,
        )

    async def run(self, user_message: str) -> None:
        await self.context.add_user_message(user_message)

        for iteration in range(self.max_iterations):
            finish_called = False

            # Add optional delay between iterations to avoid rate limits
            if iteration > 0 and self.rate_limit_delay > 0:
                await asyncio.sleep(self.rate_limit_delay)

            llm_response = await self.llm.call(
                prepared_input=self.context.get_messages(),
            )
            await self.context.add_llm_response(llm_response)

            outputs = getattr(llm_response, self._api_config["output_attr"])
            tool_calls = [output for output in outputs if output.type in ("function_call", "tool_use")]

            if tool_calls:
                # Execute all tools in parallel
                results = [await self._execute_tool(tc) for tc in tool_calls]

                # Add results to context in original order
                for tool_call, result in zip(tool_calls, results, strict=False):
                    tool_id = getattr(tool_call, self._api_config["tool_id_attr"])

                    # Check if result is a dict with LLM editor metadata
                    if isinstance(result, dict):
                        if "llm_editor_user" in result:
                            await self.context.add_llm_editor_call(
                                system=result["llm_editor_system"],
                                user=result["llm_editor_user"],
                                assistant=result["llm_editor_assistant"],
                                original_content=result["original_content"],
                                token_stats=result["llm_editor_token_stats"],
                            )
                            tool_result = result["output"]
                        elif "llm_viewer_user" in result:
                            await self.context.add_llm_viewer_call(
                                system=result["llm_viewer_system"],
                                user=result["llm_viewer_user"],
                                assistant=result["llm_viewer_assistant"],
                                token_stats=result["llm_viewer_token_stats"],
                            )
                            tool_result = result["output"]
                        elif "output" in result:
                            # Handle error cases that return dict with only output
                            tool_result = result["output"]
                    else:
                        # Regular string result
                        tool_result = result

                    await self.context.add_tool_result(tool_id, tool_result)

                    # Check if finish tool was called
                    if tool_call.name == "finish":
                        finish_called = True
            else:
                # No tool calls made, inject fake user message to continue
                await self.context.add_user_message(
                    "If you have completed the task, please use the finish tool. Otherwise, continue with your work."
                )

            if finish_called:
                break

    async def _execute_tool(self, tool_call: str) -> str:
        """Execute tool and return result (for parallel execution)."""
        tool_name = tool_call.name
        tool_args = getattr(tool_call, self._api_config["tool_args_attr"])

        if tool_name not in self._tool_handlers:
            return f"Error: Tool '{tool_name}' not found"
        else:
            return await self._tool_handlers[tool_name](tool_args)

    def reset(self) -> None:
        """Reset the agent's context and statistics."""
        self.context.clear()

    def get_model(self) -> str:
        """Get the model used by the agent."""
        return self.llm.model

    def set_save_dir(self, save_dir: str) -> None:
        """Set the save directory for the agent."""
        self.save_dir = Path(save_dir)
        self.context.set_save_dir(self.save_dir)

    def set_workdir(self, workdir: str) -> None:
        """Set the work directory for the agent."""
        for tool in self._tool_handlers.values():
            if hasattr(tool, "set_workdir"):
                tool.set_workdir(workdir)

    def save_traj(self) -> None:
        """Save agent trajectory to disk."""
        self.context.save(self.save_dir / self.session_id)

    def get_tool_call_stats(self) -> dict[str, int]:
        """Get tool call statistics."""
        return self.context.get_tool_call_stats()

    def get_token_stats(self) -> dict:
        """Get token consumption statistics."""
        return self.context.get_token_stats()
