# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from pathlib import Path
from typing import Any, Literal

import aiofiles

from sweedit.core.types import AgentInput, AgentResponse
from sweedit.tools import Tool

# Tools that have sub-commands (command parameter)
TOOLS_WITH_COMMANDS = {"str_replace_editor", "str_replace_editor_v2", "llm_file_editor", "llm_editor"}


class ContextManager:
    """Manages conversation context, token usage, and tool call statistics."""

    def __init__(
        self,
        system: str | None = None,
        tools: list[Tool] | None = None,
        api_type: Literal["OPENAI", "AZURE_OPENAI", "ANTHROPIC"] = "OPENAI",
        enable_live_save: bool = True,
        save_dir: Path | None = None,
    ):
        if enable_live_save and save_dir is None:
            raise ValueError("save_dir must be provided if enable_live_save is True")

        self._system = system
        self._tools = tools or []
        self._api_type = api_type
        self._messages: list[dict[str, Any]] = []

        # Initialize tool call counts - nested dict for tools with commands, int for others
        self._tool_call_counts: dict[str, int | dict[str, int]] = {}
        for tool in self._tools:
            if tool.name in TOOLS_WITH_COMMANDS:
                self._tool_call_counts[tool.name] = {}
            else:
                self._tool_call_counts[tool.name] = 0

        # Token tracking
        self._token_stats = {
            "total_noncached_input_tokens": 0,
            "total_cached_input_tokens": 0,
            "total_output_tokens": 0,  # Total decode work (sum all rounds)
            "total_reasoning_tokens": 0,  # Sum of all reasoning tokens (subset of output)
            "peak_context_size": 0,  # Maximum context size seen (capacity limit / memory bound)
            "num_rounds": 0,  # Number of LLM API calls (iteration complexity)
        }

        # LLM editor token tracking (separate from main agent)
        self._llm_editor_token_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "num_calls": 0,
        }

        # LLM viewer token tracking (separate from main agent)
        self._llm_viewer_token_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "num_calls": 0,
        }

        # Live save configuration
        self._enable_live_save = enable_live_save
        self.set_save_dir(save_dir)

    async def add_user_message(self, user_message: str) -> None:
        if self._api_type == "ANTHROPIC":
            message = {"role": "user", "content": user_message}
        else:
            message = {"role": "user", "content": user_message, "type": "message"}

        self._messages.append(message)
        await self._update_trajectory_async(message)

    async def add_llm_response(self, response: AgentResponse) -> None:
        if self._api_type == "ANTHROPIC":
            # Always save full response to trajectory
            full_content = [content.model_dump() for content in response.content]
            self._track_tool_calls(response.content)
            await self._update_trajectory_async({"role": "assistant", "content": full_content})

            self._messages.append({"role": "assistant", "content": full_content})
        else:
            for output in response.output:
                message = output.model_dump(exclude={"status"})
                await self._update_trajectory_async(message)
                self._messages.append(message)

            self._track_tool_calls(response.output)

        # Track token usage from response
        self._track_token_usage(response)

        await self._update_tool_stats_async()
        await self._update_token_stats_async()

    async def add_tool_result(self, tool_call_id: str, tool_call_result: str) -> None:
        if self._api_type == "ANTHROPIC":
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": tool_call_result,
                    }
                ],
            }
        else:
            message = {
                "call_id": tool_call_id,
                "output": tool_call_result,
                "type": "function_call_output",
            }

        self._messages.append(message)

        await self._update_trajectory_async(message)

    async def add_llm_editor_call(
        self, system: str, user: str, assistant: str, original_content: str, token_stats: dict[str, int]
    ) -> None:
        """Track LLM editor call input, response, and token usage."""
        # Update LLM editor token stats
        self._llm_editor_token_stats["total_input_tokens"] += token_stats.get("input_tokens") or 0
        self._llm_editor_token_stats["total_output_tokens"] += token_stats.get("output_tokens") or 0
        self._llm_editor_token_stats["num_calls"] += 1

        # Save trajectory and token stats
        await self._update_llm_editor_trajectory_async(system, user, assistant, original_content=original_content)
        await self._update_llm_editor_token_stats_async()

    async def add_llm_viewer_call(self, system: str, user: str, assistant: str, token_stats: dict[str, int]) -> None:
        """Track LLM viewer call input, response, and token usage."""
        # Update LLM viewer token stats
        self._llm_viewer_token_stats["total_input_tokens"] += token_stats.get("input_tokens") or 0
        self._llm_viewer_token_stats["total_output_tokens"] += token_stats.get("output_tokens") or 0
        self._llm_viewer_token_stats["num_calls"] += 1

        # Save trajectory and token stats
        await self._update_llm_viewer_trajectory_async(system, user, assistant)
        await self._update_llm_viewer_token_stats_async()

    def get_messages(self) -> AgentInput:
        tool_schemas = None
        if self._tools and self._api_type:
            tool_schemas = [tool.schema(api_type=self._api_type) for tool in self._tools]

        return AgentInput(messages=self._messages, system=self._system, tools=tool_schemas)

    def get_tool_call_stats(self) -> dict[str, int | dict[str, int]]:
        return self._tool_call_counts

    def get_token_stats(self) -> dict[str, Any]:
        """Get token consumption statistics."""
        return {
            "agent": self._token_stats,
            "llm_editor": self._llm_editor_token_stats,
            "llm_viewer": self._llm_viewer_token_stats,
        }

    def _track_tool_calls(self, output: list[AgentResponse]) -> None:
        for item in output:
            if item.type in ("function_call", "tool_use"):
                tool_name = item.name

                # Check if this tool has sub-commands
                if tool_name in TOOLS_WITH_COMMANDS:
                    # Extract command from arguments
                    command = self._extract_command(item)
                    if command:
                        # Initialize command count if not present
                        if tool_name not in self._tool_call_counts:
                            self._tool_call_counts[tool_name] = {}
                        if command not in self._tool_call_counts[tool_name]:
                            self._tool_call_counts[tool_name][command] = 0
                        self._tool_call_counts[tool_name][command] += 1
                else:
                    # Simple count for tools without commands
                    if tool_name not in self._tool_call_counts:
                        self._tool_call_counts[tool_name] = 0
                    self._tool_call_counts[tool_name] += 1

    def _extract_command(self, item) -> str | None:
        """Extract the command parameter from tool arguments."""
        try:
            # Handle different API formats
            if hasattr(item, "arguments"):
                # OpenAI format - arguments is a JSON string
                args = json.loads(item.arguments) if isinstance(item.arguments, str) else item.arguments
            elif hasattr(item, "input"):
                # Anthropic format - input is already a dict
                args = item.input
            else:
                return None

            return args.get("command")
        except (json.JSONDecodeError, AttributeError, TypeError):
            return None

    def _track_token_usage(self, response: AgentResponse) -> None:
        """Track token usage from LLM response."""
        if not hasattr(response, "usage") or response.usage is None:
            return

        usage = response.usage

        # Extract token counts based on API type
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)

        # Reasoning tokens are part of output tokens
        reasoning_tokens = 0
        if hasattr(usage, "output_tokens_details") and usage.output_tokens_details:
            reasoning_tokens = getattr(usage.output_tokens_details, "reasoning_tokens", 0)

        # Cached input tokens are part of input tokens
        cached_input_tokens = 0
        if hasattr(usage, "input_tokens_details") and usage.input_tokens_details:
            cached_input_tokens = getattr(usage.input_tokens_details, "cached_tokens", 0)

        self._token_stats["total_noncached_input_tokens"] += input_tokens - cached_input_tokens
        self._token_stats["total_cached_input_tokens"] += cached_input_tokens
        self._token_stats["total_output_tokens"] += output_tokens
        self._token_stats["total_reasoning_tokens"] += reasoning_tokens
        self._token_stats["peak_context_size"] = max(self._token_stats["peak_context_size"], input_tokens)
        self._token_stats["num_rounds"] += 1

    def clear(self) -> None:
        self._messages = []
        # Reset tool call counts with proper structure
        self._tool_call_counts = {}
        for tool in self._tools:
            if tool.name in TOOLS_WITH_COMMANDS:
                self._tool_call_counts[tool.name] = {}
            else:
                self._tool_call_counts[tool.name] = 0

        self._token_stats = {
            "total_noncached_input_tokens": 0,
            "total_cached_input_tokens": 0,
            "total_output_tokens": 0,
            "total_reasoning_tokens": 0,
            "num_rounds": 0,
            "peak_context_size": 0,
        }
        self._llm_editor_token_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "num_calls": 0,
        }
        self._llm_viewer_token_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "num_calls": 0,
        }

    def set_save_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._save_dir = path
        self._traj_path = self._save_dir / "traj.jsonl"
        self._tools_path = self._save_dir / "tools.jsonl"
        self._tool_call_stats_path = self._save_dir / "tool_call_stats.json"
        self._token_stats_path = self._save_dir / "token_stats.json"
        self._llm_editor_traj_path = self._save_dir / "llm_editor_traj.jsonl"
        self._llm_editor_token_stats_path = self._save_dir / "llm_editor_token_stats.json"
        self._llm_viewer_traj_path = self._save_dir / "llm_viewer_traj.jsonl"
        self._llm_viewer_token_stats_path = self._save_dir / "llm_viewer_token_stats.json"

        if self._enable_live_save:
            self._initialize_live_save()

    def save(self) -> None:
        with open(self._traj_path, "w", encoding="utf-8") as f:
            if self._system:
                system_msg = {"role": "system", "content": self._system}
                f.write(json.dumps(system_msg, ensure_ascii=False) + "\n")

            for message in self._messages:
                msg = self._serialize_message(message)

                for m in msg:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")

        with open(self._tools_path, "w", encoding="utf-8") as f:
            for tool in self._tools:
                tool_schema = tool.schema(api_type=self._api_type)
                f.write(json.dumps(tool_schema, ensure_ascii=False, indent=2) + "\n")

        with open(self._tool_call_stats_path, "w", encoding="utf-8") as f:
            json.dump(self._tool_call_counts, f, ensure_ascii=False, indent=2)

        with open(self._token_stats_path, "w", encoding="utf-8") as f:
            json.dump(self._token_stats, f, ensure_ascii=False, indent=2)

        with open(self._llm_editor_token_stats_path, "w", encoding="utf-8") as f:
            json.dump(self._llm_editor_token_stats, f, ensure_ascii=False, indent=2)

        with open(self._llm_viewer_token_stats_path, "w", encoding="utf-8") as f:
            json.dump(self._llm_viewer_token_stats, f, ensure_ascii=False, indent=2)

    def _initialize_live_save(self) -> None:
        """Initialize live save by creating directory and writing system/tools."""
        # Write system message if present
        if self._system:
            system_msg = {"role": "system", "content": self._system}
            with open(self._traj_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(system_msg, ensure_ascii=False) + "\n")

        # Write tool schemas
        with open(self._tools_path, "w", encoding="utf-8") as f:
            for tool in self._tools:
                tool_schema = tool.schema(api_type=self._api_type)
                f.write(json.dumps(tool_schema, ensure_ascii=False, indent=2) + "\n")

    async def _update_trajectory_async(self, message: dict[str, Any]) -> None:
        """Update the trajectory file with a new message."""
        if not self._enable_live_save or not self._save_dir:
            return

        serialized_messages = self._serialize_message(message)

        async with aiofiles.open(self._traj_path, "a", encoding="utf-8") as f:
            for msg in serialized_messages:
                await f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    async def _update_tool_stats_async(self) -> None:
        """Update the tool call statistics file."""
        if not self._enable_live_save or not self._save_dir:
            return

        async with aiofiles.open(self._tool_call_stats_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(self._tool_call_counts, ensure_ascii=False, indent=2))

    async def _update_token_stats_async(self) -> None:
        """Update the token statistics file."""
        if not self._enable_live_save or not self._save_dir:
            return

        async with aiofiles.open(self._token_stats_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(self._token_stats, ensure_ascii=False, indent=2))

    async def _update_llm_editor_trajectory_async(
        self, system: str, user: str, assistant: str, original_content: str
    ) -> None:
        """Update the LLM editor trajectory file with input and response."""
        if not self._enable_live_save or not self._save_dir:
            return

        # Serialize the input and response in simplified format
        trajectory_entry = {
            "system": system,
            "user": user,
            "assistant": assistant,
            "original_content": original_content,
        }

        async with aiofiles.open(self._llm_editor_traj_path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(trajectory_entry, ensure_ascii=False) + "\n")

    async def _update_llm_editor_token_stats_async(self) -> None:
        """Update the LLM editor token statistics file."""
        if not self._enable_live_save or not self._save_dir:
            return

        async with aiofiles.open(self._llm_editor_token_stats_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(self._llm_editor_token_stats, ensure_ascii=False, indent=2))

    async def _update_llm_viewer_trajectory_async(self, system: str, user: str, assistant: str) -> None:
        """Update the LLM viewer trajectory file with input and response."""
        if not self._enable_live_save or not self._save_dir:
            return

        # Serialize the input and response in simplified format
        trajectory_entry = {"system": system, "user": user, "assistant": assistant}

        async with aiofiles.open(self._llm_viewer_traj_path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(trajectory_entry, ensure_ascii=False) + "\n")

    async def _update_llm_viewer_token_stats_async(self) -> None:
        """Update the LLM viewer token statistics file."""
        if not self._enable_live_save or not self._save_dir:
            return

        async with aiofiles.open(self._llm_viewer_token_stats_path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(self._llm_viewer_token_stats, ensure_ascii=False, indent=2))

    def _serialize_message(self, message: dict[str, Any]) -> list[dict]:
        if self._api_type == "ANTHROPIC":
            # Anthropic Messages API
            if message["role"] == "user":
                if isinstance(message["content"], list):
                    return [
                        {
                            "role": "function_call_output",
                            "output": message["content"][0]["content"],
                            "tool_use_id": message["content"][0]["tool_use_id"],
                        }
                    ]
                else:
                    return [{"role": "user", "content": message["content"]}]
            else:
                serialized_messages = []
                content_list = message["content"]

                for block in content_list:
                    block_type = block.get("type")

                    if block_type == "text":
                        serialized_messages.append({"role": "assistant", "content": block["text"]})
                    elif block_type == "thinking":
                        serialized_messages.append({"role": "reasoning", "content": block["thinking"]})
                    elif block_type == "tool_use":
                        serialized_messages.append(
                            {
                                "role": "function_call",
                                "name": block["name"],
                                "arguments": block["input"],
                            }
                        )

                return serialized_messages
        else:
            # OpenAI/Azure OpenAI Responses API
            if message["type"] == "message":
                content = message["content"][0]["text"] if isinstance(message["content"], list) else message["content"]
                return [{"role": message["role"], "content": content}]
            elif message["type"] == "reasoning":
                return [{"role": "reasoning", "content": message["content"]}]
            elif message["type"] == "function_call":
                return [
                    {
                        "role": "function_call",
                        "name": message["name"],
                        "arguments": message["arguments"],
                    }
                ]
            elif message["type"] == "function_call_output":
                return [
                    {
                        "role": "function_call_output",
                        "output": message["output"],
                    }
                ]
            else:
                raise ValueError(f"Unknown message type: {message['type']}")
