# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from sweedit.core.api_pool import get_llm_editor_config
from sweedit.core.llm import LLM
from sweedit.tools.base import Params, Tool


class StrReplaceEditorV2Parameters(Params):
    command: Literal["view", "create", "str_replace", "insert"] = Field(
        description=("The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`.")
    )
    path: str = Field(description=("Absolute path to file or directory, e.g. `/workspace/file.py` or `/workspace`."))
    file_text: str | None = Field(
        default=None,
        description=("Required parameter of `create` command, with the content of the file to be created."),
    )
    old_str: str | None = Field(
        default=None,
        description=("Required parameter of `str_replace` command containing the string in `path` to replace."),
    )
    new_str: str | None = Field(
        default=None,
        description=(
            "Required parameter of `str_replace` command containing the new string. "
            "Required parameter of `insert` command containing the string to insert."
        ),
    )
    insert_line: int | None = Field(
        default=None,
        description=(
            "Required parameter of `insert` command. The `new_str` will be "
            "inserted AFTER the line `insert_line` of `path`."
        ),
    )
    query: str | None = Field(
        default=None,
        description=(
            "Required for `view` command when `path` points to a file. "
            "A natural language query describing what you're looking for in the file. "
            "An LLM will analyze the file and return only the line ranges relevant to your query. "
            "Examples: 'Where is the authentication logic?', 'Show me the class definition for User', "
            "'Find all functions that handle HTTP requests'."
        ),
    )


class StrReplaceEditorV2(Tool):
    name = "str_replace_editor_v2"
    parameters = StrReplaceEditorV2Parameters

    @staticmethod
    def _load_viewer_system_message() -> str:
        """Load system message prompt for view command from text file."""
        path = Path(__file__).parent / "descriptions" / "llm_file_viewer_prompt.txt"
        return path.read_text(encoding="utf-8").strip()

    async def execute(self, params: StrReplaceEditorV2Parameters) -> dict[str, Any] | str:
        """Execute str_replace_editor_v2 command."""
        command = params.command
        path = params.path

        if command == "view":
            return await self._view(path, params.query)
        elif command == "create":
            return await self._create(path, params.file_text)
        elif command == "str_replace":
            return await self._str_replace(path, params.old_str, params.new_str)
        elif command == "insert":
            return await self._insert(path, params.insert_line, params.new_str)
        else:
            return f"ERROR: Unknown command '{command}'"

    # ==================== VIEW COMMAND ====================

    async def _view(self, path: str, query: str | None = None) -> dict[str, Any] | str:
        """View file or directory contents."""
        path_obj = Path(path)

        if not path_obj.exists():
            return f"ERROR: Invalid `path` parameter: The path {path} does not exist. Please provide a valid path."

        if path_obj.is_dir():
            return await self._view_directory(path, path_obj)
        else:
            return await self._view_file(path, query)

    async def _view_directory(self, path: str, path_obj: Path) -> str:
        """View directory contents up to 2 levels deep."""
        try:
            items = []
            items.append(f"{path}/")

            for item in sorted(path_obj.iterdir()):
                if not item.name.startswith("."):
                    if item.is_dir():
                        items.append(f"{path}/{item.name}/")
                        try:
                            for subitem in sorted(item.iterdir()):
                                if not subitem.name.startswith("."):
                                    items.append(f"{path}/{item.name}/{subitem.name}")
                        except PermissionError:
                            pass
                    else:
                        items.append(f"{path}/{item.name}")

            hidden_count = sum(1 for item in path_obj.iterdir() if item.name.startswith("."))

            result = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n"
            result += "\n".join(items)

            if hidden_count > 0:
                result += f"\n\n{hidden_count} hidden files/directories in this directory are excluded. "
                result += f"You can use 'ls -la {path}' to see them."

            return result
        except PermissionError:
            return f"ERROR: Permission denied accessing '{path}'"

    async def _view_file(self, path: str, query: str | None = None) -> dict[str, Any] | str:
        """View file contents using LLM to find relevant sections based on query."""
        # Read file content
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            return f"ERROR: File '{path}' appears to be binary and cannot be viewed as text."
        except Exception as e:
            return f"ERROR: {str(e)}"

        total_lines = len(lines)

        # If no query provided, require one
        if query is None:
            return "ERROR: Parameter `query` is required for `view` command when viewing a file."

        # Format file content with line numbers for the LLM
        numbered_content = self._format_file_with_line_numbers(lines)

        # Prepare user message for LLM
        user_message = f"""File content:
{numbered_content}

Query: {query}

Please analyze the file and return the line ranges relevant to this query as a JSON array."""

        # Initialize LLM
        try:
            llm_editor_config = get_llm_editor_config()
            api_type = llm_editor_config["API_TYPE"]
            use_responses_api = api_type != "OPENAI"
            llm = LLM(api_type=api_type, api_config=llm_editor_config, use_responses_api=use_responses_api)

            agent_input = {
                "system": self._load_viewer_system_message(),
                "messages": [{"role": "user", "content": user_message}],
            }

            response = await llm.call(agent_input, max_tokens=4096)
            llm_output = self._extract_text_from_response(response)

        except Exception as e:
            return {"output": f"ERROR calling LLM: {str(e)}"}

        # Parse the LLM response to get line ranges
        try:
            line_ranges = self._parse_line_ranges(llm_output, total_lines)
        except Exception as e:
            return {
                "output": f"ERROR parsing LLM response: {str(e)}\nLLM output was:\n{llm_output}",
                "llm_viewer_system": self._load_viewer_system_message(),
                "llm_viewer_user": user_message,
                "llm_viewer_assistant": llm_output,
            }

        # If no relevant ranges found
        if not line_ranges:
            token_stats = self._extract_token_stats(response)
            return {
                "output": f"No content in {path} matches the query.",
                "llm_viewer_system": self._load_viewer_system_message(),
                "llm_viewer_user": user_message,
                "llm_viewer_assistant": llm_output,
                "llm_viewer_token_stats": token_stats,
            }

        # Format the output with the relevant line ranges
        formatted_output = self._format_view_output(path, lines, line_ranges)

        # Extract token statistics
        token_stats = self._extract_token_stats(response)

        return {
            "output": formatted_output,
            "llm_viewer_system": self._load_viewer_system_message(),
            "llm_viewer_user": user_message,
            "llm_viewer_assistant": llm_output,
            "llm_viewer_token_stats": token_stats,
        }

    def _extract_text_from_response(self, response) -> str:
        """Extract text from LLM response (handles OpenAI Responses, Chat Completions, and Anthropic formats)."""
        content = ""

        # OpenAI Responses API format
        if hasattr(response, "output"):
            content = response.output_text
        # OpenAI Chat Completions API format
        elif hasattr(response, "choices") and len(response.choices) > 0:
            content = response.choices[0].message.content
        # Anthropic format
        elif hasattr(response, "content") and isinstance(response.content, list):
            content = "".join([block.text for block in response.content if hasattr(block, "text")])
        else:
            content = str(response)

        # Handle reasoning models that generate <think></think> tags
        # Extract only content after the closing </think> tag
        if "</think>" in content:
            content = content.split("</think>", 1)[1].strip()

        return content

    def _extract_token_stats(self, response) -> dict[str, int]:
        """Extract token statistics from LLM response."""
        token_stats = {
            "input_tokens": 0,
            "output_tokens": 0,
        }

        if not hasattr(response, "usage") or response.usage is None:
            return token_stats

        usage = response.usage

        # Try Chat Completions API format first (prompt_tokens, completion_tokens)
        if hasattr(usage, "prompt_tokens"):
            token_stats["input_tokens"] = getattr(usage, "prompt_tokens", 0)
            token_stats["output_tokens"] = getattr(usage, "completion_tokens", 0)
        else:
            # Fall back to Responses API/Anthropic format (input_tokens, output_tokens)
            token_stats["input_tokens"] = getattr(usage, "input_tokens", 0)
            token_stats["output_tokens"] = getattr(usage, "output_tokens", 0)

        return token_stats

    def _format_file_with_line_numbers(self, lines: list[str]) -> str:
        """Format file content with line numbers for LLM input."""
        numbered_lines = [f"{i + 1:6d}\t{line.rstrip()}" for i, line in enumerate(lines)]
        return "\n".join(numbered_lines)

    def _parse_line_ranges(self, llm_output: str, total_lines: int) -> list[tuple[int, int]]:
        """Parse LLM output to extract line ranges."""
        import json

        # Try to extract JSON array from the response
        # The LLM might include some extra text, so we try to find the JSON array
        llm_output = llm_output.strip()

        # Try direct JSON parsing first
        try:
            ranges = json.loads(llm_output)
            if isinstance(ranges, list):
                return self._validate_and_merge_ranges(ranges, total_lines)
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in the output using regex
        json_pattern = r"\[\s*(?:\[\s*\d+\s*,\s*\d+\s*\]\s*,?\s*)*\]"
        match = re.search(json_pattern, llm_output)
        if match:
            try:
                ranges = json.loads(match.group())
                return self._validate_and_merge_ranges(ranges, total_lines)
            except json.JSONDecodeError:
                pass

        # If we can't parse, raise an error
        raise ValueError("Could not parse line ranges from LLM output")

    def _validate_and_merge_ranges(self, ranges: list, total_lines: int) -> list[tuple[int, int]]:
        """Validate and merge overlapping line ranges."""
        if not ranges:
            return []

        # Validate and convert ranges
        validated = []
        for r in ranges:
            if not isinstance(r, list) or len(r) != 2:
                continue
            start, end = int(r[0]), int(r[1])
            # Clamp to valid range (1-indexed)
            start = max(1, min(start, total_lines))
            end = max(1, min(end, total_lines))
            if start <= end:
                validated.append((start, end))

        if not validated:
            return []

        # Sort by start line
        validated.sort(key=lambda x: x[0])

        # Merge overlapping or adjacent ranges
        merged = [validated[0]]
        for start, end in validated[1:]:
            last_start, last_end = merged[-1]
            # Merge if overlapping or adjacent (within 3 lines)
            if start <= last_end + 3:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        return merged

    def _format_view_output(self, path: str, lines: list[str], ranges: list[tuple[int, int]]) -> str:
        """Format the view output with line numbers and separators between ranges."""
        output_parts = []
        output_parts.append(f"Relevant sections in {path}:\n")

        for i, (start, end) in enumerate(ranges):
            # Convert to 0-indexed for list access
            start_idx = start - 1
            end_idx = end

            # Format lines with line numbers
            numbered_lines = [f"{j + 1:6d}\t{lines[j].rstrip()}" for j in range(start_idx, min(end_idx, len(lines)))]
            output_parts.append("\n".join(numbered_lines))

            # Add separator between ranges (not after the last one)
            if i < len(ranges) - 1:
                next_start = ranges[i + 1][0]
                omitted_lines = next_start - end - 1
                if omitted_lines > 0:
                    output_parts.append(f"\n... ({omitted_lines} lines omitted) ...\n")
                else:
                    output_parts.append("\n")

        return "\n".join(output_parts)

    # ==================== CREATE COMMAND ====================

    async def _create(self, path: str, file_text: str | None) -> str:
        """Create a new file."""
        if file_text is None:
            return "ERROR: Parameter `file_text` is required for command: create."

        path_obj = Path(path)

        if path_obj.exists():
            return f"ERROR: File already exists at: {path}. Cannot overwrite files using command `create`."

        try:
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                f.write(file_text)

            return f"File created successfully at: {path}"
        except Exception as e:
            return f"ERROR: {str(e)}"

    async def _str_replace(self, path: str, old_str: str | None, new_str: str | None) -> str:
        """Replace a string in a file."""
        if old_str is None:
            return "ERROR: Parameter `old_str` is required for command: str_replace."

        path_obj = Path(path)

        if not path_obj.exists():
            return f"ERROR: Invalid `path` parameter: The path {path} does not exist. Please provide a valid path."

        if path_obj.is_dir():
            return f"ERROR: '{path}' is a directory, not a file."

        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()

            if new_str is not None and old_str == new_str:
                return "ERROR: No replacement was performed. `new_str` and `old_str` must be different."

            # Check if old_str exists and find all occurrences
            occurrences = []

            current_pos = 0
            while True:
                pos = content.find(old_str, current_pos)
                if pos == -1:
                    break
                line_num = content[:pos].count("\n") + 1
                occurrences.append(line_num)
                current_pos = pos + 1

            if len(occurrences) == 0:
                return "Error: No match found"

            if len(occurrences) > 1:
                return f"Error: Found {len(occurrences)} matches"

            new_content = content.replace(old_str, new_str or "", 1)

            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return "Successfully replaced text at exactly one location."
        except UnicodeDecodeError:
            return f"ERROR: File '{path}' appears to be binary and cannot be edited."
        except Exception as e:
            return f"ERROR: {str(e)}"

    async def _insert(self, path: str, insert_line: int | None, new_str: str | None) -> str:
        """Insert text after a specific line."""
        if insert_line is None:
            return "ERROR: Parameter `insert_line` is required for command: insert."
        if new_str is None:
            return "ERROR: Parameter `new_str` is required for command: insert."

        path_obj = Path(path)

        if not path_obj.exists():
            return f"ERROR: Invalid `path` parameter: The path {path} does not exist. Please provide a valid path."

        if path_obj.is_dir():
            return f"ERROR: '{path}' is a directory, not a file."

        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()

            if insert_line < 0 or insert_line > len(lines):
                return (
                    f"ERROR: Invalid `insert_line` parameter: {insert_line}. "
                    f"It should be within the range of allowed values: [0, {len(lines)}]."
                )

            if not new_str.endswith("\n"):
                new_str += "\n"

            lines.insert(insert_line, new_str)

            new_content = "".join(lines)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

            return f"Successfully inserted text after line {insert_line}."
        except UnicodeDecodeError:
            return f"ERROR: File '{path}' appears to be binary and cannot be edited."
        except Exception as e:
            return f"ERROR: {str(e)}"
