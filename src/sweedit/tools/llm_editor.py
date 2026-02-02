# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import difflib
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from sweedit.core.api_pool import get_llm_editor_config
from sweedit.core.llm import LLM
from sweedit.tools.base import Params, Tool

MAX_LINES_TO_DISPLAY = 500


class LLMEditorParameters(Params):
    command: Literal["view", "create", "edit"] = Field(
        description="The command to run. Allowed options are: `view`, `create`, `edit`."
    )
    path: str = Field(description="Absolute path to file or directory, e.g. `/workspace/file.py` or `/workspace`.")
    instruction: str | None = Field(
        default=None,
        description=(
            "Required for `edit` command. Detailed instruction describing how to modify the file. "
            "Be specific about what changes to make and where (function/class/method name)."
        ),
    )
    file_text: str | None = Field(
        default=None,
        description="Required for `create` command. The content of the file to be created.",
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


class LLMEditor(Tool):
    name = "llm_editor"
    parameters = LLMEditorParameters

    @staticmethod
    def _load_editor_system_message() -> str:
        """Load system message prompt for edit command from text file."""
        path = Path(__file__).parent / "descriptions" / "llm_file_editor_prompt.txt"
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _load_viewer_system_message() -> str:
        """Load system message prompt for view command from text file."""
        path = Path(__file__).parent / "descriptions" / "llm_file_viewer_prompt.txt"
        return path.read_text(encoding="utf-8").strip()

    async def execute(self, params: LLMEditorParameters) -> dict[str, Any] | str:
        """Execute file editor command."""
        command = params.command
        path = params.path

        if command == "view":
            return await self._view(path, params.query)
        elif command == "create":
            return await self._create(path, params.file_text)
        elif command == "edit":
            return await self._edit(path, params.instruction)
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

    # ==================== EDIT COMMAND ====================

    async def _edit(self, path: str, instruction: str | None) -> dict[str, Any]:
        """Execute LLM-based file editing."""
        if instruction is None:
            return {"output": "ERROR: Parameter `instruction` is required for command: edit."}

        # Validate path
        path_obj = Path(path)
        if not path_obj.exists():
            return {"output": f"ERROR: The file {path} does not exist. Please provide a valid file path."}

        if path_obj.is_dir():
            return {"output": f"ERROR: {path} is a directory, not a file. The `edit` command only works with files."}

        # Read file content
        try:
            with open(path, encoding="utf-8") as f:
                original_content = f.read()
        except UnicodeDecodeError:
            return {"output": f"ERROR: File '{path}' appears to be binary and cannot be edited as text."}
        except Exception as e:
            return {"output": f"ERROR reading file: {str(e)}"}

        # Prepare user message with file content and instruction
        user_message = f"""File content:
```
{original_content}
```

Instruction: {instruction}

Please provide the search-replace blocks needed to make these changes."""

        # Initialize LLM (using a fast model for this task)
        try:
            llm_editor_config = get_llm_editor_config()
            api_type = llm_editor_config["API_TYPE"]
            use_responses_api = api_type != "OPENAI"
            llm = LLM(api_type=api_type, api_config=llm_editor_config, use_responses_api=use_responses_api)

            # Prepare input
            agent_input = {
                "system": self._load_editor_system_message(),
                "messages": [{"role": "user", "content": user_message}],
            }

            # Call LLM
            response = await llm.call(agent_input, max_tokens=16384)

            # Extract text content from response
            llm_output = self._extract_text_from_response(response)

        except Exception as e:
            return {"output": f"ERROR calling LLM: {str(e)}"}

        # Parse and apply edits
        try:
            result = await self._apply_edits(path, original_content, llm_output)

            # Extract token statistics from response
            token_stats = self._extract_token_stats(response)

            # Return dict with output, input, response text, and token stats
            return {
                "output": result,
                "llm_editor_system": self._load_editor_system_message(),
                "llm_editor_user": user_message,
                "llm_editor_assistant": llm_output,
                "original_content": original_content,
                "llm_editor_token_stats": token_stats,
            }
        except Exception as e:
            return {"output": f"ERROR applying edits: {str(e)}"}

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

    def _generate_diff_output(self, path: str, original_content: str, new_content: str) -> str:
        """Generate output showing the modified regions with line numbers."""
        if original_content == new_content:
            return f"No changes made to {path}."

        original_lines = original_content.splitlines()
        new_lines = new_content.splitlines()

        # Find changed line ranges in the new content
        changed_ranges = self._find_changed_ranges(original_lines, new_lines)

        # Format the output with line numbers
        formatted_output = self._format_changed_regions(new_lines, changed_ranges)

        return f"Successfully edited {path}.\n\nThe modified regions in the file are now:\n{formatted_output}"

    def _find_changed_ranges(self, original_lines: list[str], new_lines: list[str]) -> list[tuple[int, int]]:
        """Find ranges of lines that were changed, with context."""

        matcher = difflib.SequenceMatcher(None, original_lines, new_lines)
        changed_ranges = []
        context = 2  # lines of context before/after changes

        for tag, _, _, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                # Expand range to include context
                start = max(0, j1 - context)
                end = min(len(new_lines), j2 + context)
                changed_ranges.append((start, end))

        # Merge overlapping or close ranges (within 10 lines)
        if not changed_ranges:
            return []

        merged_ranges = [changed_ranges[0]]
        for start, end in changed_ranges[1:]:
            last_start, last_end = merged_ranges[-1]
            if start - last_end <= 10:  # Merge if within 10 lines
                merged_ranges[-1] = (last_start, max(last_end, end))
            else:
                merged_ranges.append((start, end))

        return merged_ranges

    def _format_changed_regions(self, lines: list[str], ranges: list[tuple[int, int]]) -> str:
        """Format changed regions with line numbers like file_editor output."""
        output_parts = []

        if len(ranges) == 1:
            # Single region - just show it
            start, end = ranges[0]
            numbered_lines = [f"{i + 1:6d}\t{line}" for i, line in enumerate(lines[start:end], start=start)]
            output_parts.append("\n".join(numbered_lines))
        else:
            # Multiple regions - show each with omission markers between
            for i, (start, end) in enumerate(ranges):
                numbered_lines = [f"{i + 1:6d}\t{line}" for i, line in enumerate(lines[start:end], start=start)]
                output_parts.append("\n".join(numbered_lines))

                # Add omission marker if not the last range
                if i < len(ranges) - 1:
                    next_start = ranges[i + 1][0]
                    omitted_lines = next_start - end
                    output_parts.append(f"\n[... {omitted_lines} lines omitted ...]\n")

        return "\n".join(output_parts)

    async def _apply_edits(self, path: str, original_content: str, llm_output: str) -> str:
        """Parse LLM output and apply search-replace edits to the file."""
        # Pattern to match search-replace blocks
        pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
        matches = list(re.finditer(pattern, llm_output, re.DOTALL))

        if not matches:
            return f"ERROR: LLM did not produce valid search-replace blocks. LLM output was:\n{llm_output}"

        # Check if this is a full rewrite (first search block is empty)
        first_search = matches[0].group(1)
        if first_search.strip() == "":
            # Full rewrite mode: must have exactly one SEARCH/REPLACE block
            if len(matches) > 1:
                return "ERROR: Full rewrite mode (empty SEARCH) requires exactly one SEARCH/REPLACE block."
            new_content = matches[0].group(2).rstrip("\n")
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_content)
            except Exception as e:
                return f"ERROR writing file: {str(e)}"
            return self._generate_diff_output(path, original_content, new_content)

        # Targeted edits: Apply each SEARCH/REPLACE block
        new_content = original_content

        for match in matches:
            search_block = match.group(1).rstrip("\n")
            replace_block = match.group(2).rstrip("\n")

            # Check if search block exists in the content
            if search_block not in new_content:
                return "ERROR: Could not find the search block in the file."

            # Check for multiple occurrences
            occurrences = new_content.count(search_block)
            if occurrences > 1:
                return f"ERROR: Found {occurrences} occurrences of the search block."

            # Apply the replacement: replace only the first occurrence
            new_content = new_content.replace(search_block, replace_block, 1)

        # Write the modified content back to the file
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)
        except Exception as e:
            return f"ERROR writing file: {str(e)}"

        return self._generate_diff_output(path, original_content, new_content)
