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


class LLMFileEditorParameters(Params):
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
    view_range: list[int] | None = Field(
        default=None,
        description=(
            "Optional for `view` command when `path` points to a file. "
            "If none is given, the full file is shown. If provided, "
            "the file will be shown in the indicated line number range, "
            "e.g. [11, 12] will show lines 11 and 12. Indexing starts at 1. "
            "Setting `[start_line, -1]` shows all lines from `start_line` to the end."
        ),
    )


class LLMFileEditor(Tool):
    name = "llm_file_editor"
    parameters = LLMFileEditorParameters

    @staticmethod
    def _load_system_message() -> str:
        """Load system message prompt from text file."""
        path = Path(__file__).parent / "descriptions" / "llm_file_editor_prompt.txt"
        return path.read_text(encoding="utf-8").strip()

    async def execute(self, params: LLMFileEditorParameters) -> dict[str, Any] | str:
        """Execute file editor command."""
        command = params.command
        path = params.path

        if command == "view":
            return await self._view(path, params.view_range)
        elif command == "create":
            return await self._create(path, params.file_text)
        elif command == "edit":
            return await self._edit(path, params.instruction)
        else:
            return f"ERROR: Unknown command '{command}'"

    # ==================== VIEW COMMAND ====================

    async def _view(self, path: str, view_range: list[int] | None = None) -> str:
        """View file or directory contents."""
        path_obj = Path(path)

        if not path_obj.exists():
            return f"ERROR: Invalid `path` parameter: The path {path} does not exist. Please provide a valid path."

        if path_obj.is_dir():
            return await self._view_directory(path, path_obj)
        else:
            return await self._view_file(path, path_obj, view_range)

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

    async def _view_file(self, path: str, path_obj: Path, view_range: list[int] | None = None) -> str:
        """View file contents with optional line range."""
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()

            total_lines = len(lines)

            if view_range:
                start = view_range[0]
                end = view_range[1] if len(view_range) > 1 else -1

                if end == -1:
                    end = len(lines)

                start_idx = max(0, start - 1)
                end_idx = min(len(lines), end)

                if start_idx >= len(lines):
                    return f"ERROR: Start line {start} is beyond file length ({len(lines)} lines)"

                # Check if the requested range is effectively the full file and would exceed display limit
                requested_lines = end_idx - start_idx
                if requested_lines > MAX_LINES_TO_DISPLAY:
                    # Apply clipping even for explicit ranges that are too large
                    half = MAX_LINES_TO_DISPLAY // 2
                    head_lines = lines[start_idx : start_idx + half]
                    tail_lines = lines[end_idx - half : end_idx]

                    numbered_head = [f"{i + start_idx + 1:6d}\t{line.rstrip()}" for i, line in enumerate(head_lines)]
                    numbered_tail = [
                        f"{i + end_idx - half + 1:6d}\t{line.rstrip()}" for i, line in enumerate(tail_lines)
                    ]

                    clipped_start = start_idx + half + 1
                    clipped_end = end_idx - half

                    numbered_lines = (
                        numbered_head
                        + [f"\n... ({clipped_end - clipped_start + 1} lines omitted) ...\n"]
                        + numbered_tail
                    )
                else:
                    lines = lines[start_idx:end_idx]
                    numbered_lines = [f"{i + start_idx + 1:6d}\t{line.rstrip()}" for i, line in enumerate(lines)]
            else:
                if total_lines > MAX_LINES_TO_DISPLAY:
                    half = MAX_LINES_TO_DISPLAY // 2
                    head_lines = lines[:half]
                    tail_lines = lines[-half:]

                    numbered_head = [f"{i + 1:6d}\t{line.rstrip()}" for i, line in enumerate(head_lines)]
                    numbered_tail = [
                        f"{i + total_lines - half + 1:6d}\t{line.rstrip()}" for i, line in enumerate(tail_lines)
                    ]

                    numbered_lines = (
                        numbered_head
                        + [f"\n... ({total_lines - MAX_LINES_TO_DISPLAY} lines omitted) ...\n"]
                        + numbered_tail
                    )
                else:
                    numbered_lines = [f"{i + 1:6d}\t{line.rstrip()}" for i, line in enumerate(lines)]

            return "\n".join(numbered_lines)
        except UnicodeDecodeError:
            return f"ERROR: File '{path}' appears to be binary and cannot be viewed as text."
        except Exception as e:
            return f"ERROR: {str(e)}"

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
                "system": self._load_system_message(),
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
                "llm_editor_system": self._load_system_message(),
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
