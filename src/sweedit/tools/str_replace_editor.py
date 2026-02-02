# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path
from typing import Literal

from pydantic import Field

from sweedit.tools.base import Params, Tool

MAX_LINES_TO_DISPLAY = 500


class StrReplaceEditorParameters(Params):
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
    view_range: list[int] | None = Field(
        default=None,
        description=(
            "Optional parameter of `view` command when `path` points to a "
            "file. If none is given, the full file is shown. If provided, "
            "the file will be shown in the indicated line number range, "
            "e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to "
            "start. Setting `[start_line, -1]` shows all lines from "
            "`start_line` to the end of the file."
        ),
    )


class StrReplaceEditor(Tool):
    name = "str_replace_editor"
    parameters = StrReplaceEditorParameters

    async def execute(self, params: StrReplaceEditorParameters) -> str:
        """Execute str_replace_editor command."""
        command = params.command
        path = params.path

        if command == "view":
            return await self._view(path, params.view_range)
        elif command == "create":
            return await self._create(path, params.file_text)
        elif command == "str_replace":
            return await self._str_replace(path, params.old_str, params.new_str)
        elif command == "insert":
            return await self._insert(path, params.insert_line, params.new_str)
        else:
            return f"ERROR: Unknown command '{command}'"

    async def _view(self, path: str, view_range: list[int] | None = None) -> str:
        """View file or directory contents."""
        path_obj = Path(path)

        if not path_obj.exists():
            return f"ERROR: Invalid `path` parameter: The path {path} does not exist. Please provide a valid path."

        if path_obj.is_dir():
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
        else:
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

                    # Check if the requested range exceeds display limit
                    requested_lines = end_idx - start_idx
                    if requested_lines > MAX_LINES_TO_DISPLAY:
                        # Apply clipping even for explicit ranges that are too large
                        half = MAX_LINES_TO_DISPLAY // 2
                        head_lines = lines[start_idx : start_idx + half]
                        tail_lines = lines[end_idx - half : end_idx]

                        numbered_head = [
                            f"{i + start_idx + 1:6d}\t{line.rstrip()}" for i, line in enumerate(head_lines)
                        ]
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
