# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import re

from pydantic import Field

from sweedit.tools.base import Params, Tool

# Output limits
MAX_OUTPUT_CHARS = 50_000
MAX_LINE_LENGTH = 2_000


def _truncate_line(line: str, max_length: int = MAX_LINE_LENGTH) -> tuple[str, bool]:
    """Truncate a line if it exceeds max_length. Returns (truncated_line, was_truncated)."""
    if len(line) <= max_length:
        return line, False
    # Preserve line breaks at the end
    match = re.search(r"[\r\n]+$", line)
    linebreak = match.group(0) if match else ""
    marker = "[...truncated]"
    end = marker + linebreak
    return line[: max_length - len(end)] + end, True


def _truncate_output(text: str, max_chars: int = MAX_OUTPUT_CHARS) -> tuple[str, bool]:
    """Truncate output text. Returns (truncated_text, was_truncated)."""
    lines = text.splitlines(keepends=True)
    # Handle text with no newlines
    if not lines and text:
        lines = [text]

    result = []
    total_chars = 0
    truncated = False

    for line in lines:
        if total_chars >= max_chars:
            truncated = True
            break

        remaining = max_chars - total_chars
        # Truncate line if it exceeds remaining space or MAX_LINE_LENGTH
        if len(line) > remaining:
            line, _ = _truncate_line(line, remaining)
            result.append(line)
            truncated = True
            break

        # Apply per-line truncation
        truncated_line, was_truncated = _truncate_line(line)
        if was_truncated:
            truncated = True
        result.append(truncated_line)
        total_chars += len(truncated_line)

    return "".join(result), truncated


class ExecuteBashParameters(Params):
    command: str = Field(
        description=(
            "The bash command to execute. You can only execute one "
            "bash command at a time. If you need to run multiple "
            "commands sequentially, you can use `&&` or `;` to chain "
            "them together."
        )
    )
    timeout: int = Field(
        default=120,
        description=(
            "Optional timeout in seconds for the command execution. "
            "If the command takes longer than this, it will be terminated."
        ),
        ge=1,
        le=300,
    )


class ExecuteBash(Tool):
    name = "execute_bash"
    parameters = ExecuteBashParameters
    swebench_mode = False  # Default: normal mode
    conda_prefix = ". /opt/miniconda3/etc/profile.d/conda.sh; conda activate testbed"
    workdir = None

    async def execute(self, params: ExecuteBashParameters) -> str:
        """Execute a bash command and return its output."""
        timeout = params.timeout

        # Prefix command with conda activation if in swebench mode
        command = params.command
        if self.swebench_mode:
            command = f"{self.conda_prefix} && {command}"

        try:
            # Run the command using subprocess
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                shell=True,
                cwd=self.workdir,
            )

            # Wait for the command to complete with timeout
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except TimeoutError:
                # Kill the process if it times out
                process.kill()
                await process.wait()
                return f"Command timed out after {timeout} seconds. Consider increasing the timeout parameter."

            # Decode the output
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")
            exit_code = process.returncode

            # Truncate outputs
            stdout_str, stdout_truncated = _truncate_output(stdout_str)
            stderr_str, stderr_truncated = _truncate_output(stderr_str)

            # Construct the output message
            output_parts = []
            if stdout_str:
                output_parts.append(stdout_str.rstrip())
            if stderr_str:
                output_parts.append(f"STDERR:\n{stderr_str.rstrip()}")

            output = "\n".join(output_parts) if output_parts else ""

            # Add metadata footer
            metadata = []
            if stdout_truncated or stderr_truncated:
                metadata.append(f"[Output truncated: exceeded {MAX_OUTPUT_CHARS} character limit]")
            metadata.append(f"Exit Code: {exit_code}")

            if output:
                output += "\n\n" + "\n".join(metadata)
            else:
                output = "\n".join(metadata)

            return output

        except Exception as e:
            return f"Error executing command: {str(e)}"

    def set_workdir(self, workdir: str):
        self.workdir = workdir


class ExecuteBashSWEBench(ExecuteBash):
    """ExecuteBash tool configured for SWEBench with conda environment activation."""

    swebench_mode = True
    workdir = "/workspace"
