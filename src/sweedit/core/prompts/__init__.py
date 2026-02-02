# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pathlib import Path

_PROMPT_DIR = Path(__file__).parent


def __getattr__(name):
    """Lazily load prompts when accessed."""
    prompt_file = _PROMPT_DIR / f"{name.lower()}.md"
    if prompt_file.exists():
        return prompt_file.read_text(encoding="utf-8").strip()
    raise AttributeError(f"Prompt '{name}' not found")
