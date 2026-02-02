# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
from typing import Any

from dotenv import load_dotenv

from sweedit.utils import load_api_configs, load_llm_editor_config

load_dotenv()


class APIPool:
    """Manages a pool of API configurations and randomly selects one."""

    def __init__(self):
        self._pool = load_api_configs()
        if not self._pool:
            raise ValueError("No API configurations found in environment variables")

    def get_random_config(self) -> dict[str, Any]:
        """Randomly select an API configuration from the pool."""
        return random.choice(self._pool)

    def size(self) -> int:
        """Return the number of available API configurations."""
        return len(self._pool)


# Global API pool instance
_api_pool = APIPool()


def get_api_config() -> dict[str, Any]:
    """Get a random API configuration from the pool."""
    return _api_pool.get_random_config()


def get_llm_editor_config() -> dict[str, Any]:
    """Get the API configuration for the LLM file editor."""
    return load_llm_editor_config()
