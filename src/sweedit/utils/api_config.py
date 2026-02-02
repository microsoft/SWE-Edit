# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utilities for loading API configurations from environment variables."""

import os
from typing import Any


def load_api_configs() -> list[dict[str, Any]]:
    """Load all available API configurations from environment variables.

    Supports multiple APIs by using numbered suffixes:
    - API_TYPE_1, API_TYPE_2, ...
    - OPENAI_API_KEY_1, OPENAI_API_KEY_2, ...
    - ANTHROPIC_API_KEY_1, ANTHROPIC_API_KEY_2, ...
    - AZURE_API_KEY_1, AZURE_ENDPOINT_1, DEPLOYMENT_1, ...

    Returns:
        List of API configuration dictionaries with uppercase environment variable names as keys
    """
    configs = []

    # --- Load numbered API configurations (API_TYPE_1, API_TYPE_2, etc.) ---
    i = 1
    while True:
        api_type = os.getenv(f"API_TYPE_{i}")
        if not api_type:
            break

        config = {"API_TYPE": api_type}

        if api_type == "OPENAI":
            if key := os.getenv(f"OPENAI_API_KEY_{i}"):
                config["OPENAI_API_KEY"] = key
            if url := os.getenv(f"OPENAI_BASE_URL_{i}"):
                config["OPENAI_BASE_URL"] = url
            if model := os.getenv(f"MODEL_{i}"):
                config["MODEL"] = model
        elif api_type == "AZURE_OPENAI":
            if key := os.getenv(f"AZURE_API_KEY_{i}"):
                config["AZURE_API_KEY"] = key
            if endpoint := os.getenv(f"AZURE_ENDPOINT_{i}"):
                config["AZURE_ENDPOINT"] = endpoint
            if deployment := os.getenv(f"DEPLOYMENT_{i}"):
                config["DEPLOYMENT"] = deployment
            if version := os.getenv(f"AZURE_API_VERSION_{i}"):
                config["AZURE_API_VERSION"] = version
        elif api_type == "ANTHROPIC":
            if key := os.getenv(f"ANTHROPIC_API_KEY_{i}"):
                config["ANTHROPIC_API_KEY"] = key
            if model := os.getenv(f"MODEL_{i}"):
                config["MODEL"] = model

        configs.append(config)
        i += 1

    # --- Fallback to single API configuration (backward compatibility) ---
    if not configs:
        api_type = os.getenv("API_TYPE")
        if api_type:
            config = {"API_TYPE": api_type}

            if api_type == "OPENAI":
                if key := os.getenv("OPENAI_API_KEY"):
                    config["OPENAI_API_KEY"] = key
                if url := os.getenv("OPENAI_BASE_URL"):
                    config["OPENAI_BASE_URL"] = url
                if model := os.getenv("MODEL"):
                    config["MODEL"] = model
            elif api_type == "AZURE_OPENAI":
                if key := os.getenv("AZURE_API_KEY"):
                    config["AZURE_API_KEY"] = key
                if endpoint := os.getenv("AZURE_ENDPOINT"):
                    config["AZURE_ENDPOINT"] = endpoint
                if deployment := os.getenv("DEPLOYMENT"):
                    config["DEPLOYMENT"] = deployment
                if version := os.getenv("AZURE_API_VERSION"):
                    config["AZURE_API_VERSION"] = version
            elif api_type == "ANTHROPIC":
                if key := os.getenv("ANTHROPIC_API_KEY"):
                    config["ANTHROPIC_API_KEY"] = key
                if model := os.getenv("MODEL"):
                    config["MODEL"] = model

            configs.append(config)

    return configs


def load_llm_editor_config() -> dict[str, Any]:
    """Load the API configuration for the LLM file editor."""
    api_type = os.getenv("LLM_EDITOR_API_TYPE")
    if not api_type:
        raise ValueError("LLM_EDITOR_API_TYPE must be provided in environment variables")

    config = {"API_TYPE": api_type}

    if api_type == "OPENAI":
        if key := os.getenv("LLM_EDITOR_OPENAI_API_KEY"):
            config["OPENAI_API_KEY"] = key
        if url := os.getenv("LLM_EDITOR_OPENAI_BASE_URL"):
            config["OPENAI_BASE_URL"] = url
        if model := os.getenv("LLM_EDITOR_MODEL"):
            config["MODEL"] = model

    elif api_type == "AZURE_OPENAI":
        if key := os.getenv("LLM_EDITOR_AZURE_API_KEY"):
            config["AZURE_API_KEY"] = key
        if endpoint := os.getenv("LLM_EDITOR_AZURE_ENDPOINT"):
            config["AZURE_ENDPOINT"] = endpoint
        if deployment := os.getenv("LLM_EDITOR_DEPLOYMENT"):
            config["DEPLOYMENT"] = deployment
        if version := os.getenv("LLM_EDITOR_AZURE_API_VERSION"):
            config["AZURE_API_VERSION"] = version

    elif api_type == "ANTHROPIC":
        if key := os.getenv("LLM_EDITOR_ANTHROPIC_API_KEY"):
            config["ANTHROPIC_API_KEY"] = key
        if model := os.getenv("LLM_EDITOR_MODEL"):
            config["MODEL"] = model

    else:
        raise ValueError(f"Unknown API type: {api_type}")

    return config
