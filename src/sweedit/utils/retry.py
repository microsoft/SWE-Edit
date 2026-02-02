# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Retry utilities for handling transient errors."""

import asyncio
from functools import wraps

import anthropic
import openai


def retry_with_exponential_backoff(max_retries: int = 10, initial_delay: float = 2.0, max_delay: float = 60.0):
    """Decorator to retry async functions with exponential backoff on transient errors.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay in seconds between retries

    Returns:
        Decorated async function with retry logic
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (
                    openai.RateLimitError,
                    anthropic.RateLimitError,
                    openai.APITimeoutError,  # Retry on timeout (network issues, slow response)
                ) as e:
                    last_exception = e
                    if attempt == max_retries - 1:
                        raise

                    # Exponential backoff with jitter
                    wait_time = min(delay * (2**attempt), max_delay)
                    jitter = wait_time * 0.1
                    actual_wait = wait_time + jitter

                    error_type = "Rate limit" if "RateLimit" in type(e).__name__ else "API timeout"
                    print(f"{error_type} hit (attempt {attempt + 1}/{max_retries}). Retrying in {actual_wait:.2f}s...")
                    await asyncio.sleep(actual_wait)

            raise last_exception

        return wrapper

    return decorator
