# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Literal

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI

from sweedit.core.types import AgentInput, AgentResponse
from sweedit.utils import retry_with_exponential_backoff

load_dotenv()


class LLM:
    """Wrapper for OpenAI client and Response API."""

    def __init__(
        self,
        api_type: Literal["OPENAI", "AZURE_OPENAI", "ANTHROPIC"] = "OPENAI",
        api_config: dict[str, Any] | None = None,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        model: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 5,
        use_responses_api: bool = True,
    ):
        self._api_type = api_type
        self._api_config = api_config or {}
        self._reasoning_effort = reasoning_effort
        self._model_override = model
        self._timeout = timeout
        self._max_retries = max_retries
        self._use_responses_api = use_responses_api
        self._payload: dict[str, Any] = {}

        if self._api_type == "OPENAI":
            self._init_openai()
        elif self._api_type == "AZURE_OPENAI":
            self._init_azure_openai()
        elif self._api_type == "ANTHROPIC":
            self._init_anthropic()
        else:
            raise ValueError(f"Unknown api_type: {self._api_type}")

    def _init_openai(self) -> None:
        self._client = AsyncOpenAI(
            api_key=self._api_config.get("OPENAI_API_KEY"),
            base_url=self._api_config.get("OPENAI_BASE_URL"),
            timeout=self._timeout,
            max_retries=self._max_retries,
        )
        model = self._model_override or self._api_config.get("MODEL")
        if not model:
            raise ValueError("MODEL must be provided via parameter or api_config")
        self._payload["model"] = model
        if self._use_responses_api:
            self._payload["reasoning"] = {"effort": self._reasoning_effort}

    def _init_azure_openai(self) -> None:
        self._client = AsyncAzureOpenAI(
            api_key=self._api_config.get("AZURE_API_KEY"),
            api_version=self._api_config.get("AZURE_API_VERSION", "2025-03-01-preview"),
            azure_endpoint=self._api_config.get("AZURE_ENDPOINT"),
            timeout=self._timeout,
            max_retries=self._max_retries,
        )
        model = self._model_override or self._api_config.get("DEPLOYMENT")
        if not model:
            raise ValueError("MODEL must be provided via parameter or api_config")
        self._payload["model"] = model
        if self._use_responses_api:
            self._payload["reasoning"] = {"effort": self._reasoning_effort}

    def _init_anthropic(self) -> None:
        self._client = AsyncAnthropic(
            api_key=self._api_config.get("ANTHROPIC_API_KEY"),
            timeout=self._timeout,
            max_retries=self._max_retries,
        )
        model = self._model_override or self._api_config.get("MODEL")
        if not model:
            raise ValueError("MODEL must be provided via parameter or api_config")
        self._payload["model"] = model
        self._payload["thinking"] = {"type": "enabled", "budget_tokens": 8192}

    @property
    def model(self) -> str:
        return self._payload["model"]

    @retry_with_exponential_backoff(max_retries=10, initial_delay=2.0, max_delay=60.0)
    async def call(
        self,
        prepared_input: AgentInput,
        max_tokens: int = 32768,
    ) -> AgentResponse:
        payload = self._payload.copy()

        if prepared_input.get("tools", None):
            payload["tools"] = prepared_input["tools"]

        if self._api_type in ["OPENAI", "AZURE_OPENAI"]:
            if self._use_responses_api:
                # Use Responses API
                if prepared_input.get("system", None):
                    payload["instructions"] = prepared_input["system"]
                payload["input"] = prepared_input["messages"]
                payload["max_output_tokens"] = max_tokens
                return await self._client.responses.create(**payload)
            else:
                # Use Chat Completions API
                messages = []
                if prepared_input.get("system", None):
                    messages.append({"role": "system", "content": prepared_input["system"]})
                messages.extend(prepared_input["messages"])
                payload["messages"] = messages
                payload["max_tokens"] = max_tokens
                return await self._client.chat.completions.create(**payload)
        else:
            if prepared_input.get("system", None):
                payload["system"] = prepared_input["system"]
            payload["messages"] = prepared_input["messages"]
            payload["max_tokens"] = max_tokens
            return await self._client.messages.create(**payload)
