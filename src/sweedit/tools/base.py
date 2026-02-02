# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError
from pydantic.json_schema import GenerateJsonSchema


class NoTitleGenerateJsonSchema(GenerateJsonSchema):
    def field_title_should_be_set(self, schema):
        return False


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Tool(ABC):
    name: str
    description: str
    parameters: type[Params]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "name"):
            raise TypeError(f"{cls.__name__} must define 'name' class attribute")
        if not hasattr(cls, "parameters"):
            raise TypeError(f"{cls.__name__} must define 'parameters' class attribute")

        cls.description = cls._load_description(cls.name)

    @staticmethod
    def _load_description(name: str) -> str:
        """Load description from markdown file."""
        path = Path(__file__).parent / "descriptions" / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(f"Description file not found: {path}")
        description = path.read_text(encoding="utf-8").strip()
        if not description:
            raise ValueError(f"Description file is empty: {path}")
        return description

    async def __call__(self, arguments: str | dict[str, Any]) -> Any:
        """Parse and validate arguments, then delegate to execute"""
        try:
            if isinstance(arguments, str):
                params = self.parameters.model_validate_json(arguments)
            elif isinstance(arguments, dict):
                params = self.parameters.model_validate(arguments)
        except ValidationError as e:
            error_messages = [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
            return f"Invalid function arguments: {'; '.join(error_messages)}"
        except json.JSONDecodeError as e:
            return f"Invalid JSON in arguments: {str(e)}"

        return await self.execute(params)

    @abstractmethod
    async def execute(self, params: Params) -> Any:
        """Execute the tool with validated parameters"""
        raise NotImplementedError

    @classmethod
    def schema(cls, api_type: Literal["OPENAI", "AZURE_OPENAI", "ANTHROPIC"] = "OPENAI") -> dict[str, Any]:
        """Return tool schema for specified API type."""
        parameters_schema = cls.parameters.model_json_schema(schema_generator=NoTitleGenerateJsonSchema)
        parameters_schema.pop("title", None)

        if api_type in ("OPENAI", "AZURE_OPENAI"):
            # OpenAI Response API function calling schema
            return {
                "type": "function",
                "name": cls.name,
                "description": cls.description,
                "parameters": parameters_schema,
            }
        elif api_type == "ANTHROPIC":
            # Anthropic Messages API function calling schema
            return {
                "name": cls.name,
                "description": cls.description,
                "input_schema": parameters_schema,
            }
        else:
            raise ValueError(f"Unknown api_type: {api_type}")
