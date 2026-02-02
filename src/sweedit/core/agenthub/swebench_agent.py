# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""SWEBench agent"""

from typing import Literal

from sweedit.core.agent import AgentBase
from sweedit.tools import ExecuteBashSWEBench, Finish, LLMEditor, LLMFileEditor, StrReplaceEditor, StrReplaceEditorV2


class SwebenchAgent(AgentBase):
    def __init__(
        self,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        session_id: str | None = None,
        save_dir: str = "/tmp/.traj/swebench",
    ):
        super().__init__(
            tools=[ExecuteBashSWEBench(), StrReplaceEditor(), Finish()],
            reasoning_effort=reasoning_effort,
            max_iterations=100,
            session_id=session_id,
            save_dir=save_dir,
            rate_limit_delay=1.0,  # Add delay for evaluation to avoid rate limits
        )


class SwebenchAgentV2(AgentBase):
    def __init__(
        self,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        session_id: str | None = None,
        save_dir: str = "/tmp/.traj/swebench",
    ):
        super().__init__(
            tools=[ExecuteBashSWEBench(), StrReplaceEditorV2(), Finish()],
            reasoning_effort=reasoning_effort,
            max_iterations=100,
            session_id=session_id,
            save_dir=save_dir,
            rate_limit_delay=1.0,  # Add delay for evaluation to avoid rate limits
        )


class SwebenchAgentWithLLMFileEditor(AgentBase):
    def __init__(
        self,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        session_id: str | None = None,
        save_dir: str = "/tmp/.traj/swebench",
    ):
        super().__init__(
            tools=[ExecuteBashSWEBench(), LLMFileEditor(), Finish()],
            reasoning_effort=reasoning_effort,
            max_iterations=100,
            session_id=session_id,
            save_dir=save_dir,
            rate_limit_delay=1.0,
        )


class SwebenchAgentWithLLMEditor(AgentBase):
    def __init__(
        self,
        reasoning_effort: Literal["low", "medium", "high"] = "medium",
        session_id: str | None = None,
        save_dir: str = "/tmp/.traj/swebench",
    ):
        super().__init__(
            tools=[ExecuteBashSWEBench(), LLMEditor(), Finish()],
            reasoning_effort=reasoning_effort,
            max_iterations=100,
            session_id=session_id,
            save_dir=save_dir,
            rate_limit_delay=1.0,
        )
