# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Minimal agent"""

from sweedit.core.agent import AgentBase
from sweedit.core.prompts import MINIMAL
from sweedit.tools import ExecuteBash, Finish, StrReplaceEditor


class MinAgent(AgentBase):
    def __init__(self, session_id: str, save_dir: str = ".traj/"):
        super().__init__(
            system=MINIMAL,
            tools=[ExecuteBash(), StrReplaceEditor(), Finish()],
            session_id=session_id,
            save_dir=save_dir,
        )
