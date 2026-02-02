# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from pydantic import Field

from sweedit.tools.base import Params, Tool


class FinishParameters(Params):
    message: str = Field(description="Brief status: 'Done' or reason blocked (e.g., 'Blocked: missing X').")


class Finish(Tool):
    name = "finish"
    parameters = FinishParameters

    async def execute(self, params: FinishParameters) -> str:
        return "success"
