# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import os

from sweedit.core.agenthub import (
    SwebenchAgent,
    SwebenchAgentV2,
    SwebenchAgentWithLLMEditor,
    SwebenchAgentWithLLMFileEditor,
)


def main():
    task_prompt = os.environ.get("TASK_PROMPT")
    instance_id = os.environ.get("INSTANCE_ID")
    workdir = os.environ.get("WORKDIR")
    reasoning_effort = os.environ.get("REASONING_EFFORT")
    save_dir = os.environ.get("SAVE_DIR")
    agent_type = os.environ.get("AGENT_TYPE", "llm-editor")

    if not task_prompt:
        raise ValueError("TASK_PROMPT environment variable not set")

    # Select agent based on agent_type
    if agent_type == "baseline":
        agent = SwebenchAgent(session_id=instance_id, reasoning_effort=reasoning_effort, save_dir=save_dir)
    elif agent_type == "baseline-v2":
        agent = SwebenchAgentV2(session_id=instance_id, reasoning_effort=reasoning_effort, save_dir=save_dir)
    elif agent_type == "llm-file-editor":
        agent = SwebenchAgentWithLLMFileEditor(
            session_id=instance_id, reasoning_effort=reasoning_effort, save_dir=save_dir
        )
    elif agent_type == "llm-editor":
        agent = SwebenchAgentWithLLMEditor(session_id=instance_id, reasoning_effort=reasoning_effort, save_dir=save_dir)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent.set_workdir(workdir)
    asyncio.run(agent.run(user_message=task_prompt))


if __name__ == "__main__":
    main()
