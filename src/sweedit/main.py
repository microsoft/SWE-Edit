# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import asyncio

from sweedit.core.agenthub import MinAgent


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="SWE-Edit Minimal Agent",
    )

    parser.add_argument("task", type=str, help="Task prompt for the agent to execute")

    args = parser.parse_args()

    agent = MinAgent()
    asyncio.run(agent.run(user_message=args.task))


if __name__ == "__main__":
    main()
