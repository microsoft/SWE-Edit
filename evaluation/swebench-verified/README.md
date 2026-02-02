# SWE-Bench Verified Evaluation

## Setup
Run the following command to install the dependencies for evaluation:
```bash
uv sync --extra eval
```

Before running the agent, you should build the sweedit wheel file first.
```bash
uv build
```
This will build the sweedit wheel file and save it to the `dist` directory.

## Run
Run the following command to run the agent on the SWE-bench verified dataset. You can check `main.py` for more details.

```bash
uv run evaluation/swebench-verified/main.py \
--local_traj_path <absolute-path-to-the-agent-trajectories> \
--run_name <name-of-the-experiment> \
--reasoning_effort high \
--agent_type baseline \
--llm_editor_model gpt-5-mini \
--llm_viewer_model gpt-5-mini \
--max_workers <number-of-parallel-workers> \
--timeout <timeout-in-seconds> \
```

**agent_type**:
- llm-file-editor: baseline + LLM-based Editor
- baseline-v2: baseline + LLM-based Viewer
- llm-editor: SWE-Edit (LLM-based Viewer + LLM-based Editor)

## Evaluate
Run the following command to evaluate the agent's predictions:

```bash
RUN_NAME=<name-of-the-experiment>
uv run python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --predictions_path evaluation/swebench-verified/outputs/$RUN_NAME.jsonl \
    --max_workers 50 \
    --run_id sweb
```

## Cleanup
Manually clean up Docker containers if experiment failed to clean up itself:

```bash
bash evaluation/swebench-verified/cleanup.sh
```

This script will:
- Remove all swebench Docker containers (with confirmation prompt)
