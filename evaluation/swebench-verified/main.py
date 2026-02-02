# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import docker
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from swebench.harness.utils import load_swebench_dataset
from tqdm import tqdm
from utils import (
    build_swebench_container,
    cleanup_container_and_logger,
    exec_run_with_timeout,
    get_git_diff,
    load_api_key_and_task_metadata,
    setup_global_logger,
    try_get_git_diff_safe,
)

from sweedit.utils import load_api_configs

# Global logger instance
glogger: logging.Logger = None


def process_instance(client, instance, template, args, traj_mount, api_config):
    """
    Process a single SWE-bench instance.

    Args:
        client: Docker client
        instance: SWE-bench instance to process
        template: Jinja2 template for rendering prompts
        args: Command line arguments
        traj_mount: Dictionary with trajectory mount configuration
        api_config: Pre-selected API configuration for this worker

    Returns:
        dict: Result dictionary with instance_id, model_name_or_path, and model_patch
    """
    global glogger

    # Record actual start time (after any queue wait)
    actual_start_time = time.time()

    # Add random delay to stagger API calls and avoid rate limits
    time.sleep(random.uniform(0, 2))

    container = None
    instance_logger = None  # Per-instance logger from swebench
    workspace_path = None
    instance_id = instance["instance_id"]

    try:
        # Set up workspace copy (like OpenHands approach)
        # Agent works in /workspace/repo/, not directly in /testbed/
        repo_name = instance["repo"].split("/")[-1]
        workspace_path = f"/workspace/{repo_name}/"

        # IMPORTANT: Set repo_path BEFORE rendering template (template uses it)
        instance["repo_path"] = workspace_path

        task_prompt = template.render(instance=instance)
        env_vars = load_api_key_and_task_metadata(
            task_prompt,
            instance_id,
            api_config,
            agent_type=args.agent_type,
            llm_editor_model=args.llm_editor_model,
            llm_viewer_model=args.llm_viewer_model,
        )

        env_vars["WORKDIR"] = workspace_path
        env_vars["REASONING_EFFORT"] = args.reasoning_effort
        env_vars["SAVE_DIR"] = args.save_dir
        env_vars["AGENT_TYPE"] = args.agent_type

        container, instance_logger = build_swebench_container(
            client,
            instance,
            namespace="swebench",
            run_id=f"sweb-{instance_id}",
            whl_file_path=args.whl_file_path,
            agent_script_path=args.agent_script_path,
            env_vars=env_vars,
            mount=traj_mount,
        )

        # Copy /testbed to workspace (pristine copy for agent to work in)
        instance_logger.info(f"Setting up workspace at {workspace_path}...")
        exec_result = container.exec_run(
            ["bash", "-c", f"mkdir -p {workspace_path} && cp -r /testbed/. {workspace_path}/"]
        )
        if exec_result.exit_code != 0:
            raise Exception(f"Failed to setup workspace: {exec_result.output.decode()}")

        # Reset git to clean state in workspace
        exec_result = container.exec_run(["bash", "-c", f"cd {workspace_path} && git reset --hard"])
        if exec_result.exit_code != 0:
            raise Exception(f"Git reset failed: {exec_result.output.decode()}")

        instance_logger.info("Running the sweedit agent...")

        agent_cmd = (
            "/root/.local/bin/uv run --python 3.12 --with /agent/sweedit-1.0.0-py3-none-any.whl /agent/run_agent.py"
        )

        exec_result = exec_run_with_timeout(
            container,
            ["bash", "-c", agent_cmd],
            timeout=args.timeout,
            logger=instance_logger,
            user="root",
            environment=env_vars,
        )

        # Check if the command completed successfully
        if exec_result.exit_code != 0:
            instance_logger.error(f"Command failed with exit code {exec_result.exit_code}")
            instance_logger.error(f"Error output: {exec_result.output.decode()}")
            raise Exception(f"sweedit agent execution failed: {exec_result.output.decode()}")
        else:
            instance_logger.info("sweedit agent completed successfully")
            instance_logger.info(f"Output: {exec_result.output.decode()}")

        # Get diff from workspace (no filtering needed - workspace is isolated)
        model_patch = get_git_diff(container, workspace_path, base_commit=instance.get("base_commit"))

        result = {
            "instance_id": instance_id,
            "model_name_or_path": args.model_name_or_path,
            "model_patch": model_patch,
            "_start_time": actual_start_time,  # For elapsed time calculation
        }

        instance_logger.info(f"Successfully processed {instance_id}")

        return result

    except TimeoutError as e:
        error_msg = f"Timeout processing {instance_id}: {str(e)}"
        glogger.warning(error_msg)
        if instance_logger:
            instance_logger.error(error_msg)

        # Try to salvage the git diff even after timeout
        model_patch = ""
        if container and workspace_path:
            model_patch = try_get_git_diff_safe(
                container, workspace_path, base_commit=instance.get("base_commit"), logger=instance_logger
            )
            if model_patch:
                glogger.warning(f"[{instance_id}] Timeout but salvaged {len(model_patch)} bytes patch")
                if instance_logger:
                    instance_logger.info(f"Salvaged git diff after timeout ({len(model_patch)} bytes)")

        # Return result with salvaged patch (may be empty)
        return {
            "instance_id": instance_id,
            "model_name_or_path": args.model_name_or_path,
            "model_patch": model_patch,
            "timeout": True,  # Mark as timed out for tracking
            "_start_time": actual_start_time,  # For elapsed time calculation
        }

    except Exception as e:
        error_msg = f"Error processing {instance_id}: {str(e)}"
        glogger.error(error_msg)
        if instance_logger:
            instance_logger.error(error_msg)
        raise
    finally:
        # Clean up
        if container and instance_logger:
            cleanup_container_and_logger(client, container, instance_logger)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_traj_path", type=str, required=True, help="The absolute local path to the agent trajectories"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="The name of this experimental run. Used to construct save_dir and output_file automatically.",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="short",
        choices=["short", "long"],
        help="The type of prompt to use",
    )
    parser.add_argument(
        "--whl_file_path",
        type=str,
        default="dist/sweedit-1.0.0-py3-none-any.whl",
        help="The path to sweedit wheel",
    )
    parser.add_argument(
        "--agent_script_path",
        type=str,
        default="evaluation/swebench-verified/run_agent.py",
        help="The path to the agent script",
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="The reasoning effort to use",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Maximum number of instances to run in parallel",
    )
    parser.add_argument(
        "--instance_ids_file",
        type=str,
        default=None,
        help="Optional path to a text file containing instance IDs to run (one ID per line)",
    )
    parser.add_argument(
        "--split_size",
        type=int,
        default=500,
        help="The size of the split to run",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Timeout in seconds (default 20 min). Worker is terminated after this.",
    )
    parser.add_argument(
        "--agent_type",
        type=str,
        default="baseline",
        choices=["baseline", "baseline-v2", "llm-file-editor", "llm-editor", "finish-with-status"],
        help="Type of agent to use: 'baseline' (str-replace editor) or 'baseline-v2' (str-replace editor v2) or 'llm-file-editor' (LLM-based file editor) or 'llm-editor' (LLM-based editor) or 'finish-with-status' (finish with status tool)",  # noqa: E501
    )
    parser.add_argument(
        "--llm_editor_model",
        type=str,
        default=None,
        choices=[
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "qwen3-4b",
            "qwen3-4b-rl",
            "qwen3-8b",
            "qwen3-8b-rl",
            "qwen3-8b-fr",
        ],
        help="Model to use for LLM editor (required for 'llm-file-editor' and 'llm-editor' agent types)",
    )
    parser.add_argument(
        "--llm_viewer_model",
        type=str,
        default=None,
        choices=[
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "qwen3-4b",
            "qwen3-4b-rl",
            "qwen3-8b",
            "qwen3-8b-rl",
            "qwen3-8b-fr",
        ],
        help="Model to use for LLM viewer (required for 'llm-editor' and 'baseline-v2' agent types)",
    )

    args = parser.parse_args()

    # Validate that required model parameters are provided for each agent type
    if args.agent_type in ["llm-file-editor", "llm-editor"] and not args.llm_editor_model:
        parser.error(f"--llm_editor_model is required when agent_type is '{args.agent_type}'")

    if args.agent_type in ["llm-editor", "baseline-v2"] and not args.llm_viewer_model:
        parser.error(f"--llm_viewer_model is required when agent_type is '{args.agent_type}'")

    # Automatically construct save_dir and output_file from run_name
    args.save_dir = f"/tmp/.traj/{args.run_name}"
    args.output_file = f"evaluation/swebench-verified/outputs/{args.run_name}.jsonl"
    # For compatibility with existing code, keep model_name_or_path as alias to run_name
    args.model_name_or_path = args.run_name

    # Setup global logger
    glogger = setup_global_logger()
    glogger.info("=" * 60)
    glogger.info("SWE-bench Evaluation Started")
    glogger.info("=" * 60)

    # Convert relative paths to absolute paths for multithreading
    args.whl_file_path = os.path.abspath(args.whl_file_path)
    args.agent_script_path = os.path.abspath(args.agent_script_path)

    # set docker client
    client = docker.from_env(timeout=600)

    # load instance IDs if specified
    instance_ids = None
    if args.instance_ids_file:
        instance_ids_path = os.path.abspath(args.instance_ids_file)
        glogger.info(f"Loading instance IDs from {instance_ids_path}")
        with open(instance_ids_path) as f:
            # Read one ID per line, strip whitespace, and ignore empty lines
            instance_ids = [line.strip() for line in f if line.strip()]
        glogger.info(f"Loaded {len(instance_ids)} instance ID(s) to process")

    # load dataset
    dataset = load_swebench_dataset("princeton-nlp/SWE-bench_Verified", instance_ids=instance_ids)[
        : args.split_size
    ]  # only run the first split_size instances
    glogger.info(f"Running {len(dataset)} instances")

    # set prompt template
    env = Environment(loader=FileSystemLoader("evaluation/swebench-verified"))
    template = env.get_template(f"prompt_{args.prompt_template}.j2")
    glogger.info(f"Using prompt template: prompt_{args.prompt_template}.j2")

    # set traj mount
    traj_mount = {
        "local_path": args.local_traj_path,
        "container_path": "/tmp/.traj",
    }

    # Load all API configs once at startup
    api_configs = load_api_configs()
    if not api_configs:
        raise ValueError("No API configurations found in environment variables")
    glogger.info(f"Loaded {len(api_configs)} API configuration(s) for round-robin assignment")

    output_file = os.path.abspath(args.output_file)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Log run configuration
    glogger.info("Configuration:")
    glogger.info(f"  - Run name: {args.run_name}")
    glogger.info(f"  - Instances: {len(dataset)}")
    glogger.info(f"  - Max workers: {args.max_workers}")
    glogger.info(f"  - Timeout: {args.timeout}s")
    glogger.info(f"  - Save dir: {args.save_dir}")
    glogger.info(f"  - Output file: {output_file}")
    glogger.info(f"  - Reasoning effort: {args.reasoning_effort}")
    glogger.info(f"  - Agent type: {args.agent_type}")
    if args.agent_type in ["llm-editor", "llm-file-editor"]:
        glogger.info(f"  - LLM editor model: {args.llm_editor_model}")
    if args.agent_type in ["llm-editor", "baseline-v2"]:
        glogger.info(f"  - LLM viewer model: {args.llm_viewer_model}")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Track submission time for each future
        future_to_instance = {}
        future_start_times = {}

        for idx, instance in enumerate(dataset):
            future = executor.submit(
                process_instance,
                client,
                instance,
                template,
                args,
                traj_mount,
                api_configs[idx % len(api_configs)],  # Round-robin assignment
            )
            future_to_instance[future] = instance
            future_start_times[future] = time.time()

        completed = 0
        failed = 0
        timed_out = 0

        with tqdm(total=len(dataset), desc="Processing instances", unit="instance") as pbar:
            for future in as_completed(future_to_instance):
                instance = future_to_instance[future]
                instance_id = instance["instance_id"]
                total_elapsed = time.time() - future_start_times[future]

                try:
                    result = future.result()

                    # Calculate actual execution time (excludes queue wait)
                    start_time = result.pop("_start_time", None)
                    exec_time = time.time() - start_time if start_time else total_elapsed
                    queue_time = total_elapsed - exec_time

                    # Check if it was a timeout with salvaged patch
                    if result.get("timeout"):
                        timed_out += 1
                        patch_len = len(result.get("model_patch", ""))
                        time_info = f"exec={exec_time:.0f}s, queued={queue_time:.0f}s"
                        if patch_len > 0:
                            glogger.warning(f"[{instance_id}] Timed out ({time_info}), salvaged {patch_len}B")
                        else:
                            glogger.warning(f"[{instance_id}] Timed out ({time_info}), no patch salvaged")

                    # Save the result (remove internal fields before saving)
                    result_to_save = {k: v for k, v in result.items() if not k.startswith("_") and k != "timeout"}
                    with open(output_file, "a") as f:
                        f.write(json.dumps(result_to_save) + "\n")

                    completed += 1
                    pbar.set_postfix({"✓": completed, "✗": failed, "⏱": timed_out}, refresh=True)
                    pbar.update(1)

                except Exception as e:
                    failed += 1
                    glogger.error(f"[{instance_id}] Failed after {total_elapsed:.1f}s: {str(e)}")
                    pbar.set_postfix({"✓": completed, "✗": failed, "⏱": timed_out}, refresh=True)
                    pbar.update(1)

    # Final summary
    glogger.info("=" * 60)
    glogger.info("Evaluation Complete!")
    glogger.info(f"  Successful: {completed}")
    glogger.info(f"  Failed: {failed}")
    glogger.info(f"  Timed out (with salvage attempt): {timed_out}")
    glogger.info(f"  Output saved to: {output_file}")
    glogger.info("=" * 60)
