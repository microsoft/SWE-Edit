# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import os
import tarfile
import tempfile
import threading
import traceback
from datetime import datetime
from pathlib import Path

import docker
import docker.errors
from swebench.harness.constants import DOCKER_USER, SWEbenchInstance
from swebench.harness.docker_build import BuildImageError, build_instance_image, close_logger, setup_logger
from swebench.harness.docker_utils import cleanup_container, remove_image
from swebench.harness.test_spec.test_spec import TestSpec, make_test_spec


def setup_global_logger(log_dir: str = "logs") -> logging.Logger:
    """
    Set up a global logger that outputs to both console and file.

    Args:
        log_dir: Directory to store log files

    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamp-based log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"swebench_run_{timestamp}.log")

    # Create logger
    logger = logging.getLogger("swebench_eval")
    logger.setLevel(logging.INFO)

    # Prevent propagation to root logger (avoids duplicate messages)
    logger.propagate = False

    # Clear existing handlers
    logger.handlers.clear()

    # File handler - info level for run configuration and progress
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Console handler - info level for run configuration and progress
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging evaluation configurations and special messages to {log_file}")
    return logger


def exec_run_with_timeout(container, cmd, timeout=1200, logger=None, **kwargs):
    """
    Execute a command in a container with a timeout.

    Args:
        container: Docker container
        cmd: Command to execute
        timeout: Timeout in seconds (default 1200 = 20 minutes)
        logger: Optional logger for debug output
        **kwargs: Additional arguments to pass to exec_run

    Returns:
        exec_result from container.exec_run

    Raises:
        TimeoutError: If command execution exceeds timeout
    """
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = container.exec_run(cmd, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        # Thread is still running, we've timed out, kill all python/uv processes in container to stop the agent
        try:
            kill_result = container.exec_run(["bash", "-c", "pkill -9 -f 'python|uv' || true"], user="root")
            if logger:
                logger.info(f"Killed container processes after timeout (exit_code={kill_result.exit_code})")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to kill container processes: {e}")
        raise TimeoutError(f"Command execution timed out after {timeout} seconds")

    if exception[0]:
        raise exception[0]

    return result[0]


def copy_to_container_safe(container, src: Path, dst: Path):
    """
    Thread-safe version of copy_to_container that uses temporary files
    to avoid race conditions when running with multiple workers.

    Args:
        container: Docker container to copy to
        src (Path): Source file path
        dst (Path): Destination file path in the container
    """
    # Check if destination path is valid
    if os.path.dirname(dst) == "":
        raise ValueError(f"Destination path parent directory cannot be empty!, dst: {dst}")

    # Create a unique temporary tar file to avoid race conditions
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as tmp_tar:
        tar_path = Path(tmp_tar.name)

    try:
        # Create tar file
        with tarfile.open(tar_path, "w") as tar:
            tar.add(src, arcname=dst.name)

        # Get bytes for put_archive cmd
        with open(tar_path, "rb") as tar_file:
            data = tar_file.read()

        # Make directory if necessary
        container.exec_run(f"mkdir -p {dst.parent}")

        # Send tar file to container and extract
        container.put_archive(os.path.dirname(dst), data)
    finally:
        # Clean up temporary file
        if tar_path.exists():
            tar_path.unlink()


def build_container(
    test_spec: TestSpec,
    client: docker.DockerClient,
    run_id: str,
    logger: logging.Logger,
    nocache: bool,
    force_rebuild: bool = False,
    mount: dict = None,
):
    if force_rebuild:
        remove_image(client, test_spec.instance_image_key, "quiet")
    if not test_spec.is_remote_image:
        build_instance_image(test_spec, client, logger, nocache)
    else:
        try:
            client.images.get(test_spec.instance_image_key)
        except docker.errors.ImageNotFound:
            try:
                client.images.pull(test_spec.instance_image_key)
            except docker.errors.NotFound as e:
                raise BuildImageError(test_spec.instance_id, str(e), logger) from e
            except Exception as e:
                raise Exception(f"Error occurred while pulling image {test_spec.base_image_key}: {str(e)}")  # noqa: B904

    container = None
    try:
        # Create the container
        logger.info(f"Creating container for {test_spec.instance_id}...")

        # Define arguments for running the container
        run_args = test_spec.docker_specs.get("run_args", {})
        cap_add = run_args.get("cap_add", [])

        container = client.containers.create(
            image=test_spec.instance_image_key,
            name=test_spec.get_instance_container_name(run_id),
            user=DOCKER_USER,
            detach=True,
            command="tail -f /dev/null",
            platform=test_spec.platform,
            cap_add=cap_add,
            volumes={mount["local_path"]: {"bind": mount["container_path"], "mode": "rw"}},
        )
        logger.info(f"Container for {test_spec.instance_id} created: {container.id}")
        return container
    except Exception as e:
        # If an error occurs, clean up the container and raise an exception
        logger.error(f"Error creating container for {test_spec.instance_id}: {e}")
        logger.info(traceback.format_exc())
        cleanup_container(client, container, logger)
        raise BuildImageError(test_spec.instance_id, str(e), logger) from e


def build_swebench_container(
    client: docker.DockerClient,
    instance: SWEbenchInstance,
    namespace: str,
    run_id: str = "test_run",
    whl_file_path: Path = None,
    agent_script_path: Path = None,
    env_vars: dict = None,
    mount: dict = None,
):
    """
    Builds a SWE-bench container and sets up the environment.

    Args:
        client (docker.DockerClient): Docker client
        instance (SWEbenchInstance): SWE-bench instance
        namespace (str): Namespace of the SWE-bench instance
        run_id (str): Run ID
        whl_file_path (Path): Path to the lita wheel file
        agent_script_path (Path): Path to the agent script
        env_vars (dict): Environment variables
        mount (dict): Mount a local path to the container. It should be a dictionary with the following format:
            {
                "local_path": <local_path>,
                "container_path": <container_path>,
            }
    """

    assert whl_file_path is not None, "whl_file_path is required"
    assert agent_script_path is not None, "agent_script_path is required"

    whl_file_path = Path(whl_file_path)
    agent_script_path = Path(agent_script_path)

    # set up logger
    instance_id = instance["instance_id"]
    log_dir = Path("logs") / run_id / instance_id
    log_dir.mkdir(parents=True, exist_ok=True)

    test_spec = make_test_spec(
        instance=instance,
        namespace=namespace,
        instance_image_tag="latest",
    )

    logger = setup_logger(instance_id, log_dir / "agent.log")

    # build and start the container
    container = build_container(
        test_spec=test_spec,
        client=client,
        run_id=run_id,
        logger=logger,
        nocache=False,
        force_rebuild=False,
        mount=mount,
    )
    container.start()

    # 1. Install uv system-wide in the container
    logger.info("Installing uv...")
    install_uv_cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exec_result = container.exec_run(
        ["bash", "-c", install_uv_cmd],
        user="root",
    )
    if exec_result.exit_code != 0:
        raise Exception(f"Failed to install uv: {exec_result.output.decode()}")

    # 2. Copy opencat wheel to container
    logger.info("Copying wheel file...")
    copy_to_container_safe(container, whl_file_path, Path(f"/agent/{whl_file_path.name}"))

    # 3. Copy agent script to container
    logger.info("Copying run_agent.py...")
    copy_to_container_safe(container, agent_script_path, Path(f"/agent/{agent_script_path.name}"))

    # 4. Set environment variables
    logger.info("Setting environment variables...")
    set_environment_variables(container, env_vars)

    return container, logger


def set_environment_variables(container, env_vars):
    for key, value in env_vars.items():
        container.exec_run(["bash", "-c", f"export {key}='{value}'"])
        container.exec_run(["bash", "-c", f"echo 'export {key}=\"{value}\"' >> ~/.bashrc"])


def load_api_key_and_task_metadata(
    task_prompt: str,
    instance_id: str,
    api_config: dict[str, str],
    agent_type: str = "baseline",
    llm_editor_model: str | None = None,
    llm_viewer_model: str | None = None,
) -> dict[str, str]:
    """Load API configuration and task metadata for a specific worker.

    Args:
        task_prompt: The task prompt
        instance_id: The instance ID
        api_config: Pre-selected API configuration for this worker
        agent_type: Type of agent
        llm_editor_model: Model to use for LLM editor (required for 'llm-file-editor' and 'llm-editor')
        llm_viewer_model: Model to use for LLM viewer (required for 'llm-editor' and 'baseline-v2')

    Returns:
        Environment variables dictionary with API config and task metadata
    """
    # Start with the provided API config
    env_vars = api_config.copy()

    # Load LLM editor config for agents that need it
    if agent_type in ["llm-file-editor", "llm-editor"]:
        llm_editor_config = load_llm_editor_config_container(model_type=llm_editor_model)
        env_vars.update(llm_editor_config)

    # Load LLM viewer config for agents that need it
    if agent_type in ["llm-editor", "baseline-v2"]:
        llm_viewer_config = load_llm_viewer_config_container(model_type=llm_viewer_model)
        env_vars.update(llm_viewer_config)

    # Add task metadata
    env_vars["TASK_PROMPT"] = task_prompt
    env_vars["INSTANCE_ID"] = instance_id

    return env_vars


def get_git_diff(container, repo_path="/testbed", base_commit=None):
    """
    Get git diff for changes made in the repository.

    This follows the OpenHands approach with targeted setup file cleanup:
    1. Remove setup-specific artifacts (uv.lock, pyproject.toml, tox.ini, setup.py changes)
    2. Stage all remaining changes (including deletions)
    3. Commit the changes
    4. Diff from base_commit to HEAD

    Args:
        container: Docker container
        repo_path: Path to the repository (should be workspace copy, not /testbed)
        base_commit: Base commit to diff against (if None, use HEAD~1)

    Returns:
        str: Git diff output
    """
    # Setup files that should be excluded (same as OpenHands)
    setup_files = ["pyproject.toml", "tox.ini", "setup.py"]

    # Clean up uv-specific artifacts before staging
    # Remove uv.lock (always created by uv run)
    container.exec_run(["bash", "-c", f"cd {repo_path} && rm -f uv.lock"])

    # Reset setup files to prevent environment setup changes from being included in patch
    for setup_file in setup_files:
        check_tracked = container.exec_run(
            ["bash", "-c", f"cd {repo_path} && git ls-files --error-unmatch {setup_file} 2>/dev/null"]
        )
        if check_tracked.exit_code == 0:
            # File is tracked, reset any modifications
            container.exec_run(["bash", "-c", f"cd {repo_path} && git checkout HEAD -- {setup_file}"])

    # Stage all changes including deletions
    add_cmd = f"cd {repo_path} && git add -A"
    exec_result = container.exec_run(["bash", "-c", add_cmd])
    if exec_result.exit_code != 0:
        raise Exception(f"Git add failed: {exec_result.output.decode()}")

    # Configure git user (required for commit)
    config_cmds = [
        f"cd {repo_path} && git config user.email 'evaluation@sweedit.dev'",
        f"cd {repo_path} && git config user.name 'SWE-Edit Evaluation'",
    ]
    for cmd in config_cmds:
        exec_result = container.exec_run(["bash", "-c", cmd])
        if exec_result.exit_code != 0:
            raise Exception(f"Git config failed: {exec_result.output.decode()}")

    # Commit the changes (may fail if no changes, which is fine)
    commit_cmd = f"cd {repo_path} && git commit -m 'patch' --allow-empty"
    container.exec_run(["bash", "-c", commit_cmd])

    # Get the diff, explicitly excluding setup files and uv artifacts using git pathspec
    exclude_patterns = " ".join([f"':!{f}'" for f in setup_files + ["uv.lock"]])

    if base_commit:
        diff_cmd = f"cd {repo_path} && git --no-pager diff --no-color {base_commit} HEAD -- {exclude_patterns}"
    else:
        diff_cmd = f"cd {repo_path} && git --no-pager diff --no-color HEAD~1 HEAD -- {exclude_patterns}"

    exec_result = container.exec_run(["bash", "-c", diff_cmd])

    if exec_result.exit_code == 0:
        try:
            decoded = exec_result.output.decode("utf-8")
        except UnicodeDecodeError:
            # Binary data detected - skip entire file diffs that contain binary content
            output = exec_result.output
            clean_sections = []

            # Split by "diff --git" to get individual file diffs
            sections = output.split(b"diff --git")

            for i, section in enumerate(sections):
                if i == 0 and not section.strip():
                    continue
                try:
                    decoded_section = section.decode("utf-8")
                    # Skip binary file diff entries (they can't be applied)
                    if "Binary files" in decoded_section and "differ" in decoded_section:
                        continue
                    clean_sections.append("diff --git" + decoded_section)
                except UnicodeDecodeError:
                    # This file section contains binary - skip it entirely
                    continue

            return "".join(clean_sections)

        # Even if decode succeeded, filter out binary file entries (they can't be applied)
        clean_sections = []
        sections = decoded.split("diff --git")

        for i, section in enumerate(sections):
            if i == 0 and not section.strip():
                continue
            if "Binary files" in section and "differ" in section:
                continue
            clean_sections.append("diff --git" + section)

        return "".join(clean_sections)
    else:
        raise Exception(f"Git diff failed: {exec_result.output.decode()}")


def try_get_git_diff_safe(container, repo_path="/testbed", base_commit=None, logger=None):
    """
    Attempt to get git diff safely, even if the main agent command timed out.
    This is a best-effort function that won't raise exceptions.

    Args:
        container: Docker container
        repo_path: Path to the repository
        base_commit: Base commit to diff against
        logger: Optional logger for debug output

    Returns:
        str: Git diff output, or empty string if failed
    """
    try:
        if logger:
            logger.info("Attempting to extract git diff after timeout...")
        return get_git_diff(container, repo_path, base_commit)
    except Exception as e:
        if logger:
            logger.warning(f"Failed to extract git diff after timeout: {e}")
        # Try a simpler approach - just get uncommitted changes
        try:
            cmd = f"cd {repo_path} && git --no-pager diff --no-color"
            exec_result = container.exec_run(["bash", "-c", cmd])
            try:
                decoded = exec_result.output.decode("utf-8")
            except UnicodeDecodeError:
                # Skip entire file diffs with binary content
                output = exec_result.output
                clean_sections = []
                sections = output.split(b"diff --git")

                for i, section in enumerate(sections):
                    if i == 0 and not section.strip():
                        continue
                    try:
                        decoded_section = section.decode("utf-8")
                        if "Binary files" in decoded_section and "differ" in decoded_section:
                            continue
                        clean_sections.append("diff --git" + decoded_section)
                    except UnicodeDecodeError:
                        continue

                return "".join(clean_sections)

            # Filter out binary file entries even if decode succeeded
            clean_sections = []
            sections = decoded.split("diff --git")
            for i, section in enumerate(sections):
                if i == 0 and not section.strip():
                    continue
                if "Binary files" in section and "differ" in section:
                    continue
                clean_sections.append("diff --git" + section)
            return "".join(clean_sections)
        except Exception:
            return ""


def cleanup_container_and_logger(client, container, logger):
    cleanup_container(client, container, logger)
    close_logger(logger)


def _load_llm_config_from_env(prefix: str) -> dict[str, str]:
    """Helper function to load LLM config from environment variables.

    Args:
        prefix: Environment variable prefix (e.g., "LLM_EDITOR_" or "LLM_VIEWER_")

    Returns:
        Configuration dictionary
    """
    api_type = os.getenv(f"{prefix}API_TYPE")
    if not api_type:
        raise ValueError(f"{prefix}API_TYPE must be provided in environment variables")

    config = {f"{prefix}API_TYPE": api_type}

    if api_type == "OPENAI":
        if key := os.getenv(f"{prefix}OPENAI_API_KEY"):
            config[f"{prefix}OPENAI_API_KEY"] = key
        if url := os.getenv(f"{prefix}OPENAI_BASE_URL"):
            config[f"{prefix}OPENAI_BASE_URL"] = url
        if model := os.getenv(f"{prefix}MODEL"):
            config[f"{prefix}MODEL"] = model

    elif api_type == "AZURE_OPENAI":
        if key := os.getenv(f"{prefix}AZURE_API_KEY"):
            config[f"{prefix}AZURE_API_KEY"] = key
        if endpoint := os.getenv(f"{prefix}AZURE_ENDPOINT"):
            config[f"{prefix}AZURE_ENDPOINT"] = endpoint
        if deployment := os.getenv(f"{prefix}DEPLOYMENT"):
            config[f"{prefix}DEPLOYMENT"] = deployment
        if version := os.getenv(f"{prefix}AZURE_API_VERSION"):
            config[f"{prefix}AZURE_API_VERSION"] = version

    elif api_type == "ANTHROPIC":
        if key := os.getenv(f"{prefix}ANTHROPIC_API_KEY"):
            config[f"{prefix}ANTHROPIC_API_KEY"] = key
        if model := os.getenv(f"{prefix}MODEL"):
            config[f"{prefix}MODEL"] = model

    else:
        raise ValueError(f"Unknown API type: {api_type}")

    return config


def _generate_llm_config(model_type: str, config_type: str) -> dict[str, str]:
    """Generate LLM config from JSON file.

    Args:
        model_type: Model type (e.g., "gpt-5-mini")
        config_type: Config type ("editor" or "viewer")

    Returns:
        Configuration dictionary
    """
    config_path = Path(f"api_configs/{config_type}/{model_type}.json")
    assert config_path.exists(), f"{config_type.capitalize()} config file not found: {config_path}"

    with open(config_path) as f:
        return json.load(f)


def load_llm_editor_config_container(model_type: str = None):
    """Load the API configuration for the LLM file editor.

    Args:
        model_type: Optional model type override (e.g., "gpt-5", "gpt-5-mini").
                   If not provided, reads from LLM_EDITOR_API_TYPE env var.
    """
    if model_type:
        return _generate_llm_config(model_type, "editor")
    return _load_llm_config_from_env("LLM_EDITOR_")


def load_llm_viewer_config_container(model_type: str = None):
    """Load the API configuration for the LLM file viewer.

    Args:
        model_type: Optional model type override (e.g., "gpt-5", "gpt-5-mini").
                   If not provided, reads from LLM_VIEWER_API_TYPE env var.
    """
    if model_type:
        return _generate_llm_config(model_type, "viewer")
    return _load_llm_config_from_env("LLM_VIEWER_")
