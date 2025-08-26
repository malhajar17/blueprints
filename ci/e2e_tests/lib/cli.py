import datetime
import json
import os
import subprocess
import sys

binary = "flexai"

verbose = False

dry_run = False

global_runtime = None


class CliError(RuntimeError):
    """
    Custom error class for CLI-related errors.
    This is used to differentiate between different types of errors.
    """

    # note: __cause__ is always a subprocess.CalledProcessError with stderr and stdout as strings


class RawArg:
    """
    Represents a raw CLI argument that should be appended as-is,
    without any additional formatting or processing.
    """


def run(*args: str, timeout: int = None) -> subprocess.CompletedProcess:
    """
    Run a CLI command with the specified arguments and handle updates if necessary.

    Args:
        *args (str): Arguments to pass to the CLI.
        timeout (int, optional): Timeout for the command in seconds. Defaults to None.

    Returns:
        subprocess.CompletedProcess: The result of the command execution.

    Raises:
        CliError: If the command fails with a non-zero exit code.
    """
    for _ in range(5):
        try:
            return _run_cli(*args, timeout=timeout)
        except CliError as e:
            call_process_error = e.__cause__
            if "CLI update required" in call_process_error.stderr:
                print("Updating CLI...")
                _run_cli("update")
                continue

            raise

    raise RuntimeError("CLI update required. Updated 5 times?!")


def _run_cli(*args: str, timeout: int = None) -> subprocess.CompletedProcess:
    cmd = [binary] + list(args)

    # Get always same output
    env = dict(os.environ)
    env["DEBUG"] = "1"

    try:
        if verbose:
            print()
            print(f"=== COMMAND (timeout={timeout}) ===")
            print(" ".join(cmd))

        if dry_run:
            print("Dry run enabled, not executing the command.")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        output = subprocess.run(
            cmd, env=env, check=True, capture_output=True, text=True, timeout=timeout
        )

        if verbose:
            print(">>>>>")
            print(output.stdout.strip())
            print("====================================")
            print()

        return output
    except subprocess.CalledProcessError as e:
        msg = [f"Failed to run {e.cmd} (exit code {e.returncode}):"]

        if len(e.stderr.strip()) > 0:
            msg.append("StdErr:")
            msg.extend(map(lambda x: f"  {x}", e.stderr.splitlines()))

        if len(e.stdout.strip()) > 0:
            msg.append("StdOut:")
            msg.extend(map(lambda x: f"  {x}", e.stdout.splitlines()))

        msg = "\n".join(msg)

        raise CliError(msg) from e


def training_run(
    name: str,
    *,
    accelerators: int = None,
    nodes: int = None,
    input_checkpoint: str = None,
    dataset: str = None,
    env: dict[str, str] = None,
    secrets: dict[str, str] = None,
    repository_url: str = None,
    repository_revision: str = None,
    requirements_path: str = None,
    entry_point: str,
    model_args: dict[str, str] = {},
    runtime: str | None = None,
):
    """
    Run a training job with the specified parameters.
    """

    flags = []

    if accelerators is not None:
        flags.append(f"--accels={accelerators}")
    if nodes is not None:
        flags.append(f"--nodes={nodes}")
    if input_checkpoint is not None:
        flags.append(f"--checkpoint={input_checkpoint}")
    if dataset is not None:
        flags.append(f"--dataset={dataset}")
    if repository_url is not None:
        flags.append(f"--repository-url={repository_url}")
        if repository_revision is not None:
            flags.append(f"--repository-revision={repository_revision}")
    if requirements_path is not None:
        flags.append(f"--requirements-path={requirements_path}")

    selected_runtime = global_runtime if global_runtime else runtime
    if selected_runtime:
        flags.append(f"--runtime={selected_runtime}")

    if env is not None:
        for key, value in env.items():
            flags.append(f"--env={key}={value}")
    if secrets is not None:
        for key, value in secrets.items():
            flags.append(f"--secret={key}={value}")

    model_args_list = []
    for key, value in model_args.items():
        if isinstance(value, RawArg):
            model_args_list.append(key)
            continue

        if isinstance(value, bool):
            if value:
                model_args_list.append(f"--{key}")
            continue

        model_args_list.append(f"--{key}={str(value)}")

    try:
        run(
            "training",
            "run",
            name,
            *flags,
            "--",
            entry_point,
            *model_args_list,
        )
    except CliError:
        _raise_refined_training_error(name)


def training_inspect(name: str):
    try:
        res = run(
            "training",
            "inspect",
            name,
            "--json",
        )
    except CliError:
        _raise_refined_training_error(name)

    return json.loads(res.stdout.strip())


def training_logs(name: str, timeout: int = 300):
    try:
        res = run(
            "training",
            "logs",
            name,
            timeout=timeout,
        )
    except CliError:
        _raise_refined_training_error(name)

    return res.stdout.strip()


def training_list_checkpoints(name: str):
    """
    List checkpoints for a training job.
    """
    try:
        res = run(
            "training",
            "checkpoints",
            "--json",
            name,
        )
    except CliError:
        _raise_refined_training_error(name)

    data = json.loads(res.stdout.strip())

    # parse iso8601 timestamps to datetime objects
    for row in data:
        try:
            row["timestamp"] = datetime.datetime.strptime(
                row["timestamp"], "%Y-%m-%d %H:%M:%S.%f %z %Z"
            )
        except ValueError:
            try:
                # Try without microseconds
                row["timestamp"] = datetime.datetime.strptime(
                    row["timestamp"], "%Y-%m-%d %H:%M:%S %z %Z"
                )
            except ValueError as exc:
                raise ValueError(
                    f"Unrecognized timestamp format: {row['timestamp']}"
                ) from exc

    return data


def checkpoint_fetch(id: str, target_directory: str):
    """
    Fetch a checkpoint by its ID.
    """

    run("checkpoint", "fetch", id, "--destination", target_directory)


def checkpoint_inspect(id: str):
    """
    Inspect a checkpoint by its ID.
    """

    res = run("checkpoint", "inspect", id, "--json")

    return json.loads(res.stdout.strip())


def checkpoint_push(name: str, *, path: str, storage_provider: str = None):
    """
    Push path as a checkpoint with the given name and storage provider.

    If storage_provider is not specified, it will use local filesystem.
    """

    if storage_provider is None:
        run("checkpoint", "push", name, "--file", path)
    else:
        run(
            "checkpoint",
            "push",
            name,
            "--source-path",
            path,
            "--storage-provider",
            storage_provider,
        )


def checkpoint_delete(name: str):
    """
    Delete a checkpoint by its name.
    """

    run("checkpoint", "delete", name)


def dataset_push(name: str, *, path: str, storage_provider: str = None):
    """
    Push a dataset from the given path with the specified name and storage provider.

    If storage_provider is not specified, it will use local filesystem.
    """

    if storage_provider is None:
        run("dataset", "push", name, "--file", path)
    else:
        run(
            "dataset",
            "push",
            name,
            "--source-path",
            path,
            "--storage-provider",
            storage_provider,
        )


def dataset_inspect(name: str):
    """
    Inspect a dataset by its name.
    """

    res = run("dataset", "inspect", name, "--json")

    return json.loads(res.stdout.strip())


def dataset_delete(name: str):
    """
    Delete a dataset by its name.
    """

    run("dataset", "delete", name)


def _raise_refined_training_error(name: str):
    """
    Try to refine the error message for a training, and raise it.
    """

    try:
        e = sys.exception()
    except AttributeError:
        e = sys.exc_info()[1]

    call_process_error = e.__cause__

    if '"code":"ET0002"' in call_process_error.stderr:
        raise RuntimeError(f"Unknown training: '{name}'.") from e

    if '"code":"ET0009"' in call_process_error.stderr:
        raise RuntimeError(f"Training job '{name}' already exists.") from e

    raise
