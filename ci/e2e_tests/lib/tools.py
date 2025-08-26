import argparse
import os
import random
import string
import sys
import time
from typing import Callable

import __main__

from . import cli


class TrainingRunError(RuntimeError):
    """
    Custom exception for errors during training runs.
    This can be used to differentiate between different types of errors.
    """

    training_name: str
    status: str

    def __init__(self, training_name: str, status: str):
        match status:
            case "rejected":
                message = f"Training run '{training_name}' was rejected."
            case "failed":
                message = f"Training run '{training_name}' failed."
            case "interrupted":
                message = f"Training run '{training_name}' was interrupted."

        super().__init__(message)

        self.training_name = training_name
        self.status = status


def setup(*options: Callable[[argparse.ArgumentParser], None]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    parser.add_argument("-b", "--binary", help="Set the CLI binary to use")

    parser.add_argument("--dry-run", action="store_true", help="Enable dry run mode")

    parser.add_argument("-r", "--runtime", help="Set the runtime to use")

    for option in options:
        option(parser)

    args = parser.parse_args()

    if args.verbose:
        cli.verbose = True
    if args.binary:
        cli.binary = args.binary
    if args.dry_run:
        cli.dry_run = True
    if args.runtime:
        cli.global_runtime = args.runtime
    return args


def gen_training_name(model: str = None, *, nodes=1, accelerators=1) -> str:
    """
    Generate a unique training name.

    Returns:
        str: A unique training name.
    """
    if model is None:
        # Infer model name from the script name
        base_name = os.path.splitext(os.path.basename(__main__.__file__))[0]
        model = f"{base_name}-{nodes}x{accelerators}-e2e"

    suffix = os.getenv("TRAINING_SUFFIX")
    if not suffix:
        suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
        print(f"TRAINING_SUFFIX not set, using random suffix: {suffix}")

    return f"{model}-{suffix}"


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
    wait_timeout: int = 1800,
    runtime: str | None = None,
):
    """
    Run a training job with the specified parameters.

    Wait for the training to complete.

    If any error occurs during training, it will:
    - dump training logs if available for debugging,
    - raise an error.

    Args:
        name (str): The name of the training run.
        accelerators (int): Number of accelerators to use.
        nodes (int): Number of nodes to use.
        input_checkpoint (str): The input checkpoint ID.
        dataset (str): The dataset to use for training.
        env (dict[str, str]): Environment variables to set for the training job.
        secrets (dict[str, str]): Secrets to set for the training job.
        repository_url (str): The URL of the repository containing the training code.
        repository_revision (str): The revision of the repository to use.
        requirements_path (str): Path to the requirements file to install dependencies.
        entry_point (str): The entry point script for the training job.
        model_args (dict[str, str]): Additional model arguments.
        wait_timeout (int): The maximum time to wait for the training to complete in seconds. Default is 1800 seconds.
        runtime (str | None): The runtime to use for the training job. If None, uses the global runtime.

    Returns:
        None
    """

    cli.training_run(
        name=name,
        accelerators=accelerators,
        nodes=nodes,
        input_checkpoint=input_checkpoint,
        dataset=dataset,
        env=env,
        secrets=secrets,
        repository_url=repository_url,
        repository_revision=repository_revision,
        requirements_path=requirements_path,
        entry_point=entry_point,
        model_args=model_args,
        runtime=runtime,
    )

    print(f"Training started with name: {name}")

    try:
        wait_for_training(name=name, timeout=wait_timeout)

    except TrainingRunError as e:
        # Try to fetch logs on failed training, to dump them
        if e.status == "failed":
            try:
                logs = TrainingLogs.fetch(training_name=name)
                # On CI, stderr and stdout can be interleaved, so we dump logs to stderr
                # to keep the output ordered and clean.
                logs.dump(file=sys.stderr)
            except BaseException:
                # If fetching logs fails, to not throw here. Print an error message
                print(f"Failed to fetch logs for training '{name}'")

        raise


def wait_for_training(*, name: str, timeout: int = 1800) -> None:
    """
    Wait for a training run to complete.

    Args:
        name (str): The name of the training run.
        timeout (int): The maximum time to wait in seconds. Default is 1800 seconds.
    """
    start_time = time.time()
    while True:
        res = cli.training_inspect(name=name)

        status = res["runtime"]["status"]
        if status is None:
            raise RuntimeError("Failed to get training status")

        print(f"Training run '{name}' status: {status}")

        match status:
            case "succeeded":
                print(f"Training run '{name}' succeeded.")
                return

            case "rejected" | "failed" | "interrupted":
                raise TrainingRunError(training_name=name, status=status)

        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise TimeoutError(
                f"Training run '{name}' did not complete within the timeout period of {timeout} seconds."
            )

        time.sleep(10)  # Wait for 10 seconds before checking again


def checkpoint_push(
    name: str, *, path: str, storage_provider: str = None, timeout: int = 1800
):
    """
    Push a checkpoint to the server, and wait it to be available.

    Args:
        name (str): The name to use for the uploaded checkpoint.
        path (str): The path to the checkpoint directory.
        storage_provider (str): The storage provider to use for the dataset.
        timeout (int): The maximum time to wait in seconds. Default is 1800 seconds.
    """
    print(f"Pushing checkpoint '{name}' from '{path}'...")

    cli.checkpoint_push(name, path=path, storage_provider=storage_provider)

    wait_for_checkpoint(name=name, timeout=timeout)

    print(f"Checkpoint '{name}' pushed successfully.")


def checkpoint_exists(id: str) -> bool:
    """
    Check if a checkpoint exists by its ID.

    Args:
        id (str): The ID of the checkpoint.

    Returns:
        bool: True if the checkpoint exists, False otherwise.
    """
    try:
        cli.checkpoint_inspect(id=id)
        return True
    except cli.CliError as e:
        if "Unknown Checkpoint" in e.__cause__.stderr:
            return False

        if "Failed to get checkpoint" in e.__cause__.stderr:
            # This happens on deleted checkpoints, see PAAS-2156
            return False

        raise  # Re-raise unexpected errors


def wait_for_checkpoint(name: str, timeout: int = 1800) -> None:
    """
    Wait for a checkpoint to be created.

    Args:
        name (str): The name of the training run.
        timeout (int): The maximum time to wait in seconds. Default is 1800 seconds.
    """
    start_time = time.time()
    while True:
        res = cli.checkpoint_inspect(id=name)
        match res["status"]["status"]:
            case "available":
                return

            case "failed":
                raise RuntimeError(f"Checkpoint '{name}' failed to be created.")

        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise TimeoutError(
                f"No checkpoint found for training '{name}' within {timeout} seconds."
            )

        time.sleep(10)  # Wait for 10 seconds before checking again


def dataset_push(
    name: str, *, path: str, storage_provider: str = None, timeout: int = 1800
):
    """
    Push a dataset to the server, and wait it to be available.

    Args:
        name (str): The name to use for the uploaded dataset.
        path (str): The path to the dataset directory.
        storage_provider (str): The storage provider to use for the dataset.
        timeout (int): The maximum time to wait in seconds. Default is 1800 seconds.
    """
    print(
        f"Pushing dataset '{name}' from '{path}' (storage_provider={storage_provider})..."
    )

    cli.dataset_push(name, path=path, storage_provider=storage_provider)

    wait_for_dataset(name=name, timeout=timeout)

    print(f"Dataset '{name}' pushed successfully.")


def wait_for_dataset(name: str, timeout: int = 1800) -> None:
    """
    Wait for a dataset to be created.

    Args:
        name (str): The name of the dataset.
        timeout (int): The maximum time to wait in seconds. Default is 1800 seconds.
    """
    start_time = time.time()
    while True:
        res = cli.dataset_inspect(name=name)
        match res["status"]["status"]:
            case "available":
                return

            case "failed":
                raise RuntimeError(f"Dataset '{name}' failed to be created.")

        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            raise TimeoutError(
                f"No dataset found for '{name}' within {timeout} seconds."
            )

        time.sleep(10)  # Wait for 10 seconds before checking again


def dataset_exists(name: str) -> bool:
    """
    Check if a dataset exists by its name.

    Args:
        name (str): The name of the dataset.

    Returns:
        bool: True if the dataset exists, False otherwise.
    """
    try:
        cli.dataset_inspect(name=name)
        return True
    except cli.CliError as e:
        if "Unknown Dataset" in e.__cause__.stderr:
            return False

        if "Failed to get dataset" in e.__cause__.stderr:
            # This happens on deleted datasets, see PAAS-2156
            return False

        raise  # Re-raise unexpected errors


class Not:
    """
    Class to handle negation in TrainingLogs.assert_contains assertions.
    """

    def __init__(self, value: str):
        self.value = value


class TrainingLogs:
    """
    Class to handle training logs.
    """

    _logs: str = None

    @staticmethod
    def fetch(training_name: str) -> "TrainingLogs":
        """
        Fetch training logs for a given training name.

        Args:
            training_name (str): The name of the training run.

        Returns:
            TrainingLogs: An instance of TrainingLogs with fetched logs.
        """
        print("Fetching logs...")

        obj = TrainingLogs()
        obj._logs = cli.training_logs(name=training_name)
        return obj

    def dump(self, file=sys.stdout) -> None:
        """
        Dump the training logs to a file.

        Args:
            file (str): The file path to dump the logs.
        """
        print("=== TRAINING LOGS ===", file=file)
        print(self._logs, file=file)
        print("=====================", file=file)

    def assert_contains(self, *phrases: str | Not) -> None:
        """
        Assert that the training logs contain all specified phrases.

        Args:
            phrases (list of str): Phrases to check in the logs.

        Raises:
            AssertionError: If any of the phrases are not found in the logs.
        """
        print("Checking log content...")

        for phrase in phrases:
            if isinstance(phrase, Not):
                assert (
                    phrase.value not in self._logs
                ), f"'{phrase.value}' found in logs."
            else:
                assert phrase in self._logs, f"'{phrase}' not found in logs."

        print("Phrases check passed in logs.")


class ExpectedCheckpoint:
    """
    Class to provide details on what is expected on a checkpoint
    """

    def __init__(self, *, name: str = None, node: str = None, files: list[str] = None):
        self.name = name
        self.node = node
        self.files = files


class _ActualCheckpoint:
    """
    Reflect of ExpectedCheckpoint, with corresponding actual values
    """

    def __init__(self, expected: ExpectedCheckpoint, checkpoint_data):
        self.name = None
        self.node = None
        self.files = None

        # Fetch only data if there are matching expectations
        if expected.name is not None:
            self.name = checkpoint_data.get("name")

        if expected.node is not None:
            self.node = checkpoint_data.get("node")

        if expected.files is not None:
            self.fetch_files(checkpoint_data.get("id"))

    def fetch_files(self, checkpoint_id: str):
        """
        Fetch the files from the checkpoint.
        """
        data = cli.checkpoint_inspect(id=checkpoint_id)

        self.files = dict()

        for item in data["status"]["files"]:
            path = item["path"]
            size = item["size"]
            self.files[path] = size

    def __str__(self):
        value = ""
        if self.name is not None:
            value += f"name='{self.name}' "
        if self.node is not None:
            value += f"node='{self.node}' "
        if self.files is not None:
            files = sorted(self.files.keys())
            value += f"files={files}"
        return value


def assert_checkpoints(training_name: str, expected: list[ExpectedCheckpoint]):
    print("Fetching checkpoints...")

    data = cli.training_list_checkpoints(name=training_name)

    # Sort checkpoints by node, name (empty names last)
    actual = sorted(data, key=lambda x: (x["node"], x["name"] == "", x["name"]))

    assert len(actual) == len(expected), (
        f"Expected {len(expected)} checkpoints, got {len(actual)}\n"
        + "\n"
        + "Actual checkpoints:\n"
        + "\n".join([f" - '{c['name']}' on node {c['node']}" for c in actual])
    )

    # Build actual data
    for index, act in enumerate(actual):
        act = _ActualCheckpoint(expected[index], act)
        actual[index] = act

    # Build list of differences
    diffs = []

    for index, act, exp in zip(range(len(actual)), actual, expected):
        if act.name != exp.name:
            diffs.append(
                (
                    f"Expected checkpoint name '{exp.name}', got '{act.name}' for checkpoint #{index}"
                )
            )

        if act.node != exp.node:
            diffs.append(
                (
                    f"Expected checkpoint node '{exp.node}', got '{act.node}' for checkpoint #{index}"
                )
            )

        if not _match_files(act, exp):
            diffs.append(
                (
                    f"Checkpoint #{index} ({act.name}) on node {act.node} has unexpected files"
                )
            )

    assert not diffs, (
        "\n"
        + "Issues found:\n"
        + "".join(f"- {d}\n" for d in diffs)
        + "\n"
        + "Actual checkpoints:\n"
        + "\n".join([f" - {c}" for c in actual])
    )

    print("All expected checkpoints exist.")


def _match_files(act: _ActualCheckpoint, exp: ExpectedCheckpoint) -> bool:
    """
    Check if the actual files match the expected files.
    """

    if exp.files is None:
        return True  # No expectations

    for file in exp.files:
        if act.files.get(file) is None:
            return False  # Expected file not found

    return True
