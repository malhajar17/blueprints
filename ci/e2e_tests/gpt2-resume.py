#!/usr/bin/env python3

import os
import sys

from lib import cli, tools

checkpoint_name = "ci-gpt2-ckpt6"


def main():
    args = tools.setup(
        lambda args: args.add_argument(
            "--clean",
            action="store_true",
            help="Clean up the test environment before the test.",
        )
    )

    if args.clean:
        clean()

    checkpoint_id = prepare()

    training_name = tools.gen_training_name()

    tools.training_run(
        name=training_name,
        dataset="ci-gpt2-tokenized-wikitext",
        input_checkpoint=checkpoint_id,
        repository_url="https://github.com/flexaihq/fcs-experiments-private.git",
        repository_revision=os.getenv("TRAINING_REVISION", "main"),
        requirements_path="code/causal-language-modeling/requirements.txt",
        entry_point="code/causal-language-modeling/train.py",
        runtime="nvidia-25.06",
        model_args={
            "do_eval": True,
            "do_train": True,
            "dataset_name": "wikitext",
            "tokenized_dataset_load_dir": "/input/ci-gpt2-tokenized-wikitext",
            "model_name_or_path": "/input-checkpoint",
            "resume_from_checkpoint": "/input-checkpoint",
            "output_dir": "/output-checkpoint",
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "max_steps": 99,
            "eval_strategy": "steps",
        },
    )

    logs = tools.TrainingLogs.fetch(training_name)
    try:
        logs.assert_contains(
            "Loading tokenized dataset from:",
            "Continuing training from global step 5",
            "***** eval metrics *****",
            "Total train batch size (w. parallel, distributed & accumulation) = 8",
        )

        print("Fetching checkpoints...")

        checkpoints = cli.training_list_checkpoints(name=training_name)

        assert len(checkpoints) == 2, f"Expected 2 checkpoints, got {len(checkpoints)}"

        for item in checkpoints:
            checkpoint = tools.Checkpoint(item["id"])
            checkpoint.assert_exist("model.safetensors")

        print("Training done successfully!")

    except:
        # On CI, stderr and stdout can be interleaved, so we dump logs to stderr
        # to keep the output ordered and clean.
        logs.dump(file=sys.stderr)
        raise


def clean():
    """
    Clean up the test environment by removing the checkpoint.
    """

    print("Cleaning up the test environment...")

    if tools.checkpoint_exists(checkpoint_name):
        print(f"Removing checkpoint '{checkpoint_name}'...")
        cli.checkpoint_delete(checkpoint_name)

    print("Checkpoint removed successfully!")


def prepare():
    """
    Prepare the test environment by uploading a checkpoint to resume from.
    """

    print("Preparing the test environment...")

    training_name = tools.gen_training_name("gpt2-1x1-upload-e2e")

    tools.training_run(
        name=training_name,
        dataset="ci-gpt2-tokenized-wikitext",
        repository_url="https://github.com/flexaihq/fcs-experiments-private.git",
        repository_revision=os.getenv("TRAINING_REVISION", "main"),
        requirements_path="code/causal-language-modeling/requirements.txt",
        entry_point="code/causal-language-modeling/train.py",
        runtime="nvidia-25.06",
        model_args={
            "do_train": True,
            "dataset_name": " wikitext",
            "tokenized_dataset_load_dir": "/input/ci-gpt2-tokenized-wikitext",
            "model_name_or_path": "openai-community/gpt2",
            "output_dir": "/output-checkpoint",
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "max_steps": 5,
        },
    )

    # Output is 2 checkpoints:
    # - checkpoint-5, during the training. It contains data we can resume from, notably optimizer.pt
    # - <root>, which contains the final model, but we cannot resume from it.

    checkpoint_id = None

    for item in cli.training_list_checkpoints(name=training_name):
        metadata = cli.checkpoint_inspect(item["id"])

        for file in metadata["status"]["files"]:
            if file["path"] == "optimizer.pt":
                checkpoint_id = item["id"]
                break

        if checkpoint_id is not None:
            break

    if checkpoint_id is None:
        raise RuntimeError("No checkpoint with optimizer.pt found in the training run.")

    print(f"Preparation done successfully: checkpoint id: {checkpoint_id}!")
    return checkpoint_id


if __name__ == "__main__":
    main()
