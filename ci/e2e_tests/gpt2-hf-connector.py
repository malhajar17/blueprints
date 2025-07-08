#!/usr/bin/env python3

import os
import sys

from lib import cli, tools

dataset_name = "gpt2-hf-connector-1x1-e2e"
checkpoint_name = "gpt2-hf-connector-1x1-e2e"


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

    prepare()

    training_name = tools.gen_training_name()

    tools.training_run(
        name=training_name,
        dataset=f"{dataset_name}=wikitext",
        input_checkpoint=checkpoint_name,
        repository_url="https://github.com/flexaihq/fcs-experiments-private.git",
        repository_revision=os.getenv("TRAINING_REVISION", "main"),
        entry_point="code/causal-language-modeling/train.py",
        model_args={
            "do_eval": True,
            "do_train": True,
            "dataset_name": "/input/wikitext",
            "dataset_config_name": "wikitext-2-raw-v1",
            "dataset_group_text": "true",
            "model_name_or_path": "/input-checkpoint",
            "output_dir": "/output-checkpoint",
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "logging_steps": 10,
            "save_steps": 50,
            "eval_steps": 50,
            "max_steps": 99,
            "eval_strategy": "steps",
        },
    )

    logs = tools.TrainingLogs.fetch(training_name)
    try:
        logs.assert_contains(
            "***** eval metrics *****",
            "Total train batch size (w. parallel, distributed & accumulation) = 8",
        )

        print("Fetching checkpoints...")

        checkpoints = cli.training_list_checkpoints(name=training_name)

        assert len(checkpoints) == 3, f"Expected 3 checkpoints, got {len(checkpoints)}"

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
    Clean up the test environment by removing the dataset and checkpoint.
    """

    print("Cleaning up the test environment...")

    if tools.dataset_exists(dataset_name):
        print(f"Removing dataset {dataset_name}...")
        cli.dataset_delete(dataset_name)

    if tools.checkpoint_exists(checkpoint_name):
        print(f"Removing checkpoint '{checkpoint_name}'...")
        cli.checkpoint_delete(checkpoint_name)

    print("Cleanup completed.")


def prepare():
    """
    Prepare the test environment by uploading a dataset and checkpoint.
    """

    print("Preparing the test environment...")

    if tools.dataset_exists(dataset_name):
        print(f"Dataset {dataset_name} already exists, skipping upload.")
    else:
        print(f"Uploading dataset {dataset_name}...")
        tools.dataset_push(
            dataset_name, path="wikitext", storage_provider="CI-HF-STORAGE"
        )

    if tools.checkpoint_exists(checkpoint_name):
        print(f"Checkpoint '{checkpoint_name}' already exists. Skipping upload.")
    else:
        print(f"Uploading checkpoint '{checkpoint_name}'...")
        tools.checkpoint_push(
            checkpoint_name,
            path="openai-community/gpt2",
            storage_provider="CI-HF-STORAGE",
        )

    print("Preparation completed.")


if __name__ == "__main__":
    main()
