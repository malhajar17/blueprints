#!/usr/bin/env python3

import os
import sys

from lib import cli, tools
from lib.tools import Not


def main():
    args = tools.setup(
        lambda args: args.add_argument(
            "--lite",
            action="store_true",
            help="Run the test in lite mode, for smaller environments.",
        )
    )

    training_name = tools.gen_training_name()

    model_args = {
        "do_train": True,
        "eval_strategy": "no",
        "dataset_name": "HuggingFaceFW/fineweb",
        "dataset_config_name": "CC-MAIN-2024-10",
        "dataset_streaming": True,
        "dataset_group_text": True,
        "dataloader_num_workers": 8,
        "max_steps": 99,
        "model_name_or_path": "openai-community/gpt2",
        "output_dir": "/output-checkpoint",
        "per_device_train_batch_size": 8,
        "logging_steps": 10,
        "save_steps": 50,
    }

    if args.lite:
        model_args["dataloader_num_workers"] = 2
        model_args["max_steps"] = 20
        model_args["per_device_train_batch_size"] = 2
        model_args["logging_steps"] = 2
        model_args["save_steps"] = 15

    tools.training_run(
        name=training_name,
        repository_url="https://github.com/flexaihq/fcs-experiments-private.git",
        repository_revision=os.getenv("TRAINING_REVISION", "main"),
        requirements_path="code/causal-language-modeling/requirements.txt",
        entry_point="code/causal-language-modeling/train.py",
        model_args=model_args,
    )

    logs = tools.TrainingLogs.fetch(training_name)
    try:
        logs.assert_contains(
            "Streaming dataset",
            Not("Loading tokenized dataset from:"),
            f"Total train batch size (w. parallel, distributed & accumulation) = {8 if not args.lite else 2}",
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


if __name__ == "__main__":
    main()
