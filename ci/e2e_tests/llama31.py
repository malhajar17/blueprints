#!/usr/bin/env python3

import os
import sys

from lib import cli, tools


def main():
    args = tools.setup(
        lambda args: args.add_argument(
            "--lite",
            action="store_true",
            help="Run the test in lite mode, for smaller environments.",
        )
    )

    accelerators = 2 if not args.lite else 1

    training_name = tools.gen_training_name(accelerators=accelerators)

    model_args = {
        "model_name_or_path": "meta-llama/Meta-Llama-3.1-8B",
        "dataset_name": "/input/ci-llama-tokenized-oag",
        "tokenized_dataset_load_dir": "/input/ci-llama-tokenized-oag",
        "dataset_text_field": "text",
        "load_in_4bit": True,
        "use_peft": True,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "output_dir": "/output-checkpoint",
        "max_steps": 99,
        "log_level": "info",
    }

    if args.lite:
        model_args["per_device_train_batch_size"] = 1
        model_args["gradient_accumulation_steps"] = 2
        model_args["max_steps"] = 9

    tools.training_run(
        name=training_name,
        accelerators=accelerators,
        dataset="ci-llama-tokenized-oag",
        env={
            "WANDB_PROJECT": "ci-e2e",
        },
        secrets={
            "HF_TOKEN": "GC_HF_TOKEN",
            "WANDB_API_KEY": "GC_WANDB_API_KEY",
        },
        repository_url="https://github.com/flexaihq/fcs-experiments-private.git",
        repository_revision=os.getenv("TRAINING_REVISION", "main"),
        entry_point="code/causal-language-modeling-qlora/train.py",
        model_args=model_args,
    )

    logs = tools.TrainingLogs.fetch(training_name)
    try:
        if not args.lite:
            batch_size = 16  # 1*2*2*4
        else:
            batch_size = 2  # 1*1*2*1

        logs.assert_contains(
            "Training completed.",
            f"Total train batch size (w. parallel, distributed & accumulation) = {batch_size}",
        )

        print("Fetching checkpoints...")

        checkpoints = cli.training_list_checkpoints(name=training_name)

        # TODO: refactor when metadata on checkpoints is available
        found_model = False
        for item in checkpoints:
            checkpoint = tools.Checkpoint(item["id"])
            if checkpoint.exists("adapter_model.safetensors"):
                found_model = True
                break

        assert found_model, "No checkpoint with model data found."

        print("Training done successfully!")

    except:
        # On CI, stderr and stdout can be interleaved, so we dump logs to stderr
        # to keep the output ordered and clean.
        logs.dump(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
