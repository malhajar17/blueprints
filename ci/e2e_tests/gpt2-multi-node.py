#!/usr/bin/env python3

import os

from lib import cli, tools


def main():
    args = tools.setup(
        lambda args: args.add_argument(
            "--lite",
            action="store_true",
            help="Run the test in lite mode, for smaller environments.",
        )
    )

    nodes = 2
    accelerators = 8 if not args.lite else 1

    training_name = tools.gen_training_name(nodes=nodes, accelerators=accelerators)

    tools.training_run(
        name=training_name,
        nodes=nodes,
        accelerators=accelerators,
        dataset="ci-gpt2-tokenized-wikitext",
        repository_url="https://github.com/flexaihq/fcs-experiments-private.git",
        repository_revision=os.getenv("TRAINING_REVISION", "main"),
        entry_point="code/causal-language-modeling/train.py",
        model_args={
            "do_eval": True,
            "do_train": True,
            "dataset_name": "wikitext",
            "tokenized_dataset_load_dir": "/input/ci-gpt2-tokenized-wikitext",
            "model_name_or_path": "openai-community/gpt2",
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
            "Loading tokenized dataset from:",
            "***** eval metrics *****",
            f"Total train batch size (w. parallel, distributed & accumulation) = {128 if not args.lite else 16}",
        )

        print("Fetching checkpoints...")

        checkpoints = cli.training_list_checkpoints(name=training_name)

        # TODO: refactor when metadata on checkpoints is available
        found_model = False
        for item in checkpoints:
            checkpoint = tools.Checkpoint(item["id"])
            if checkpoint.exists("model.safetensors"):
                found_model = True
                break

        assert found_model, "No checkpoint with model data found."

        print("Training done successfully!")

    except:
        logs.dump()
        raise


if __name__ == "__main__":
    main()
