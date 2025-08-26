#!/usr/bin/env python3

import os
import sys

from lib import tools


def main():
    tools.setup()

    training_name = tools.gen_training_name()

    tools.training_run(
        name=training_name,
        dataset="ci-gpt2-tokenized-wikitext",
        repository_url="https://github.com/flexaihq/fcs-experiments-private.git",
        repository_revision=os.getenv("TRAINING_REVISION", "main"),
        requirements_path="code/causal-language-modeling/requirements.txt",
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
            "Total train batch size (w. parallel, distributed & accumulation) = 8",
        )

        tools.assert_checkpoints(
            training_name,
            [
                tools.ExpectedCheckpoint(
                    name="checkpoint-50", files=["model.safetensors"]
                ),
                tools.ExpectedCheckpoint(
                    name="checkpoint-99", files=["model.safetensors"]
                ),
                tools.ExpectedCheckpoint(name="", files=["model.safetensors"]),
            ],
        )

        print("Training done successfully!")

    except:
        # On CI, stderr and stdout can be interleaved, so we dump logs to stderr
        # to keep the output ordered and clean.
        logs.dump(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
