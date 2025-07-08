#!/usr/bin/env python3

import os
import sys

from lib import cli, tools


def main():
    print("Flash-Attention is disabled at the moment, see AIS-214")
    return

    tools.setup()

    training_name = tools.gen_training_name()

    tools.training_run(
        name=training_name,
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
            "attn_implementation": "flash_attention_2",
            "torch_dtype": "float16",
            "eval_strategy": "steps",
        },
    )

    logs = tools.TrainingLogs.fetch(training_name)
    try:
        logs.assert_contains(
            "Loading tokenized dataset from:",
            "(attn): GPT2FlashAttention2***** eval metrics *****",
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


if __name__ == "__main__":
    main()
