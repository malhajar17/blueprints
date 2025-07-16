#!/usr/bin/env python3

import os
import sys

from lib import cli, tools


def main():
    tools.setup()

    training_name = tools.gen_training_name()

    tools.training_run(
        name=training_name,
        dataset="ci-gpt2-tokenized-wikitext",  # This dataset is not used in this test, but required for now to enable companion (bug to be fixed in infra)
        secrets={
            "HF_TOKEN": "GC_HF_TOKEN",
        },
        repository_url="https://github.com/flexaihq/fcs-experiments-private.git",
        repository_revision=os.getenv("TRAINING_REVISION", "main"),
        requirements_path="code/llama-factory/requirements.txt",
        # entry_point="llamafactory-cli", TODO: replace with this
        entry_point="/layers/flexai_pip-install/packages/bin/llamafactory-cli",
        model_args={
            "train": cli.RawArg(),
            "ci/e2e_tests/configs/llama3_sft_e2e.yaml": cli.RawArg(),
        },
    )

    # TODO: look for the correct information in the logs
    # ====>
    logs = tools.TrainingLogs.fetch(training_name)
    try:
        logs.assert_contains(
            "Training completed.",
            "***** eval metrics *****",
            "Total train batch size (w. parallel, distributed & accumulation) = 2",
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
