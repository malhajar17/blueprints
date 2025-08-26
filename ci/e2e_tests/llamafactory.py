#!/usr/bin/env python3

import os
import sys

from lib import cli, tools


def main():
    tools.setup()

    training_name = tools.gen_training_name()

    tools.training_run(
        name=training_name,
        env={
            "FORCE_TORCHRUN": "1",
        },
        secrets={
            "HF_TOKEN": "GC_HF_TOKEN",
        },
        repository_url="https://github.com/flexaihq/fcs-experiments-private.git",
        repository_revision=os.getenv("TRAINING_REVISION", "main"),
        requirements_path="code/llama-factory/requirements.txt",
        entry_point="/layers/flexai_pip-install/packages/bin/llamafactory-cli",
        runtime="nvidia-25.06",
        model_args={
            "train": cli.RawArg(),
            "ci/e2e_tests/configs/llama3_sft_e2e.yaml": cli.RawArg(),
        },
    )

    logs = tools.TrainingLogs.fetch(training_name)
    try:
        logs.assert_contains(
            "Training completed.",
            "***** eval metrics *****",
            "Total train batch size (w. parallel, distributed & accumulation) = 2",
        )

        # Verifies that the training was run with DeepSpeed ZeRO3
        logs.assert_contains(
            "DeepSpeed ZeRO3 detected",
            "DeepSpeed Final Optimizer = DeepSpeedZeroOptimizer_Stage3",
        )

        tools.assert_checkpoints(
            training_name,
            [
                tools.ExpectedCheckpoint(
                    name="checkpoint-50",
                    # zero_to_fp32.py means that Deepspeed is in the checkpoint
                    files=["model.safetensors", "zero_to_fp32.py"],
                ),
                tools.ExpectedCheckpoint(
                    name="checkpoint-99",
                    files=["model.safetensors", "zero_to_fp32.py"],
                ),
                tools.ExpectedCheckpoint(name="global_step50"),
                tools.ExpectedCheckpoint(name="global_step98"),
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
