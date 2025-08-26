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

    nodes = 2
    accelerators = 8 if not args.lite else 1

    training_name = tools.gen_training_name(nodes=nodes, accelerators=accelerators)

    tools.training_run(
        name=training_name,
        nodes=nodes,
        accelerators=accelerators,
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
            f"Total train batch size (w. parallel, distributed & accumulation) = {32 if not args.lite else 4}",
            # Verifies that the training was run with DeepSpeed ZeRO3
            "DeepSpeed ZeRO3 detected",
            "DeepSpeed Final Optimizer = DeepSpeedZeroOptimizer_Stage3",
        )

        tools.assert_checkpoints(
            training_name,
            [
                # zero_to_fp32.py indicates DeepSpeed checkpoint
                tools.ExpectedCheckpoint(
                    name="checkpoint-50",
                    node="0",
                    files=["model.safetensors", "zero_to_fp32.py"],
                ),
                tools.ExpectedCheckpoint(
                    name="checkpoint-99",
                    node="0",
                    files=["model.safetensors", "zero_to_fp32.py"],
                ),
                tools.ExpectedCheckpoint(name="global_step50", node="0"),
                tools.ExpectedCheckpoint(name="global_step99", node="0"),
                tools.ExpectedCheckpoint(
                    name="", node="0", files=["model.safetensors"]
                ),
                # data only on node 0
                tools.ExpectedCheckpoint(name="checkpoint-50", node="1"),
                tools.ExpectedCheckpoint(name="checkpoint-99", node="1"),
                tools.ExpectedCheckpoint(name="global_step50", node="1"),
                tools.ExpectedCheckpoint(name="global_step99", node="1"),
                tools.ExpectedCheckpoint(name="", node="1"),
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
