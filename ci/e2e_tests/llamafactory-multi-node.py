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
        dataset="ci-gpt2-tokenized-wikitext",  # This dataset is not used in this test, but required for now to enable companion (bug to be fixed in infra)
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
        )

        # Verifies that the training was run with DeepSpeed ZeRO3
        logs.assert_contains(
            "DeepSpeed ZeRO3 detected",
            "DeepSpeed Final Optimizer = DeepSpeedZeroOptimizer_Stage3",
        )

        print("Fetching checkpoints...")

        checkpoints = cli.training_list_checkpoints(name=training_name)
        checkpoints_len = len(checkpoints)
        # TODO(@runtime): perfect checkpoint detection for DeepSpeed
        expected_checkpoints = 10

        assert (
            checkpoints_len == expected_checkpoints
        ), f"Expected {expected_checkpoints} checkpoints, got {checkpoints_len}"

        is_deepspeed_in_ckpt = False

        for item in checkpoints:
            checkpoint = tools.Checkpoint(item["id"])

            if not item["name"].startswith("global_"):
                # Find all checkpoints with this name
                same_name_checkpoints = [
                    c for c in checkpoints if c["name"] == item["name"]
                ]
                # Only one of them should have model.safetensors
                found = 0
                for c in same_name_checkpoints:
                    chk = tools.Checkpoint(c["id"])
                    if chk.exists("model.safetensors"):
                        found += 1
                assert (
                    found == 1
                ), f"Expected exactly one checkpoint with model.safetensors for name '{item['name']}', found {found}"

            if not is_deepspeed_in_ckpt:
                is_deepspeed_in_ckpt = checkpoint.exists("zero_to_fp32.py")

        assert is_deepspeed_in_ckpt, "Expected a checkpoint with DeepSpeed Zero3"

        print("Training done successfully!")

    except:
        # On CI, stderr and stdout can be interleaved, so we dump logs to stderr
        # to keep the output ordered and clean.
        logs.dump(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
