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

    tools.training_run(
        name=training_name,
        dataset="ci-sdxl-tokenized-naruto",
        accelerators=accelerators,
        env={
            "WANDB_PROJECT": "ci-e2e",
        },
        secrets={
            "HF_TOKEN": "GC_HF_TOKEN",
            "WANDB_API_KEY": "GC_WANDB_API_KEY",
        },
        repository_url="https://github.com/flexaihq/fcs-experiments-private.git",
        repository_revision=os.getenv("TRAINING_REVISION", "main"),
        entry_point="code/diffuser/train_text_to_image_lora_sdxl.py",
        model_args={
            "pretrained_model_name_or_path": "stabilityai/stable-diffusion-xl-base-1.0",
            "pretrained_vae_model_name_or_path": "madebyollin/sdxl-vae-fp16-fix",
            "train_dataset_load_dir": "/input/ci-sdxl-tokenized-naruto",
            "caption_column": "text",
            "resolution": 1024,
            "random_flip": True,
            "train_batch_size": 1,
            "checkpointing_steps": 50,
            "learning_rate": 1e-04,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            "mixed_precision": "fp16",
            "seed": 42,
            "output_dir": "/output-checkpoint",
            "max_train_steps": 99,
            "validation_prompt": "cute dragon creature",
        },
    )

    logs = tools.TrainingLogs.fetch(training_name)
    try:
        if not args.lite:
            batch_size = 2  # 1x2x1
        else:
            batch_size = 1

        logs.assert_contains(
            "Training completed.",
            f"Total train batch size (w. parallel, distributed & accumulation) = {batch_size}",
        )

        print("Fetching checkpoints...")

        checkpoints = cli.training_list_checkpoints(name=training_name)

        assert len(checkpoints) == 2, f"Expected 2 checkpoints, got {len(checkpoints)}"

        for item in checkpoints:
            checkpoint = tools.Checkpoint(item["id"])
            checkpoint.assert_exist("pytorch_lora_weights.safetensors")

        print("Training done successfully!")

    except:
        # On CI, stderr and stdout can be interleaved, so we dump logs to stderr
        # to keep the output ordered and clean.
        logs.dump(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
