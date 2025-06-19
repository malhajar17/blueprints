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

    accelerators = 2 if not args.lite else 1

    training_name = tools.gen_training_name(accelerators=accelerators)

    model_args = {
        "model_name_or_path": "parler-tts/parler_tts_mini_v0.1",
        "save_to_disk": "/input/ci-text-to-speech-fr",
        "temporary_save_to_disk": "./audio_code_tmp/",
        "wandb_project": "ci-e2e",
        "feature_extractor_name": "ylacombe/dac_44khZ_8kbps",
        "description_tokenizer_name": "google/flan-t5-large",
        "prompt_tokenizer_name": "google/flan-t5-large",
        "report_to": "wandb",
        "overwrite_output_dir": True,
        "output_dir": "/output-checkpoint",
        "train_dataset_name": "PHBJT/cml-tts-20percent-subset",
        "train_metadata_dataset_name": "PHBJT/cml-tts-20percent-subset-description",
        "train_dataset_config_name": "default",
        "train_split_name": "train",
        "eval_dataset_name": "PHBJT/cml-tts-20percent-subset",
        "eval_metadata_dataset_name": "PHBJT/cml-tts-20percent-subset-description",
        "eval_dataset_config_name": "default",
        "eval_split_name": "test",
        "target_audio_column_name": "audio",
        "description_column_name": "text_description",
        "prompt_column_name": "text",
        "max_eval_samples": 2,
        "max_duration_in_seconds": 30,
        "min_duration_in_seconds": 2.0,
        "max_text_length": 600,
        "group_by_length": True,
        "add_audio_samples_to_wandb": True,
        "preprocessing_num_workers": 1,
        "do_train": True,
        "max_steps": 99,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": "false",
        "per_device_train_batch_size": 6,
        "learning_rate": 0.00095,
        "adam_beta1": 0.9,
        "adam_beta2": 0.99,
        "weight_decay": 0.01,
        "lr_scheduler_type": "constant_with_warmup",
        "warmup_steps": 5,
        "logging_steps": 10,
        "freeze_text_encoder": True,
        "do_eval": True,
        "predict_with_generate": True,
        "include_inputs_for_metrics": True,
        "evaluation_strategy": "steps",
        "eval_steps": 50,
        "save_steps": 50,
        "save_total_limit": 2,
        "per_device_eval_batch_size": 6,
        "audio_encoder_per_device_batch_size": 1,
        "dtype": "bfloat16",
        "seed": 456,
        "dataloader_num_workers": 8,
        "attn_implementation": "sdpa",
    }

    if args.lite:
        model_args["max_steps"] = 20
        model_args["per_device_train_batch_size"] = 1
        model_args["eval_steps"] = 15
        model_args["save_steps"] = 15
        model_args["per_device_eval_batch_size"] = 1
        model_args["dataloader_num_workers"] = 2

    tools.training_run(
        name=training_name,
        dataset="ci-text-to-speech-fr",
        accelerators=accelerators,
        env={
            "WANDB_PROJECT": "ci-e2e",
        },
        secrets={
            "WANDB_API_KEY": "GC_WANDB_API_KEY",
        },
        repository_url="https://github.com/flexaihq/fcs-experiments-private.git",
        repository_revision=os.getenv("TRAINING_REVISION", "main"),
        entry_point="code/text-to-speech/run_parler_tts_training.py",
        model_args=model_args,
    )

    logs = tools.TrainingLogs.fetch(training_name)
    try:
        if not args.lite:
            batch_size = 48  # 1x2x4x6
        else:
            batch_size = 4  # 1x1x4x1

        logs.assert_contains(
            "Loading dataset from disk:",
            "wandb: Run summary:",
            f"Total train batch size (w. parallel & distributed) = {batch_size}",
        )

        print("Fetching checkpoints...")

        checkpoints = cli.training_list_checkpoints(name=training_name)

        expected_checkpoint_counts = 3
        assert (
            len(checkpoints) == expected_checkpoint_counts
        ), f"Expected {expected_checkpoint_counts} checkpoints, got {len(checkpoints)}"

        # We have one checkpoint-15 without config.json, and with a pytorch_model.bin
        # And one final checkpoint with config.json, and a model.safetensors
        for item in checkpoints:
            checkpoint = tools.Checkpoint(item["id"])
            if checkpoint.exists("config.json"):
                checkpoint.assert_exist("model.safetensors")
            else:
                checkpoint.assert_exist("pytorch_model.bin")

        print("Training done successfully!")

    except:
        logs.dump()
        raise


if __name__ == "__main__":
    main()
