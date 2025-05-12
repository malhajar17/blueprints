#!/usr/bin/env bash
set -ex

flexai checkpoint inspect ci-gpt2-ckpt5 && { echo "Checkpoint ci-gpt2-ckpt5 already exists. Skipping upload."; exit 0; }

TRAINING_NAME=gpt2-1x1-upload-e2e-$TRAINING_SUFFIX
flexai training run "$TRAINING_NAME" -D ci-gpt2-tokenized-wikitext -u $SOURCE -b $TRAINING_REVISION -- code/causal-language-modeling/train.py \
    --do_train \
    --dataset_name wikitext \
    --tokenized_dataset_load_dir /input/ci-gpt2-tokenized-wikitext \
    --model_name_or_path openai-community/gpt2 \
    --output_dir /output \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_steps 5

./ci/wait_for_training.sh "$TRAINING_NAME"
timeout 300 flexai training logs "$TRAINING_NAME" > logs.txt || { echo "Error: Timeout while getting logs."; exit 1; }

flexai training fetch "$TRAINING_NAME"
unzip -l output_0.zip
unzip output_0.zip

# upload the checkpoint-5 folder to the training server
flexai checkpoint push ci-gpt2-ckpt5 --file output/checkpoint-5
