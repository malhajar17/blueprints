#!/bin/bash
set -e

TRAINING_NAME=gpt2-1x1-resume-e2e-$TRAINING_SUFFIX
flexai training run $TRAINING_NAME -D ci-gpt2-tokenized-wikitext -u $SOURCE --checkpoint ci-gpt2-ckpt500 -b $TRAINING_REVISION -- code/causal-language-modeling/train.py \
    --do_eval \
    --do_train \
    --dataset_name wikitext \
    --tokenized_dataset_load_dir /input \
    --model_name_or_path /checkpoint \
    --resume_from_checkpoint /checkpoint \
    --output_dir /output \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --max_steps 599 \
    --eval_strategy steps

./ci/wait_for_training.sh $TRAINING_NAME
timeout 300 flexai training logs $TRAINING_NAME > logs.txt || { echo "Error: Timeout while getting logs."; exit 1; }
echo "Checking log content..."
grep "Loading tokenized dataset from:" logs.txt
grep "Continuing training from global step 500" logs.txt
grep "\*\*\*\*\* eval metrics \*\*\*\*\*" logs.txt
grep "Total train batch size (w. parallel, distributed & accumulation) = 8" logs.txt
echo "Checking fetch content..."
flexai training fetch $TRAINING_NAME
unzip -l output_0.zip | grep output/checkpoint-599/model.safetensors
unzip -l output_0.zip | grep output/model.safetensors
