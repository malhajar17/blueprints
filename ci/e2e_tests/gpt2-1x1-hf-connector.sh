#!/bin/bash
set -e

flexai training run $TRAINING_NAME -D $DATASET_NAME=wikitext -C $CHECKPOINT_NAME -u $SOURCE -b $TRAINING_REVISION -- code/causal-language-modeling/train.py \
    --do_eval \
    --do_train \
    --dataset_name /input/wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --dataset_group_text true \
    --model_name_or_path /checkpoint \
    --output_dir /output \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_steps 10 \
    --save_steps 50 \
    --eval_steps 50 \
    --max_steps 99 \
    --eval_strategy steps

./ci/wait_for_training.sh $TRAINING_NAME
timeout 300 flexai training logs $TRAINING_NAME > logs.txt || { echo "Error: Timeout while getting logs."; exit 1; }
echo "Checking log content..."
grep "\*\*\*\*\* eval metrics \*\*\*\*\*" logs.txt
grep "Total train batch size (w. parallel, distributed & accumulation) = 8" logs.txt
echo "Checking fetch content..."
flexai training fetch $TRAINING_NAME
unzip -l output_0.zip | grep output/checkpoint-50/model.safetensors
unzip -l output_0.zip | grep output/checkpoint-99/model.safetensors
unzip -l output_0.zip | grep output/model.safetensors
