#!/bin/bash
set -e

TRAINING_NAME=gpt2-2x8-e2e-$TRAINING_SUFFIX
flexai training run $TRAINING_NAME -a 8 -n 2 -D ci-gpt2-tokenized-wikitext -s fcs-experiments-private -r $TRAINING_REVISION -- code/causal-language-modeling/train.py \
    --do_eval \
    --do_train \
    --dataset_name wikitext \
    --tokenized_dataset_load_dir /input \
    --model_name_or_path openai-community/gpt2 \
    --output_dir /output \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_steps 10 \
    --save_steps 50 \
    --eval_steps 50 \
    --max_steps 99 \
    --eval_strategy steps

./ci/wait_for_training.sh $TRAINING_NAME
timeout 180 flexai training logs $TRAINING_NAME > logs.txt || echo "gettings logs.."
echo "Checking log content..."
grep "Loading tokenized dataset from:" logs.txt
grep "\*\*\*\*\* eval metrics \*\*\*\*\*" logs.txt
grep "Total train batch size (w. parallel, distributed & accumulation) = 128" logs.txt # 2*8*8
echo "Checking fetch content..."
flexai training fetch $TRAINING_NAME
unzip -l output_0.zip | grep output/checkpoint-50/model.safetensors
unzip -l output_0.zip | grep output/checkpoint-99/model.safetensors
unzip -l output_0.zip | grep output/model.safetensors
if [ $(stat -c %s output_1.zip) -gt 2097152 ]; then
    echo "Error: File is greater than 2MB"
    exit 1
fi
