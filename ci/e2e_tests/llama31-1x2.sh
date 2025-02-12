#!/bin/bash
set -e

TRAINING_NAME=llama3-1-1x2-e2e-$TRAINING_SUFFIX
flexai training run $TRAINING_NAME -a 2 -n 1 -D ci-llama-tokenized-oag -s fcs-experiments-private -r $TRAINING_REVISION -S HF_TOKEN=GC_HF_TOKEN -S WANDB_API_KEY=GC_WANDB_API_KEY -E WANDB_PROJECT=ci-e2e -- code/causal-language-modeling-qlora/train.py \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --dataset_name timdettmers/openassistant-guanaco \
    --tokenized_dataset_load_dir /input \
    --dataset_text_field text \
    --load_in_4bit \
    --use_peft \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --output_dir /output \
    --max_steps 99 \
    --log_level info

./ci/wait_for_training.sh $TRAINING_NAME
timeout 180 flexai training logs $TRAINING_NAME > logs.txt || echo "gettings logs.."
echo "Checking log content..."
grep "open file: /input/dataset_dict.json" logs.txt
grep "Training completed." logs.txt
grep "Total train batch size (w. parallel, distributed & accumulation) = 16" logs.txt # 1*2*2*4
echo "Checking fetch content..."
flexai training fetch $TRAINING_NAME
unzip -l output_0.zip | grep output/checkpoint-99/adapter_model.safetensors
unzip -l output_0.zip | grep output/adapter_model.safetensors
