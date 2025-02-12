#!/bin/bash
set -e

TRAINING_NAME=gpt2-flash-attention-1x1-e2e-$TRAINING_SUFFIX
flexai training run $TRAINING_NAME -D ci-gpt2-tokenized-wikitext -s fcs-experiments-private -r $TRAINING_REVISION -- code/causal-language-modeling/train.py \
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
    --attn_implementation flash_attention_2 \
    --torch_dtype float16 \
    --eval_strategy steps

./ci/wait_for_training.sh $TRAINING_NAME
timeout 180 flexai training logs $TRAINING_NAME > logs.txt || echo "gettings logs.."
echo "Checking log content..."
grep "Loading tokenized dataset from:" logs.txt
grep "(attn): GPT2FlashAttention2" logs.txt # check if the model is using the correct attention layer
grep "\*\*\*\*\* eval metrics \*\*\*\*\*" logs.txt
grep "Total train batch size (w. parallel, distributed & accumulation) = 8" logs.txt
echo "Checking fetch content..."
flexai training fetch $TRAINING_NAME
unzip -l output_0.zip | grep output/checkpoint-50/model.safetensors
unzip -l output_0.zip | grep output/checkpoint-99/model.safetensors
unzip -l output_0.zip | grep output/model.safetensors
