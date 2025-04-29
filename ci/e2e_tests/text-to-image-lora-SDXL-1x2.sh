#!/bin/bash
set -ex

TRAINING_NAME=text-to-image-lora-SDXL-1x2-e2e-$TRAINING_SUFFIX
flexai training run $TRAINING_NAME -a 2 -n 1 -D ci-sdxl-tokenized-naruto -u $SOURCE -b $TRAINING_REVISION -S HF_TOKEN=GC_HF_TOKEN -S WANDB_API_KEY=GC_WANDB_API_KEY -E WANDB_PROJECT=ci-e2e -- code/diffuser/train_text_to_image_lora_sdxl.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --pretrained_vae_model_name_or_path madebyollin/sdxl-vae-fp16-fix \
    --train_dataset_load_dir /input/ci-sdxl-tokenized-naruto \
    --caption_column text \
    --resolution 1024 \
    --random_flip \
    --train_batch_size 1 \
    --checkpointing_steps 50 \
    --learning_rate 1e-04 \
    --lr_scheduler constant \
    --lr_warmup_steps 0 \
    --mixed_precision fp16 \
    --seed 42 \
    --output_dir /output \
    --max_train_steps 99 \
    --validation_prompt dragon
# TODO: replace the above line with the following line after
# the following issue is resolved:
# https://flexaihq.atlassian.net/browse/PAAS-1033
#     --validation_prompt "'cute dragon creature'"

./ci/wait_for_training.sh $TRAINING_NAME
timeout 300 flexai training logs $TRAINING_NAME > logs.txt || { echo "Error: Timeout while getting logs."; exit 1; }
echo "Checking log content..."
grep "Training completed." logs.txt
grep "Total train batch size (w. parallel, distributed & accumulation) = 2" logs.txt # 1x2x1
echo "Checking fetch content..."
flexai training fetch $TRAINING_NAME
unzip -l output_0.zip | grep output/checkpoint-50/pytorch_lora_weights.safetensors
unzip -l output_0.zip | grep output/pytorch_lora_weights.safetensors
