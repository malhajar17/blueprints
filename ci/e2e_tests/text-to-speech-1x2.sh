#!/bin/bash
set -e

TRAINING_NAME=text-to-speech-1x2-e2e-$TRAINING_SUFFIX
flexai training run $TRAINING_NAME -a 2 -n 1 -D ci-text-to-speech-fr -s $SOURCE -r $TRAINING_REVISION -S WANDB_API_KEY=GC_WANDB_API_KEY -E WANDB_PROJECT=ci-e2e -- code/text-to-speech/run_parler_tts_training.py \
    --model_name_or_path=parler-tts/parler_tts_mini_v0.1 \
    --save_to_disk=/input \
    --temporary_save_to_disk=./audio_code_tmp/ \
    --wandb_project=ci-e2e \
    --feature_extractor_name=ylacombe/dac_44khZ_8kbps \
    --description_tokenizer_name=google/flan-t5-large \
    --prompt_tokenizer_name=google/flan-t5-large \
    --report_to=wandb \
    --overwrite_output_dir \
    --output_dir=/output \
    --train_dataset_name=PHBJT/cml-tts-20percent-subset \
    --train_metadata_dataset_name=PHBJT/cml-tts-20percent-subset-description \
    --train_dataset_config_name=default \
    --train_split_name=train \
    --eval_dataset_name=PHBJT/cml-tts-20percent-subset \
    --eval_metadata_dataset_name=PHBJT/cml-tts-20percent-subset-description \
    --eval_dataset_config_name=default \
    --eval_split_name=test \
    --target_audio_column_name=audio \
    --description_column_name=text_description \
    --prompt_column_name=text \
    --max_eval_samples=2 \
    --max_duration_in_seconds=30 \
    --min_duration_in_seconds=2.0 \
    --max_text_length=600 \
    --group_by_length \
    --add_audio_samples_to_wandb \
    --preprocessing_num_workers=1 \
    --do_train \
    --max_steps 99 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing=false \
    --per_device_train_batch_size=6 \
    --learning_rate=0.00095 \
    --adam_beta1=0.9 \
    --adam_beta2=0.99 \
    --weight_decay=0.01 \
    --lr_scheduler_type=constant_with_warmup \
    --warmup_steps=5 \
    --logging_steps=10 \
    --freeze_text_encoder \
    --do_eval \
    --predict_with_generate \
    --include_inputs_for_metrics \
    --evaluation_strategy=steps \
    --eval_steps=50 \
    --save_steps=50 \
    --save_total_limit=1 \
    --per_device_eval_batch_size=6 \
    --audio_encoder_per_device_batch_size=1 \
    --dtype=bfloat16 \
    --seed=456 \
    --dataloader_num_workers=8 \
    --attn_implementation=sdpa

./ci/wait_for_training.sh $TRAINING_NAME
timeout 300 flexai training logs $TRAINING_NAME > logs.txt || { echo "Error: Timeout while getting logs."; exit 1; }
echo "Checking log content..."
grep "Loading dataset from disk:" logs.txt
grep "wandb: Run summary:" logs.txt
grep "Total train batch size (w. parallel & distributed) = 48" logs.txt # 1x2x4x6
echo "Checking fetch content..."
flexai training fetch $TRAINING_NAME
# unzip -l output_0.zip | grep output/checkpoint-50-epoch-0/pytorch_model.bin # intermediate checkpointing is disabled
# unzip -l output_0.zip | grep output/checkpoint-99-epoch-0/pytorch_model.bin
unzip -l output_0.zip | grep output/model.safetensors
