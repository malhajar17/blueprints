#!/bin/bash

set -x

MODEL_PATH=Qwen/Qwen3-VL-30B-A3B-Instruct  # replace it with your local file path

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.tensor_parallel_size=8 \
    trainer.experiment_name=qwen3_vl_30b_geo_grpo \
    trainer.n_gpus_per_node=8
