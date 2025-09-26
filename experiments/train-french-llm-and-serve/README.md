# Training French Language Models with FlexAI and LlamaFactory

This experiment demonstrates how to use FlexAI to train a high-quality French language model using LlamaFactory, achieving excellent performance on French general tasks. We'll use the `Qwen2.5-7B` model and the `openhermes-fr` dataset to create a French-specialized language model.

You will see that this process requires configuring LlamaFactory's dataset registry, setting up training parameters, and leveraging FlexAI's managed training infrastructure to create a production-ready French language model.

> **Note**: If you haven't already connected FlexAI to GitHub, run `flexai code-registry connect` to set up a code registry connection. This allows FlexAI to pull repositories directly using the repository URL in training commands.

## Step 1: Verify Dataset Configuration

First, ensure the French dataset is properly configured in LlamaFactory's dataset registry.

Navigate to `experiments/code/llama-factory/data/dataset_info.json` and verify the `openhermes-fr` dataset entry exists:

```json
{
    "openhermes-fr": {
        "hf_hub_url": "legmlai/openhermes-fr",
        "columns": {
            "prompt": "prompt",
            "response": "accepted_completion"
        }
    }
}
```

The `openhermes-fr` dataset from `legmlai/openhermes-fr` is specifically designed for French language tasks and will help achieve high intelligence scores on French general tasks.

## Step 2: Configure Training Parameters

The [`qwen25-7B_sft.yaml`](../../code/llama-factory/qwen25-7B_sft.yaml) file contains the training configuration for our French language model. Key settings include:

- **Model**: `Qwen/Qwen2.5-7B` - Excellent multilingual base model
- **Stage**: `sft` (Supervised Fine-Tuning) - Perfect for task-specific adaptation  
- **Dataset**: `openhermes-fr` - High-quality French conversation data
- **Training**: Full fine-tuning with DeepSpeed ZeRO Stage 3 for optimal performance

```yaml
---
### model
model_name_or_path: Qwen/Qwen2.5-7B
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
deepspeed: code/llama-factory/ds_z3_config.json

### dataset
dataset: openhermes-fr
dataset_dir: code/llama-factory/data
template: qwen
cutoff_len: 2048
max_samples: 100000
overwrite_cache: true
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: /output-checkpoint
logging_steps: 10
save_steps: 1000
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
learning_rate: 1.0e-5
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
resume_from_checkpoint:
```

## Create Secrets

To access the Qwen2.5-7B model and OpenHermes-FR dataset, you need a HuggingFace token.

Use the [`flexai secret create` command](https://docs.flex.ai/cli/commands/secret/) to store your _HuggingFace Token_ as a secret. Replace `<HF_AUTH_TOKEN_SECRET_NAME>` with your desired name for the secret:

```bash
flexai secret create <HF_AUTH_TOKEN_SECRET_NAME>
```

Then paste your _HuggingFace Token_ API key value.

## Training

For a 7B model, we recommend using **1 node (8 × H100 GPUs)** to ensure reasonable training time and avoid out-of-memory issues:

```bash
flexai training run french-qwen25-7b-sft \
  --accels 8 --nodes 1 \
  --repository-url https://github.com/malhajar17/experiments \
  --env FORCE_TORCHRUN=1 \
  --env HF_HUB_CACHE=/output/.cache \
  --env HF_HUB_DISABLE_XET=1 \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/llama-factory/requirements.txt \
  --runtime nvidia-25.06 \
  -- /layers/flexai_pip-install/packages/bin/llamafactory-cli train code/llama-factory/qwen25-7B_sft.yaml
```

## Monitoring Training Progress

You can check the status and lifecycle events of your Training Job by running:

```bash
flexai training inspect french-qwen25-7b-sft
```

Additionally, you can view the logs of your Training Job by running:

```bash
flexai training logs french-qwen25-7b-sft
```

## Getting Training Checkpoints

Once the Training Job completes successfully, you will be able to list all the produced checkpoints:

```bash
flexai training checkpoints french-qwen25-7b-sft
```

Look for checkpoints marked as `INFERENCE READY = true` - these are ready for serving.

## Serving the Trained Model

Deploy your trained model directly from the checkpoint using FlexAI inference. Replace `<CHECKPOINT_ID>` with the ID from an inference-ready checkpoint:

```bash
flexai -v inference serve french-qwen25-endpoint --checkpoint <CHECKPOINT_ID>
```

You can monitor your inference endpoint status:

```bash
# List all inference endpoints
flexai inference list

# Get detailed endpoint information  
flexai inference inspect french-qwen25-endpoint

# Check endpoint logs
flexai inference logs french-qwen25-endpoint
```

## Testing Your French Language Model

Once the endpoint is running, you can test it with French prompts. The model should demonstrate strong French language understanding, proper grammar and syntax, and cultural context awareness.

Example API call:

```bash
curl -X POST "https://your-endpoint-url/v1/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "prompt": "Expliquez-moi les avantages de l'\''intelligence artificielle en français:",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

## Expected Results

After training on the `openhermes-fr` dataset, your model should achieve:

- **Strong French Language Understanding**: Natural conversation flow, proper grammar, cultural context
- **High Performance on French Tasks**: Question answering, text summarization, creative writing
- **Maintained General Capabilities**: Reasoning, code generation, mathematical problem solving

## Technical Details

### Training Configuration Breakdown:
- **DeepSpeed ZeRO Stage 3**: Enables training of 7B model on 1 node efficiently
- **Mixed Precision (bf16)**: Accelerates training while maintaining numerical stability
- **Gradient Accumulation**: Effective batch size of 4 (2 steps × 2 per device)
- **Learning Rate Schedule**: Cosine decay with 10% warmup for stable convergence
- **Context Length**: 2048 tokens, optimized for conversation tasks

### Resource Requirements

**Recommended Configuration for Qwen2.5-7B:**
- **Nodes**: 1 node (cost-effective for 7B models)
- **Accelerators**: 8 × H100 GPUs per node  
- **Memory**: ~200GB+ GPU memory total
- **Training Time**: ~2-4 hours for 3 epochs
- **Storage**: ~30GB for checkpoints

**Command Line Parameters Explained:**
- `FORCE_TORCHRUN=1`: Ensures proper distributed training setup
- `HF_HUB_CACHE=/output/.cache`: Optimizes model download caching
- `HF_HUB_DISABLE_XET=1`: Disables XET protocol for better compatibility

### Scaling Options:
- For faster training: Increase to 2 nodes (16 × H100)
- For larger datasets: Adjust `max_samples` parameter
- For longer context: Increase `cutoff_len` (requires more memory)
- For memory efficiency: Switch to `finetuning_type: lora` for QLoRA training

## Troubleshooting

**Common Issues:**

**Training Job Fails to Start:**
```bash
# Check FlexAI authentication
flexai auth status

# Verify repository access  
git clone https://github.com/malhajar17/experiments
```

**Out of Memory Errors:**
- Reduce `per_device_train_batch_size` from 1 to lower value
- Increase `gradient_accumulation_steps` to maintain effective batch size
- Consider using `finetuning_type: lora` for memory efficiency

**Checkpoint Not Inference Ready:**
- Wait for training to complete fully (check with `flexai training inspect`)
- Ensure `save_only_model: false` in YAML configuration
- Verify training completed successfully without errors

**Endpoint Deployment Issues:**
- Verify checkpoint shows `INFERENCE READY = true` status
- Check FlexAI cluster availability with `flexai inference list`
- Review detailed logs with `flexai inference logs <endpoint-name>`

---
