# Fine-Tuning a Language Model with LlamaFactory on FCS

This experiment demonstrates how to fine-tune a language model using [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) on **FlexAI**. We'll use the `Llama-3-1B` model and the `identity` and `alpaca-en-demo` [LlamaFactory datasets](https://github.com/hiyouga/LLaMA-Factory/tree/main/data) as an example, but you can adapt this guide for other models and datasets.

As you'll see below, you only need to pass your LlamaFactory configuration YAML.

> **Note**: If you haven't already connected FlexAI to GitHub, run `flexai code-registry connect` to set up a code registry connection. This allows FCS to pull repositories directly using the `-u` flag in training commands.

## Create Secrets

To be authenticated into your HuggingFace account within your code, you will use your _HuggingFace Token_.

Use the [`flexai secret create` command](https://docs.flex.ai/commands/secret) to store your _HuggingFace Token_ as a secret. Replace `<HF_AUTH_TOKEN_SECRET_NAME>` with your desired name for the secret:

```bash
flexai secret create <HF_AUTH_TOKEN_SECRET_NAME>
```

Then paste your _HuggingFace Token_ API key value.

## Step 1: Train the Model

> **Note**: The LlamaFactory example is only available on the `llama-factory` branch for now, which explains the `--repository-revision llama-factory` argument below.

The [`llama3_sft.yaml`](../../code/llama-factory/llama3_sft.yaml) file has been adapted from [this example](https://github.com/hiyouga/LLaMA-Factory/blob/0b188ca00c9de9efee63807e72e444ea74195da5/examples/train_full/llama3_full_sft.yaml#L1).

To launch a training job:

```bash
flexai training run llamafactory-sft-llama3 \
  --repository-url https://github.com/flexaihq/flexai-experiments \
  --repository-revision llama-factory \
  --secret HF_TOKEN=<HF_AUTH_TOKEN_SECRET_NAME> \
  --requirements-path code/llama-factory/requirements.txt \
  -- llamafactory-cli train code/llama-factory/llama3_sft.yaml
```

---

## Step 2: Bring Your Own Dataset

You can check our other examples, e.g. [`experiments/running-a-simple-training-job/README.md`](../running-a-simple-training-job/README.md), to see how to bring your own dataset using:

```bash
flexai dataset push my-dataset ...
```

Then, follow the [LlamaFactory dataset instructions](https://github.com/hiyouga/LLaMA-Factory/tree/main/data#readme) to prepare your data for training.

---
