# Running your first Training Job on FlexAI

This experiment demonstrates how easy it is to leverage **FlexAI** to run a Training Job with a couple of commands. We will use a simple example of training a causal language model (LLM) on the `wikitext` dataset using the `GPT-2` model.

You will see that this straightforward process only requires two components: a training script and a dataset. The training script is responsible for defining the model, setting up and applying hyperparameters, running the training loop, and applying its respective evaluation logic, while the dataset contains the information that will be used to train the model.

> **Note**: If you haven't already connected FlexAI to GitHub, you'll need to run `flexai code-registry connect` to set up a code registry connection. This allows FlexAI to pull repositories directly using the `-u` flag in training commands.

## Step 1: Preparing the Dataset

In this experiment, we will use a pre-processed version of the the `wikitext` dataset that has been set up for the `GPT-2` model.

> If you'd like to reproduce the pre-processing steps yourself to use a different dataset or simply to learn more about the process, you can refer to the [Manual Dataset Pre-processing](#manual-dataset-pre-processing) section below.

1. Download the dataset:

    ```bash
    DATASET_NAME=gpt2-tokenized-wikitext && curl -L -o ${DATASET_NAME}.zip "https://bucket-docs-samples-99b3a05.s3.eu-west-1.amazonaws.com/${DATASET_NAME}.zip" && unzip ${DATASET_NAME}.zip && rm ${DATASET_NAME}.zip
    ```

1. Upload the dataset (located in `gpt2-tokenized-wikitext/`) to FlexAI:

    ```bash
    flexai dataset push gpt2-tokenized-wikitext --file gpt2-tokenized-wikitext
    ```

## Step 2: Train the Model

Now, it's time to train your LLM on the dataset you just _pushed_ in the previous step, `gpt2-tokenized-wikitext`. This experiment uses the `GPT-2` model, however, the training script script we will use ([`code/causal-language-modeling/train.py`](../../code/causal-language-modeling/train.py)) leverages the [HuggingFace Transformers `Trainer` class](https://huggingface.co/docs/transformers/en/trainer), which makes it easy to replace `GPT-2` with another model from the [HuggingFace Model Hub](https://huggingface.co/models).

To start the Training Job, run the following command:

```bash
flexai training run first-training-job --repository-url https://github.com/flexaihq/experiments --dataset gpt2-tokenized-wikitext \
 --requirements-path code/causal-language-modeling/requirements.txt \
 -- code/causal-language-modeling/train.py \
    --do_eval \
    --do_train \
    --dataset_name wikitext \
    --tokenized_dataset_load_dir /input/gpt2-tokenized-wikitext \
    --model_name_or_path openai-community/gpt2 \
    --output_dir /output-checkpoint \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --logging_steps 50 \
    --save_steps 500 \
    --eval_steps 500 \
    --eval_strategy steps
```

The first line defines the 3 main components required to run a Training Job in FlexAI:

1. The Training Job's name (`first-training-job`).
1. The URL of the repository containing the training script (`https://github.com/flexaihq/experiments`).
1. The name of the dataset to be used (`gpt2-tokenized-wikitext`).

The second line defines the script that will be executed when the Training Job is started (`code/causal-language-modeling/train.py`).

After the second line come the script's arguments, which are passed to the script when it is executed to adjust the Training Job hyperparameters or customize its behavior. For instance, `--max_train_samples` and `--max_eval_samples` can be used to tweak the sample size.

## Step 3: Checking up on the Training Job

You can check the status and life cycle events of your Training Job by running:

```bash
flexai training inspect first-training-job
```

Additionally, you can view the logs of your Training Job by running:

```bash
flexai training logs first-training-job
```

## Step 4: Fetching the Trained Model artifacts

Once the Training Job completes successfully, you will be able to list all the produced checkpoints:

```bash
flexai training checkpoints first-training-job
```

They can be downloaded with:

```bash
flexai checkpoint fetch "<CPKT-ID>"
```

You now have a trained model that you can use for inference or further fine-tuning! Check out the [Extra](#optional-extra-steps) section below for more information on how to run your fine-tuned model locally, or even better, how to run the training script directly on FlexAI using an Interactive Training Session. You can also learn how to manually pre-process the dataset if you're interested in understanding the process better.

You can also have a look at other FlexAI experiments within this repository to explore more advanced use cases and techniques.

## Optional Extra Steps

### Try your fine-tuned model locally

You can run your newly fine-tuned model in an [FlexAI Interactive Session](#run-the-training-script-directly-on-flexai-using-an-interactive-training-session) or in a local env (e.g. `pipenv install --python 3.11`), if you have hardware that's capable of doing inference.

#### 1. Clone this repository

If you haven't already, clone this repository on your host machine:

```bash
git clone https://github.com/flexaihq/experiments.git flexai-experiments --depth 1 --branch main && cd flexai-experiments
```

#### 2. Install the dependencies

Depending on your environment, you might need to install - if not already - the experiments' dependencies by running:

```bash
pip install -r code/causal-language-modeling/requirements.txt
```

#### 3. Extract the model artifacts

First, list the available checkpoints from your training job:

```bash
flexai training checkpoints first-training-job
```

Then fetch the specific checkpoint you want to use (replace `<CHECKPOINT-ID>` with the actual checkpoint ID from the list):

```bash
flexai checkpoint fetch "<CHECKPOINT-ID>" --destination ./checkpoint
```

This will download the checkpoint to a local `checkpoint` directory. Make note of this location, as you will use it next.

#### 4. Run the inference script

Run the script made for inference on this model by running the command below, replacing `**PATH_TO_THE_CHECKPOINT_DIRECTORY**` with the path to the checkpoint directory you downloaded:

```bash
python code/causal-language-modeling/predict.py \
    --model_name_or_path **PATH_TO_THE_CHECKPOINT_DIRECTORY** \
    --input_str "Once upon a time, " \
    --max_new_tokens 30
```

### Run the training script directly on FlexAI using an Interactive Training Session

An Interactive Training Session allows you to connect to a Training Environment runtime on FlexAI and run your both training and prediction or inference scripts directly from this environment. This is a great way to test your scripts and experiment with different hyperparameters without having to create multiple Training Jobs per configuration change.

You will find the guide on how to run an Interactive Training Session in the [FlexAI Documentation](https://docs.flex.ai/cli/guides/interactive-training/). You'll need to use the path for the `flexaihq/experiments` repository as your `--repository-url` and pass the `gpt2-tokenized-wikitext` dataset you pushed earlier as `--dataset`, unless you want to leverage the Interactive Training Session's compute resources to [manually pre-process the dataset](#manual-dataset-pre-processing).

### Manual Dataset Pre-processing

To prepare and save the `wikitext` dataset for the `GPT-2` model run the following command:

```bash
python code/dataset/prepare_save_dataset.py \
    --dataset_name wikitext \
    --tokenized_dataset_save_dir gpt2-tokenized-wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --tokenizer_model_name openai-community/gpt2 \
    --dataset_group_text true
```

The generated dataset will be created in the directory set as the value of `--tokenized_dataset_save_dir`, in this case: `gpt2-tokenized-wikitext`.
Keep in mind that you can use other combinations of datasets and models available on HuggingFace.
