# Rag application

## Start the FCS endpoints

Create the FCS secret that contains your HF token in order to acces the inference models:

```bash
# Enter your HF token value when prompted
flexai secret create hf-token
```

Start the FCS endpoint of the LLM:

```bash
LLM_INFERENCE_NAME=qwen-llm
export LLM_MODEL_NAME=Qwen/Qwen2.5-32B-Instruct
flexai inference serve $LLM_INFERENCE_NAME --hf-token-secret hf-token -- --model=$LLM_MODEL_NAME --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 16384
# store the returned information
export LLM_API_KEY=<store the given API key>
export LLM_URL=$(flexai inference inspect $LLM_INFERENCE_NAME -j | jq .config.endpointUrl -r)
```

Start the FCS endpoint of the embedder:

```bash
EMBED_INFERENCE_NAME=e5-embed
export EMBEDDINGS_MODEL_NAME=intfloat/multilingual-e5-large
flexai inference serve $EMBED_INFERENCE_NAME --hf-token-secret hf-token -- --model=$EMBEDDINGS_MODEL_NAME --task=embed --trust-remote-code --dtype=float32
# store the returned information
export EMBEDDINGS_API_KEY=<store the given API key>
export EMBEDDINGS_URL=$(flexai inference inspect $EMBED_INFERENCE_NAME -j | jq .config.endpointUrl -r)
```

##  Run the RAG application

##  LangSmith

```bash
export LANGSMITH_TRACING="true"
export LANGSMITH_API_KEY="..."
```
