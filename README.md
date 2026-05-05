# flyte2-on-dgx-spark
A repo for getting started with Flyte 2 on DGX Spark

This repo contains:

- **Runnable workflows** in `hello.py`, `gpu_simple.py`, `gpu_test.py`, and `fine-tuning/`
- **Runnable app serving** examples in `app-serving/` (Gemma 4 via vLLM + a Gradio UI)

## Installing Flyte 2

On the DGX Spark:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
uv pip install flyte
```

## Starting the Flyte devbox (GPU)

On the DGX Spark:

```bash
flyte start devbox --gpu
```

The Flyte UI runs at `http://localhost:30080`.

Create a config that targets the devbox:

```bash
flyte create config \
  --endpoint localhost:30080 \
  --project flytesnacks \
  --domain development \
  --builder local \
  --insecure
```

## Running workflows (tasks) on the devbox

From this repo root (with `.flyte/config.yaml` present):

```bash
uv run python getting-started/hello.py
uv run python getting-started/gpu_simple.py
uv run python getting-started/gpu_test.py
```

For the fine-tuning example:

```bash
uv run python fine-tuning/finetune_lora.py
```

Each script prints a **run URL** you can open in the Flyte UI.

### Notes

- **Devbox must be started with `--gpu`** or pods won’t see the GPU.
- **Use `registry="localhost:30000"`** for images (devbox-local registry), matching the working examples in this repo.

## Running apps (serving) on the devbox

The `app-serving/` directory is adapted from the Gemma4 chat workshop (`unionai/workshops/tutorials/gemma4-chat`) and is tested on DGX Spark-style GPU devbox setups.

### Set up local deps for app serving

```bash
uv pip install -r app-serving/requirements.txt
```

### Add your Hugging Face token (Gemma is gated)

```bash
flyte create secret HF_TOKEN
# paste your hf_xxx token
```

### Deploy the vLLM server, then the chat UI

Order matters: the chat UI expects the vLLM app to exist.

```bash
# 1) Deploy vLLM (prefetches weights on first run)
python vllm_server.py

# 2) Deploy Gradio chat UI
python chat_app.py
```

Switch to the 31B variant:

```bash
GEMMA_VARIANT=31b python vllm_server.py
GEMMA_VARIANT=31b python chat_app.py
```
