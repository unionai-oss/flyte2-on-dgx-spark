"""vLLM model-serving app for Gemma 4.

Adapted from `unionai/workshops` gemma4-chat tutorial for this repo.

Deploy:
  python vllm_server.py
  GEMMA_VARIANT=31b python vllm_server.py
"""

from __future__ import annotations

from flyteplugins.vllm import VLLMAppEnvironment

import flyte
import flyte.app

from config import MODEL

_base = flyte.Image.from_base("vllm/vllm-openai:gemma4-cu130")
object.__setattr__(_base, "platform", ("linux/arm64",))
image = (
    _base.clone(registry="localhost:30000", name="gemma4-vllm-image", extendable=True)
    .with_commands(["/usr/bin/python3 -m pip install --no-cache-dir --pre flyteplugins-vllm"])
)

vllm_app = VLLMAppEnvironment(
    name=MODEL.app_name,
    image=image,
    model_hf_path=MODEL.hf_repo,
    model_id=MODEL.model_id,
    resources=flyte.Resources(cpu="8", memory="64Gi", gpu=MODEL.gpu, disk="20Gi"),
    stream_model=True,
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=1800),
    requires_auth=False,
    extra_args=[
        "--max-model-len",
        str(MODEL.max_model_len),
        "--trust-remote-code",
        "--gpu-memory-utilization",
        "0.85",
    ],
)

if __name__ == "__main__":
    import os

    flyte.init_from_config()

    existing_run = os.environ.get("GEMMA_PREFETCH_RUN")
    if existing_run:
        run_name = existing_run
        print(f"Reusing prefetched model from run: {run_name}")
    else:
        import flyte.prefetch
        from flyte.remote import Run

        print(f"Prefetching {MODEL.hf_repo}…")
        run: Run = flyte.prefetch.hf_model(repo=MODEL.hf_repo)
        run.wait()
        print(f"Prefetch run: {run.url}")
        run_name = run.name

    print(f"Deploying vLLM server for {MODEL.model_id} on {MODEL.gpu}…")
    app = flyte.serve(
        vllm_app.clone_with(
            name=vllm_app.name,
            model_path=flyte.app.RunOutput(type="directory", run_name=run_name),
            model_hf_path=None,
        )
    )
    print(f"vLLM app deployed: {app.url}")
    print(f" OpenAI base URL: {app.url}/v1")
    print(f" OpenAPI docs: {app.url}/docs")

