"""Shared config for the Gemma 4 chat app on Flyte 2.

Adapted from `unionai/workshops` gemma4-chat tutorial for this repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelChoice:
    hf_repo: str
    model_id: str  # name exposed over vLLM OpenAI API
    app_name: str  # Flyte app name (DNS-safe, lowercase)
    gpu: int | str  # flyte.Resources gpu spec
    max_model_len: int


GEMMA_4_26B_A4B = ModelChoice(
    hf_repo="google/gemma-4-26B-A4B-it",
    model_id="gemma-4-26b-a4b-it",
    app_name="gemma4-26b-a4b-it-vllm",
    gpu=1,
    max_model_len=8192,
)

GEMMA_4_31B = ModelChoice(
    hf_repo="google/gemma-4-31B-it",
    model_id="gemma-4-31b-it",
    app_name="gemma4-31b-it-vllm",
    gpu=1,  # change to 2 for TP=2 on multi-GPU boxes
    max_model_len=8192,
)

# Override via GEMMA_VARIANT=31b
MODEL = GEMMA_4_31B if os.environ.get("GEMMA_VARIANT") == "31b" else GEMMA_4_26B_A4B

CHAT_APP_NAME = "gemma4-chat-ui"

