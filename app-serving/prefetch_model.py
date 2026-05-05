"""Prefetch a Gemma 4 model from Hugging Face into the Flyte object store.

Usage:
  flyte create secret HF_TOKEN
  python prefetch_model.py
  GEMMA_VARIANT=31b python prefetch_model.py
"""

from __future__ import annotations

import flyte
import flyte.prefetch
from flyte.remote import Run

from config import MODEL


def prefetch() -> Run:
    flyte.init_from_config()
    print(f"Prefetching {MODEL.hf_repo} → Flyte object store…")
    run: Run = flyte.prefetch.hf_model(repo=MODEL.hf_repo)
    run.wait()
    print(f"Prefetch run: {run.url}")
    print(f"Run name (use GEMMA_PREFETCH_RUN=...): {run.name}")
    return run


if __name__ == "__main__":
    prefetch()

