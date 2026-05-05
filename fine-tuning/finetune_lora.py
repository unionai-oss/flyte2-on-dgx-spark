# finetune_lora.py
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "flyte",
#   "torch",
#   "transformers",
#   "datasets",
#   "accelerate",
#   "peft",
#   "safetensors",
# ]
# ///

from __future__ import annotations

import os
from dataclasses import dataclass

import flyte
import flyte.io


@dataclass
class FineTuneConfig:
    # Keep the default small so first run is quick.
    base_model: str = "distilbert/distilbert-base-uncased"
    dataset_name: str = "imdb"
    text_column: str = "text"
    label_column: str = "label"
    max_length: int = 256
    train_samples: int = 2000
    eval_samples: int = 500
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    seed: int = 42


image = flyte.Image.from_uv_script(__file__, name="finetune-lora", registry="localhost:30000")

prep_env = flyte.TaskEnvironment(
    name="finetune_prep",
    resources=flyte.Resources(cpu=4, memory="8Gi"),
    image=image,
)

train_env = flyte.TaskEnvironment(
    name="finetune_train",
    resources=flyte.Resources(cpu=6, memory="24Gi", gpu=1),
    image=image,
    depends_on=[prep_env],
)


@prep_env.task(cache=flyte.Cache(behavior="auto"))
async def prepare_dataset(cfg: FineTuneConfig) -> flyte.io.Dir:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset(cfg.dataset_name)
    ds_train = ds["train"].shuffle(seed=cfg.seed).select(range(cfg.train_samples))
    ds_eval = ds["test"].shuffle(seed=cfg.seed).select(range(cfg.eval_samples))

    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)

    def tokenize(batch):
        return tok(
            batch[cfg.text_column],
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
        )

    ds_train = ds_train.map(tokenize, batched=True, remove_columns=[cfg.text_column])
    ds_eval = ds_eval.map(tokenize, batched=True, remove_columns=[cfg.text_column])

    out_dir = os.path.join(os.getcwd(), "tokenized")
    os.makedirs(out_dir, exist_ok=True)
    ds_train.save_to_disk(os.path.join(out_dir, "train"))
    ds_eval.save_to_disk(os.path.join(out_dir, "eval"))

    return await flyte.io.Dir.from_local(out_dir)


@flyte.trace
def traced(msg: str) -> None:
    print(msg, flush=True)


@train_env.task
async def finetune_lora(cfg: FineTuneConfig, tokenized: flyte.io.Dir) -> tuple[flyte.io.Dir, dict]:
    import json

    import torch
    from datasets import load_from_disk
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    traced("Downloading tokenized dataset artifact")
    local_dir = await tokenized.download()
    ds_train = load_from_disk(os.path.join(local_dir, "train"))
    ds_eval = load_from_disk(os.path.join(local_dir, "eval"))

    traced("Loading model/tokenizer")
    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.base_model, num_labels=2)

    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="SEQ_CLS")
    model = get_peft_model(model, lora)

    collator = DataCollatorWithPadding(tokenizer=tok)

    out_dir = os.path.join(os.getcwd(), "outputs")
    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        evaluation_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_steps=200,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=cfg.seed,
    )

    def compute_metrics(eval_pred):
        import numpy as np

        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean().item()
        return {"accuracy": acc}

    traced("Starting training")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    metrics = dict(train_result.metrics)
    metrics.update(trainer.evaluate())

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)

    traced("Saving LoRA adapter")
    adapter_dir = os.path.join(out_dir, "lora_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)

    return await flyte.io.Dir.from_local(out_dir), metrics


@prep_env.task
async def main(cfg: FineTuneConfig = FineTuneConfig()) -> dict:
    tokenized = await prepare_dataset(cfg)
    _artifacts, metrics = await finetune_lora(cfg, tokenized)
    return metrics


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(run.url)
    run.wait()
    print(run.outputs())

