"""Preset configurations for running BERT sweeps.

Keep these in a dedicated module so starter.py stays small and copy/pasteable.
"""

from __future__ import annotations

import random
import uuid

from custom_types import (
    BertEvalRequest,
    BertFineTuneConfig,
    CoordinatorWorkflowConfig,
    SweepRequest,
    SweepSpace,
)


def _seeded_rng(seed: int | None) -> random.Random:
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    return random.Random(seed)


def _base_config(seed: int) -> CoordinatorWorkflowConfig:
    return CoordinatorWorkflowConfig(
        fine_tune_config=BertFineTuneConfig(
            model_name="bert-base-uncased",
            dataset_name="glue",
            dataset_config_name="sst2",
            num_epochs=2,
            batch_size=4,
            learning_rate=2e-5,
            max_seq_length=64,
            use_gpu=True,
            max_train_samples=2_000,
            max_eval_samples=1_000,
            shuffle_before_select=True,
            seed=seed,
        ),
        evaluation_config=BertEvalRequest(
            dataset_name="glue",
            dataset_config_name="sst2",
            split="validation",
            max_eval_samples=1_000,
            max_seq_length=64,
            batch_size=32,
            use_gpu=True,
            seed=seed,
        ),
        dataset_snapshot=None,
    )


def get_sweep_request(
    name: str,
    *,
    experiment_id: str | None = None,
    seed: int | None = None,
) -> SweepRequest:
    """Return a named SweepRequest preset.

    Presets are designed to be copy/paste-friendly and safe on laptops.
    """
    rng = _seeded_rng(seed)
    exp_id = experiment_id or f"Bert-ladder-sweep-{uuid.uuid4()}"

    if name == "fast":
        base = _base_config(seed=rng.randint(0, 10000))
        base.fine_tune_config.use_gpu = False
        base.evaluation_config.use_gpu = False
        base.fine_tune_config.num_epochs = 1
        base.fine_tune_config.max_train_samples = 300
        base.fine_tune_config.max_seq_length = 64
        base.fine_tune_config.batch_size = 2
        base.evaluation_config.batch_size = 8

        return SweepRequest(
            experiment_id=exp_id,
            base=base,
            space=SweepSpace(
                learning_rate=(1e-5, 5e-5),
                batch_size=[2, 4],
                max_seq_length=[64, 128],
                num_epochs=[1, 2],
            ),
            num_trials=3,
            max_concurrency=1,
            seed=rng.randint(0, 10000),
        )

    if name == "ladder":
        base = _base_config(seed=rng.randint(0, 10000))
        return SweepRequest(
            experiment_id=exp_id,
            base=base,
            space=SweepSpace(
                learning_rate=(5e-5, 1e-5),
                batch_size=[2, 4, 8],
                max_seq_length=[64, 128, 256],
                num_epochs=[2, 3, 4],
            ),
            num_trials=12,
            max_concurrency=4,
            seed=rng.randint(0, 10000),
        )

    raise ValueError(f"Unknown preset: {name!r}")


def list_presets() -> list[str]:
    return ["fast", "ladder"]


def get_sweep_request_from_dict(data: dict) -> SweepRequest:
    """Build a SweepRequest from a minimal dict (easy copy/paste).

    Expected keys:
      - experiment_id (str)
      - base (dict for CoordinatorWorkflowConfig)
      - space (dict for SweepSpace)
      - num_trials (int, optional)
      - max_concurrency (int, optional)
      - seed (int, optional)
    """
    if "base" not in data or "space" not in data:
        raise ValueError("config dict must include 'base' and 'space'")

    base = CoordinatorWorkflowConfig(**data["base"])
    space = SweepSpace(**data["space"])

    return SweepRequest(
        experiment_id=data.get("experiment_id", f"custom-sweep-{uuid.uuid4()}"),
        base=base,
        space=space,
        num_trials=int(data.get("num_trials", 4)),
        max_concurrency=int(data.get("max_concurrency", 1)),
        seed=int(data.get("seed", 42)),
    )
