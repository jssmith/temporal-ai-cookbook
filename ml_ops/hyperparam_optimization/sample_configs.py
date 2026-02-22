"""Sample configurations for BERT hyperparameter sweeps.

Each config pairs a model with a dataset and sensible defaults. All configs
use ``random.randint(0, 10000)`` for seeds so that every import produces a
fresh, non-deterministic starting point -- the sweep workflow will further
randomize seeds via the ``set_seed`` activity.
"""

import random

from models import (
    BertEvalRequest,
    BertFineTuneConfig,
    CoordinatorWorkflowConfig,
    SweepRequest,
    SweepSpace,
)

# ---------------------------------------------------------------------------
# Model / dataset configurations
# ---------------------------------------------------------------------------

# BERT Uncased on GLUE SST-2
config_1 = CoordinatorWorkflowConfig(
    fine_tune_config=BertFineTuneConfig(
        model_name="bert-base-uncased",
        dataset_name="glue",
        dataset_config_name="sst2",
        num_epochs=2,
        batch_size=8,
        learning_rate=2e-5,
        max_seq_length=128,
        use_gpu=True,
        max_train_samples=3_000,
        max_eval_samples=2_000,
        seed=random.randint(0, 10000),
    ),
    evaluation_config=BertEvalRequest(
        dataset_name="glue",
        dataset_config_name="sst2",
        split="validation",
        max_eval_samples=1_000,
        max_seq_length=128,
        batch_size=32,
        use_gpu=True,
    ),
)

# BERT Cased on GLUE SST-2
config_2 = CoordinatorWorkflowConfig(
    fine_tune_config=BertFineTuneConfig(
        model_name="bert-base-cased",
        dataset_name="glue",
        dataset_config_name="sst2",
        num_epochs=10,
        batch_size=16,
        learning_rate=3e-5,
        max_seq_length=128,
        use_gpu=True,
        max_train_samples=3_000,
        max_eval_samples=2_000,
        seed=random.randint(0, 10000),
    ),
    evaluation_config=BertEvalRequest(
        dataset_name="glue",
        dataset_config_name="sst2",
        split="validation",
        max_eval_samples=1_000,
        max_seq_length=128,
        batch_size=32,
        use_gpu=True,
    ),
)

# BERT Uncased on IMDB
config_3 = CoordinatorWorkflowConfig(
    fine_tune_config=BertFineTuneConfig(
        model_name="bert-base-uncased",
        dataset_name="imdb",
        dataset_config_name="plain_text",
        num_epochs=10,
        batch_size=32,
        learning_rate=2e-5,
        max_seq_length=256,
        use_gpu=True,
        max_train_samples=5_000,
        max_eval_samples=2_000,
        seed=random.randint(0, 10000),
    ),
    evaluation_config=BertEvalRequest(
        dataset_name="imdb",
        dataset_config_name="plain_text",
        split="test",
        max_eval_samples=1_000,
        max_seq_length=256,
        batch_size=32,
        use_gpu=True,
        seed=random.randint(0, 10000),
    ),
)

# DistilBERT on GLUE SST-2
config_4 = CoordinatorWorkflowConfig(
    fine_tune_config=BertFineTuneConfig(
        model_name="distilbert-base-uncased",
        dataset_name="glue",
        dataset_config_name="sst2",
        num_epochs=2,
        batch_size=4,
        learning_rate=5e-5,
        max_seq_length=64,
        use_gpu=True,
        max_train_samples=2_000,
        max_eval_samples=1_000,
        shuffle_before_select=True,
        seed=random.randint(0, 10000),
    ),
    evaluation_config=BertEvalRequest(
        dataset_name="glue",
        dataset_config_name="sst2",
        split="validation",
        max_eval_samples=1_000,
        max_seq_length=64,
        batch_size=32,
        use_gpu=True,
        seed=random.randint(0, 10000),
    ),
)

# MiniLM on GLUE SST-2
config_5 = CoordinatorWorkflowConfig(
    fine_tune_config=BertFineTuneConfig(
        model_name="microsoft/MiniLM-L12-H384-uncased",
        dataset_name="glue",
        dataset_config_name="sst2",
        num_epochs=2,
        batch_size=4,
        learning_rate=3e-5,
        max_seq_length=64,
        use_gpu=True,
        max_train_samples=2_000,
        max_eval_samples=1_000,
        shuffle_before_select=True,
        seed=random.randint(0, 10000),
    ),
    evaluation_config=BertEvalRequest(
        dataset_name="glue",
        dataset_config_name="sst2",
        split="validation",
        max_eval_samples=1_000,
        max_seq_length=64,
        batch_size=32,
        use_gpu=True,
        seed=random.randint(0, 10000),
    ),
)

# DeBERTa-v3-small on GLUE SST-2
config_6 = CoordinatorWorkflowConfig(
    fine_tune_config=BertFineTuneConfig(
        model_name="microsoft/deberta-v3-small",
        dataset_name="glue",
        dataset_config_name="sst2",
        num_epochs=2,
        batch_size=2,
        learning_rate=2e-5,
        max_seq_length=64,
        use_gpu=True,
        max_train_samples=2_000,
        max_eval_samples=1_000,
        shuffle_before_select=True,
        seed=random.randint(0, 10000),
    ),
    evaluation_config=BertEvalRequest(
        dataset_name="glue",
        dataset_config_name="sst2",
        split="validation",
        max_eval_samples=1_000,
        max_seq_length=64,
        batch_size=32,
        use_gpu=True,
        seed=random.randint(0, 10000),
    ),
)

# SciBERT on SciCite
config_7 = CoordinatorWorkflowConfig(
    fine_tune_config=BertFineTuneConfig(
        model_name="allenai/scibert_scivocab_uncased",
        dataset_name="scicite",
        dataset_config_name="default",
        use_gpu=True,
        max_train_samples=2_000,
        max_eval_samples=1_000,
        shuffle_before_select=True,
        seed=random.randint(0, 10000),
    ),
    evaluation_config=BertEvalRequest(
        dataset_name="scicite",
        dataset_config_name="default",
        split="validation",
        max_eval_samples=1_000,
        use_gpu=True,
        seed=random.randint(0, 10000),
    ),
)


# ---------------------------------------------------------------------------
# Lookup table for CLI usage
# ---------------------------------------------------------------------------
CONFIGS: dict[str, CoordinatorWorkflowConfig] = {
    "bert-uncased-sst2": config_1,
    "bert-cased-sst2": config_2,
    "bert-uncased-imdb": config_3,
    "distilbert-sst2": config_4,
    "minilm-sst2": config_5,
    "deberta-sst2": config_6,
    "scibert-scicite": config_7,
}


def build_sweep_request(
    base_config: CoordinatorWorkflowConfig,
    num_trials: int = 12,
    max_concurrency: int = 4,
    experiment_id: str | None = None,
) -> SweepRequest:
    """Build a ``SweepRequest`` from a base config and sweep parameters.

    This is the recommended way to construct a sweep request -- it wires up a
    sensible ``SweepSpace`` and generates a unique experiment ID if none is
    provided.
    """
    import uuid

    if experiment_id is None:
        experiment_id = f"bert-ladder-sweep-{uuid.uuid4()}"

    return SweepRequest(
        experiment_id=experiment_id,
        base=base_config,
        space=SweepSpace(
            learning_rate=(5e-5, 1e-5),
            batch_size=[2, 32],
            num_epochs=[2, 8],
            max_seq_length=[64, 256],
        ),
        num_trials=num_trials,
        max_concurrency=max_concurrency,
        seed=random.randint(0, 10000),
    )
