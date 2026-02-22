<!--
description: Build durable custom hyperparameter optimization to save money
tags: [ML Ops, HPO, autoML, python]
priority: 700
-->

# Durable Hyperparameter Optimization for BERT Fine-Tuning

This example uses Temporal to orchestrate a **multi-stage hyperparameter sweep** for fine-tuning BERT-family models on text classification tasks (sentiment analysis, citation intent, etc.). The sweep explores learning rate, batch size, sequence length, and epoch count using a TPE-inspired "ladder" schedule that starts many cheap trials, prunes the losers, and promotes the best candidates to longer training runs -- all with full durability so you can kill workers, restart them, and pick up exactly where you left off.

## How it works

```mermaid
graph TD
    S[starter.py] -->|SweepRequest| LS[LadderSweepWorkflow]
    LS -->|Stage 1: many cheap trials| C1[CoordinatorWorkflow]
    LS -->|Stage 2: survivors + new| C2[CoordinatorWorkflow]
    LS -->|Stage N: final rung| CN[CoordinatorWorkflow]
    C1 --> T1[CheckpointedBertTrainingWorkflow]
    C1 --> T2[CheckpointedBertTrainingWorkflow]
    C1 --> E1[BertEvalWorkflow]
    C1 --> E2[BertEvalWorkflow]
    T1 -->|bert-training-task-queue| FT[fine_tune_bert activity]
    E1 -->|bert-eval-task-queue| EV[evaluate_bert_model activity]
    FT -->|signals| T1
```

**LadderSweepWorkflow** runs multiple stages ("rungs"). Each rung:
1. Promotes the best survivors from the previous rung (more epochs, more data).
2. Proposes new candidates using a TPE-style sampler biased toward good regions.
3. Fans out to **CoordinatorWorkflow**, which starts training + evaluation child workflows.
4. Ranks results by accuracy and selects survivors for the next rung.

## Quickstart

From `ml_ops/hyperparam_optimization/`:

1. **Start Temporal Server** (if not already running):

   ```bash
   temporal server start-dev
   ```

2. **Start the training worker** (GPU / MPS recommended):

   ```bash
   uv sync --dev
   uv run -m training_worker
   ```

3. **Start the orchestration worker** (CPU is fine):

   ```bash
   uv run -m orchestration_worker
   ```

4. **Run a quick demo sweep** (~5 minutes on a MacBook):

   ```bash
   uv run -m starter --config deberta-sst2 --num-trials 4 --max-concurrency 2
   ```

   Or a full sweep (longer, more thorough):

   ```bash
   uv run -m starter --config deberta-sst2 --num-trials 12 --max-concurrency 4
   ```

   Use `--list-configs` to see all available model/dataset combinations.

## Understanding the output

Open the Temporal UI at [http://localhost:8233](http://localhost:8233). Look for the workflow whose ID starts with `bert-ladder-` -- this is the top-level `LadderSweepWorkflow`. Expand its child workflows to see individual training and evaluation runs.

**Expected log output you can ignore:**

The `UNEXPECTED` and `MISSING` key warnings in the console are normal. They come from Hugging Face Transformers loading a pre-trained language model checkpoint into a classification head -- the LM-specific weights (e.g., `lm_predictions.*`) are discarded and a new classification layer (`classifier.*`) is initialized from scratch. This is standard BERT fine-tuning behavior.

**Monitoring training progress:**

- Each training activity sends periodic **heartbeats** visible in the Temporal UI under the activity's details.
- **Checkpoint signals** are sent to the parent workflow whenever the Trainer saves a checkpoint. You can query `get_latest_checkpoint` on any running `CheckpointedBertTrainingWorkflow` to see the most recent loss and step.
- Training loss is logged to the console at regular intervals by the HuggingFace Trainer.

## Where checkpoints and data are stored

- **Model checkpoints:** `./bert_runs/{run_id}/` -- each training run writes its model, tokenizer, and mid-run checkpoints here.
- **Dataset snapshots:** `./data_snapshots/` -- downloaded datasets are saved as JSONL snapshots keyed by content hash for reproducibility.

Both directories are created automatically. They can grow large during a full sweep.

## Durability demo

To see Temporal's durability in action:

1. Start both workers and run the starter as above.
2. Wait until you see training logs from `BertFineTuneActivities` (look for lines mentioning `Starting BERT fine-tuning run`). This typically takes 30-60 seconds after the sweep starts.
3. **Kill the training worker** (Ctrl-C in its terminal).
4. Restart it:

   ```bash
   uv run -m training_worker
   ```

5. Observe in the Temporal UI and logs that:
   - In-flight training children resume from the latest checkpoint.
   - The ladder sweep continues from the last completed stage, not from scratch.
   - The final leaderboard is coherent.

## Scaling out

For production or cloud deployments, training and orchestration workers can run on separate machines. The key requirement is a **shared filesystem** for model checkpoints and dataset snapshots, since the training worker writes to `./bert_runs/` and the evaluation worker reads from it.

Options for shared storage:
- **AWS:** EFS (simple) or FSx for Lustre (high throughput)
- **GCP:** Filestore or Cloud Storage FUSE
- **Azure:** Azure Files or Azure NetApp Files
- **On-prem / Kubernetes:** NFS, CephFS, or any POSIX-compatible shared mount

Alternatively, modify the activities to upload/download checkpoints to object storage (S3, GCS) instead of relying on a shared filesystem.

## Architecture

### Workflow hierarchy

| Workflow | Responsibility |
|---|---|
| `LadderSweepWorkflow` | Multi-stage sweep with TPE-inspired proposal and survivor promotion |
| `SweepWorkflow` | Simpler random sweep (single stage) |
| `CoordinatorWorkflow` | Fan-out training + evaluation for a batch of configs |
| `CheckpointedBertTrainingWorkflow` | Single training run with checkpoint signals and dataset snapshots |
| `BertEvalWorkflow` | Single evaluation run against a saved checkpoint |

### Activities

All ML and I/O code lives in activities (`bert_activities.py`):

| Activity | Task queue | What it does |
|---|---|---|
| `fine_tune_bert` | `bert-training-task-queue` | Fine-tune a model with checkpointing and heartbeats |
| `create_dataset_snapshot` | `bert-training-task-queue` | Download and snapshot a dataset as JSONL |
| `evaluate_bert_model` | `bert-eval-task-queue` | Load a saved model and compute accuracy on a dataset split |
| `set_seed` | `bert-eval-task-queue` | Generate a non-deterministic seed for a trial |

### Determinism model

Workflows are deterministic and contain no I/O:
- All randomness in sweep workflows uses `workflow.random()`, seeded by `SweepRequest.seed`.
- Non-deterministic operations (dataset downloads, model training, seed generation) are confined to activities.
- Pydantic types are imported inside `workflow.unsafe.imports_passed_through()` to keep replay safe.

This means you can replay any workflow from its event history to debug orchestration logic without re-running ML workloads.

### Task queues

Two task queues separate concerns:
- **`bert-training-task-queue`**: training and snapshot activities (run on GPU-capable machines)
- **`bert-eval-task-queue`**: orchestration workflows + evaluation activities (CPU is fine)

## Adapting this example

Every adaptation point in the codebase is marked with a `# ADAPT:` comment. Run this to find them all:

```bash
grep -rn 'ADAPT' *.py
```

### What to change (checklist)

1. **Your training config** (`models.py`) -- Rename `BertFineTuneConfig`, add/remove fields for your model's hyperparameters.
2. **Your eval result** (`models.py`) -- Rename `BertEvalResult`, change metric fields (e.g., replace `accuracy` with `mse` for regression).
3. **Your search space** (`models.py`) -- Update `SweepSpace` dimensions to match your new config fields.
4. **Your scoring metric** (`workflows.py`) -- Edit `_score_eval_result()` to return the metric you want to maximize.
5. **Your sampling code** (`workflows.py`) -- Edit `_sample_random_hyperparams()` and the TPE-informed branch in `_tpe_suggest()` to sample your new hyperparameters.
6. **Your activities** (`bert_activities.py`) -- Replace the training, evaluation, and checkpointing implementations with your own model's logic.
7. **Constants and names** (`workflows.py`) -- Update `RUNS_DIR` and `TRAINING_TASK_QUEUE`; update workers (`training_worker.py`, `orchestration_worker.py`) to match.
8. **Sample configs** (`sample_configs.py`) -- Add your model/dataset combinations.

### Example: Adapting for image classification

Here is what steps 1-5 look like for a ResNet/CIFAR-10 scenario (illustrative, not runnable):

**Step 1 -- Training config** (`models.py`):
```python
class ImageTrainConfig(BaseModel):
    model_name: str = "resnet50"
    dataset_name: str = "cifar10"
    num_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    image_size: int = 224
    weight_decay: float = 1e-4
    use_gpu: bool = True
    max_train_samples: int | None = 5_000
    seed: int = 42
    run_id: str | None = None
```

**Step 2 -- Eval result** (`models.py`):
```python
class ImageEvalResult(BaseModel):
    run_id: str
    dataset_name: str
    split: str
    num_examples: int
    accuracy: float
    top5_accuracy: float
```

**Step 3 -- Search space** (`models.py`):
```python
class SweepSpace(BaseModel):
    learning_rate: tuple[float, float] = (1e-4, 1e-2)
    batch_size: list[int] = [32, 64, 128]
    image_size: list[int] = [160, 224]
    weight_decay: tuple[float, float] = (1e-5, 1e-3)
```

**Step 4 -- Scoring metric** (`workflows.py`):
```python
def _score_eval_result(result: ImageEvalResult) -> float:
    return result.top5_accuracy
```

**Step 5 -- Sampling code** (`workflows.py`):
```python
def _sample_random_hyperparams(rng, cfg, space) -> None:
    cfg.fine_tune_config.batch_size = rng.choice(space.batch_size)
    cfg.fine_tune_config.image_size = rng.choice(space.image_size)

    lo, hi = space.learning_rate
    u = rng.random()
    cfg.fine_tune_config.learning_rate = float(
        math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    )

    lo, hi = space.weight_decay
    u = rng.random()
    cfg.fine_tune_config.weight_decay = float(
        math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
    )
```

### Files you do NOT need to change

The ladder/TPE sweep structure, coordinator fan-out pattern, and checkpoint-signal mechanism are reusable as-is. Specifically:

- `LadderSweepWorkflow` -- stage progression, survivor selection, and TPE proposal loop
- `CoordinatorWorkflow` -- fan-out training + evaluation and `run_id` propagation
- `CheckpointedBertTrainingWorkflow` -- checkpoint signaling and query pattern
- `BertEvalWorkflow` -- evaluation child workflow structure

These workflows only interact with your model through the config types, the scoring function, and the activities.

**Connect to Temporal Cloud:** Pass `--temporal-address <your-cloud-address>` to the starter and update the worker connection code with your namespace and TLS credentials.

## Repo map

| File | Purpose |
|---|---|
| `starter.py` | CLI entrypoint -- parses args, builds a `SweepRequest`, starts the workflow |
| `sample_configs.py` | Pre-built model/dataset configs and `build_sweep_request()` helper |
| `models.py` | All Pydantic data models (workflow inputs, activity inputs, results) |
| `workflows.py` | Workflow definitions (training, eval, coordinator, sweep, ladder) |
| `bert_activities.py` | Activity implementations (training, evaluation, checkpointing, seed) |
| `orchestration_worker.py` | Worker for sweep orchestration + evaluation activities |
| `training_worker.py` | Worker for training + checkpointing activities |
| `tests/` | Workflow integration tests with mocked activities |
| `pyproject.toml` | Python project configuration and dependencies |

## Origin and credits

This example was adapted from the [temporalio/samples-python](https://github.com/temporalio/samples-python) repository and the Temporal AI cookbook's durable training module. It extends the checkpoint-aware training pattern with multi-stage hyperparameter optimization.
