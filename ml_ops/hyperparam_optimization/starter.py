"""CLI entrypoint for running BERT hyperparameter sweeps with Temporal.

This script stays intentionally small and tutorial-friendly:

- Load a sweep preset from configs.py.
- Connect to Temporal and execute the ladder workflow.
- Print a compact summary of the best runs.
"""

import asyncio
import argparse
import json
from pathlib import Path

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from configs import get_sweep_request, get_sweep_request_from_dict, list_presets
from workflows import (
    LadderSweepWorkflow,
)

# ------------------------------------------------------------------------------
# Starter Main Function
# ------------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a BERT ladder sweep with Temporal.")
    parser.add_argument(
        "--config",
        default="fast",
        choices=list_presets(),
        help="Which preset sweep to run.",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Optional path to a JSON file with a SweepRequest-like dict.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Optional experiment ID prefix. Default: generated UUID.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for preset generation.",
    )
    args = parser.parse_args()

    # 1. Connect to Temporal Server using the Pydantic data converter so our
    # request/response models can be passed directly as workflow arguments.
    client = await Client.connect("localhost:7233", data_converter=pydantic_data_converter)

    # 2. Pick the request to run. For the tutorial this is a single ladder
    # sweep, but you can easily swap in a different ``SweepRequest`` here.
    if args.config_file:
        data = json.loads(Path(args.config_file).read_text())
        request = get_sweep_request_from_dict(data)
    else:
        request = get_sweep_request(args.config, experiment_id=args.experiment_id, seed=args.seed)

    # 3. Start the workflow and wait for the result. We call the ``run`` method
    # on the workflow class directly; Temporal will assign a fresh workflow
    # execution to the ID provided below.
    result = await client.execute_workflow(
        LadderSweepWorkflow.run,
        request,
        id=f"bert-ladder-{request.experiment_id}",
        task_queue="bert-eval-task-queue",
    )

    # 4. Print a concise, tabular summary of the winning candidate & result.
    results = result if isinstance(result, (list, tuple)) else [result]

    print("\n=== BERT evaluation summary ===")
    header = f"{'run_id':<36} {'dataset':<20} {'split':<10} {'examples':>10} {'accuracy':>9}"
    print(header)
    print("-" * len(header))

    for item in results:
        dataset = f"{item.dataset_name}/{item.dataset_config_name}"
        print(
            f"{item.run_id:<36} "
            f"{dataset:<20} "
            f"{item.split:<10} "
            f"{item.num_examples:>10} "
            f"{item.accuracy:>9.3f}",
        )

    # If the ladder workflow annotated the best result with ablation metadata,
    # print the improvement in accuracy over the ablation baseline.
    best = results[0]
    baseline = getattr(best, "baseline_accuracy", None)
    improvement = getattr(best, "improvement_vs_baseline", None)
    if baseline is not None and improvement is not None:
        print(
            "\nBest run "
            f"{best.run_id} improved accuracy by {improvement:.3f} "
            f"over the ablation baseline "
            f"(baseline={baseline:.3f}, best={best.accuracy:.3f}).",
        )


# CLI Hook
if __name__ == "__main__":
    asyncio.run(main())
