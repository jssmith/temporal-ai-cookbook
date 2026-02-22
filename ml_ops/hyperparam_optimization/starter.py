"""CLI entrypoint for running BERT hyperparameter sweeps with Temporal.

Usage examples::

    # Quick demo run (a few minutes on a MacBook):
    uv run -m starter --config deberta-sst2 --num-trials 4 --max-concurrency 2

    # Full sweep with defaults:
    uv run -m starter --config deberta-sst2

    # List available configs:
    uv run -m starter --list-configs
"""

import argparse
import asyncio
import sys

from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter

from sample_configs import CONFIGS, build_sweep_request
from workflows import LadderSweepWorkflow


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a BERT ladder hyperparameter sweep via Temporal.",
    )
    parser.add_argument(
        "--config",
        choices=list(CONFIGS.keys()),
        default="deberta-sst2",
        help="Base model/dataset configuration to sweep over (default: deberta-sst2).",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=12,
        help="Number of trials for the sweep (default: 12). Use 4 for a quick demo.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Max parallel training/eval pipelines (default: 4). Use 2 for laptops.",
    )
    parser.add_argument(
        "--temporal-address",
        default="localhost:7233",
        help="Temporal server address (default: localhost:7233).",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="Print available config names and exit.",
    )

    args = parser.parse_args()

    if args.list_configs:
        for name, cfg in CONFIGS.items():
            model = cfg.fine_tune_config.model_name
            dataset = f"{cfg.fine_tune_config.dataset_name}/{cfg.fine_tune_config.dataset_config_name}"
            print(f"  {name:<25} {model:<40} {dataset}")
        sys.exit(0)

    base_config = CONFIGS[args.config]
    request = build_sweep_request(
        base_config=base_config,
        num_trials=args.num_trials,
        max_concurrency=args.max_concurrency,
    )

    # Connect to Temporal Server using the Pydantic data converter so our
    # request/response models can be passed directly as workflow arguments.
    client = await Client.connect(
        args.temporal_address,
        data_converter=pydantic_data_converter,
    )

    # Start the workflow and wait for the result.
    result = await client.execute_workflow(
        LadderSweepWorkflow.run,
        request,
        id=f"bert-ladder-{request.experiment_id}",
        task_queue="bert-eval-task-queue",
    )

    # Print a concise, tabular summary of the winning candidates.
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
