import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd
import torch
from torch import nn

from lib.defaults import get_default_kwargs
from lib.evaluator import QuantusEvaluator, default_explainers, default_metrics
from lib.setup import setup_notebook
from lib.surrogates import (
    PVTAttention,
    soften_module_inplace_,
)


@dataclass
class Ablation:
    parameter: str
    value: float
    default_value: float
    explainers: Tuple[str, ...]
    temperature_updates: Optional[Dict[Type[nn.Module], float]] = None
    pga_updates: Dict[str, float] = field(default_factory=dict)


TAU_ABLATIONS = (
    ("tau_relu", (nn.ReLU,), (0.3, 0.6, 1.0)),
    ("tau_maxpool", (nn.MaxPool2d,), (0.01, 0.3, 0.5)),
    ("tau_gelu", (nn.GELU,), (0.5, 1.0, 2.0)),
    (
        "tau_attention",
        (nn.MultiheadAttention, PVTAttention),
        (0.5, 1.0, 5.0),
    ),
)


def model_contains_any(model, module_classes):
    return any(isinstance(module, module_classes) for module in model.modules())


def build_ablations(model, default_temperatures):
    ablations = []

    for parameter, module_classes, values in TAU_ABLATIONS:
        if not model_contains_any(model, module_classes):
            print(f"Skipping {parameter}: the model has no matching modules.")
            continue

        default_value = values[1]
        for value in (values[0], values[2]):
            temperature_updates = {
                module_class: value
                for module_class in module_classes
                if module_class in default_temperatures
            }
            ablations.append(
                Ablation(
                    parameter=parameter,
                    value=value,
                    default_value=default_value,
                    explainers=("PullbackAscent", "SoftPullback"),
                    temperature_updates=temperature_updates,
                )
            )

    for value in (3, 10):
        ablations.append(
            Ablation(
                parameter="K",
                value=value,
                default_value=5,
                explainers=("PullbackAscent",),
                pga_updates={"steps": value},
            )
        )

    for value in (10, 40):
        ablations.append(
            Ablation(
                parameter="alpha",
                value=value,
                default_value=20,
                explainers=("PullbackAscent",),
                pga_updates={"alpha": value},
            )
        )

    return ablations


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def take_batches(loader, n_batches):
    batches = []
    for batch_idx, batch in enumerate(loader):
        batches.append(batch)
        if batch_idx + 1 >= n_batches:
            break
    return batches


def format_results(results, ablation, model_name):
    df = QuantusEvaluator.as_dataframe(results)
    df.index.name = "explainer"
    df = df.reset_index()
    df.insert(0, "model_name", model_name)
    df.insert(1, "parameter", ablation.parameter)
    df.insert(2, "value", ablation.value)
    df.insert(3, "default_value", ablation.default_value)
    return df


def main(args):
    if args.n_batches < 1:
        raise ValueError("--n_batches must be at least 1.")

    device, loader, model = setup_notebook(
        batch_size=args.batch_size,
        model_name=args.model_name,
        model_source=args.model_source,
        seed=args.seed,
        dataset=args.dataset,
    )
    (
        default_temperatures,
        default_pga_kwargs_counterfactual,
        _,
        _,
    ) = get_default_kwargs()

    ablations = build_ablations(model, default_temperatures)
    batches = take_batches(loader, args.n_batches)
    if not batches:
        raise RuntimeError("The data loader returned no batches.")

    metrics_filter = None
    if args.test_mode:
        metrics_filter = [
            "infidelity",
            "faithfulness_correlation",
            "faithfulness_estimate",
            "random_logit",
        ]

    output_file = f"{args.output_file}_model_name={args.model_name}"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    all_results = []

    for experiment_idx, ablation in enumerate(ablations, start=1):
        print(
            f"Running ablation {experiment_idx}/{len(ablations)}: "
            f"{ablation.parameter}={ablation.value} for "
            f"{', '.join(ablation.explainers)}"
        )

        temperatures = {
            **default_temperatures,
            **(ablation.temperature_updates or {}),
        }
        soften_module_inplace_(
            model,
            temperatures=temperatures,
            standard_backward=False,
            fill_default_temperatures=True,
        )

        pga_kwargs = {
            **default_pga_kwargs_counterfactual,
            **ablation.pga_updates,
        }

        # Reuse both the examples and RNG state for a fair one-at-a-time
        # comparison between hyperparameter values.
        seed_everything(args.seed)
        explainers = default_explainers(
            ablation.explainers,
            pga_kwargs_counterfactual=pga_kwargs,
        )
        metrics = default_metrics(metrics_filter)

        evaluator = QuantusEvaluator(
            model,
            metrics,
            explainers,
            device=device,
        )
        results = evaluator.evaluate_loader(batches, n_batches=len(batches))
        all_results.append(format_results(results, ablation, args.model_name))

        # Checkpoint after every (potentially long) configuration.
        pd.concat(all_results, ignore_index=True).to_pickle(f"{output_file}.pkl")

    print(f"Results saved to {output_file}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one-at-a-time Semantic Pullbacks hyperparameter ablations."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output path without the .pkl extension or model name.",
    )
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument(
        "--model_source",
        type=str,
        default="torchvision",
        choices=["torchvision", "timm"],
    )
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--n_batches", type=int, default=20)
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Use only two metrics for a quicker smoke test.",
    )
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument(
        "--dataset",
        default="imagenette",
        choices=["imagenette", "imagenet"],
    )
    main(parser.parse_args())
