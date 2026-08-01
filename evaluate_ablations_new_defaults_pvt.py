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
from lib.surrogates import PVTAttention, soften_module_inplace_


@dataclass
class Ablation:
    parameter: str
    value: object
    default_value: object
    explainers: Tuple[str, ...]
    temperature_updates: Optional[Dict[Type[nn.Module], float]] = None
    pga_updates: Dict[str, float] = field(default_factory=dict)
    is_default: bool = False


# PVT-focused baseline defaults from PVT ablation analysis.
TAU_ABLATIONS = (
    ("tau_attention", (PVTAttention,), (1.0, 0.5, 5.0)),
    ("tau_gelu", (nn.GELU,), (1.0, 0.5, 2.0)),
)
K_ABLATIONS = (5, 3, 2, 1, 10)
ALPHA_ABLATIONS = (40, 20, 10, 5)


def model_contains_any(model, module_classes):
    return any(isinstance(module, module_classes) for module in model.modules())


def build_ablations(model, base_temperatures):
    ablations = [
        Ablation(
            parameter="default",
            value="default",
            default_value="default",
            explainers=("PullbackAscent", "SoftPullback"),
            is_default=True,
        )
    ]

    for parameter, module_classes, values in TAU_ABLATIONS:
        if not model_contains_any(model, module_classes):
            print(f"Skipping {parameter}: the model has no matching modules.")
            continue

        default_value = values[0]
        for value in dict.fromkeys(values[1:]):
            temperature_updates = {
                module_class: value
                for module_class in module_classes
                if module_class in base_temperatures
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

    for value in dict.fromkeys(K_ABLATIONS[1:]):
        ablations.append(
            Ablation(
                parameter="K",
                value=value,
                default_value=K_ABLATIONS[0],
                explainers=("PullbackAscent",),
                pga_updates={"steps": value},
            )
        )

    for value in dict.fromkeys(ALPHA_ABLATIONS[1:]):
        ablations.append(
            Ablation(
                parameter="alpha",
                value=value,
                default_value=ALPHA_ABLATIONS[0],
                explainers=("PullbackAscent",),
                pga_updates={"alpha": value},
            )
        )

    return ablations


def ablation_label(ablation):
    return f"{ablation.parameter}={ablation.value}"


def select_ablations(ablations, selector):
    if not selector:
        return ablations

    selector = selector.strip()

    if selector.isdigit():
        index = int(selector)
        if index < 1 or index > len(ablations):
            raise ValueError(
                f"--only_ablation index out of range: {index}. "
                f"Expected 1..{len(ablations)}."
            )
        return [ablations[index - 1]]

    matches = [a for a in ablations if ablation_label(a) == selector]
    if selector.lower() == "default":
        matches = [a for a in ablations if a.is_default]

    if len(matches) == 1:
        return matches

    available = "\n".join(
        f"  {idx}. {ablation_label(ablation)}"
        for idx, ablation in enumerate(ablations, start=1)
    )
    raise ValueError(
        "Could not uniquely select one ablation with --only_ablation. "
        "Use a 1-based index (e.g. --only_ablation 3), 'default', "
        "or an exact 'parameter=value' label (e.g. --only_ablation K=1).\n"
        f"Available ablations:\n{available}"
    )


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
    df.insert(4, "is_default", ablation.is_default)
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

    base_temperatures = {**default_temperatures}
    for _, module_classes, values in TAU_ABLATIONS:
        default_value = values[0]
        for module_class in module_classes:
            if module_class in base_temperatures:
                base_temperatures[module_class] = default_value

    base_pga_kwargs = {
        **default_pga_kwargs_counterfactual,
        "steps": K_ABLATIONS[0],
        "alpha": ALPHA_ABLATIONS[0],
    }

    ablations = build_ablations(model, base_temperatures)
    ablations = select_ablations(ablations, args.only_ablation)

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

    test_mode_tag = "test" if args.test_mode else "full"
    output_file = (
        f"{args.output_file}_preset=new_defaults_pvt"
        f"_model_name={args.model_name}"
        f"_n_batches={args.n_batches}_test_mode={test_mode_tag}"
    )
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    all_results = []

    for experiment_idx, ablation in enumerate(ablations, start=1):
        print(
            f"Running ablation {experiment_idx}/{len(ablations)}: "
            f"{ablation.parameter}={ablation.value} for "
            f"{', '.join(ablation.explainers)}"
        )

        temperatures = {
            **base_temperatures,
            **(ablation.temperature_updates or {}),
        }
        soften_module_inplace_(
            model,
            temperatures=temperatures,
            standard_backward=False,
            fill_default_temperatures=True,
        )

        pga_kwargs = {
            **base_pga_kwargs,
            **ablation.pga_updates,
        }

        seed_everything(args.seed)
        explainers = default_explainers(
            ablation.explainers,
            pga_kwargs_counterfactual=pga_kwargs,
        )
        metrics = default_metrics(
            metrics_filter,
            random_logit_seed=args.seed,
        )

        evaluator = QuantusEvaluator(
            model,
            metrics,
            explainers,
            device=device,
        )
        results = evaluator.evaluate_loader(batches, n_batches=len(batches))
        all_results.append(format_results(results, ablation, args.model_name))

        pd.concat(all_results, ignore_index=True).to_pickle(f"{output_file}.pkl")

    print(f"Results saved to {output_file}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run PVT-focused Semantic Pullbacks ablations around a new-default "
            "preset (tau_attention=1.0, tau_gelu=1.0, K=5, alpha=40)."
        )
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output path without the .pkl extension or model name.",
    )
    parser.add_argument("--model_name", type=str, default="pvt_v2_b1")
    parser.add_argument(
        "--model_source",
        type=str,
        default="timm",
        choices=["torchvision", "timm"],
    )
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--n_batches", type=int, default=10)
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Use a reduced metric subset for a quicker run.",
    )
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument(
        "--dataset",
        default="imagenet",
        choices=["imagenette", "imagenet"],
    )
    parser.add_argument(
        "--only_ablation",
        type=str,
        default=None,
        help=(
            "Run only one ablation. Accepts a 1-based index (e.g. 3), "
            "'default', or an exact label 'parameter=value' (e.g. K=3)."
        ),
    )
    main(parser.parse_args())
