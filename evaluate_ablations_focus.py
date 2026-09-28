import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import numpy as np
import pandas as pd
import quantus
import torch
from torch import nn

from lib.defaults import get_default_kwargs
from lib.evaluator import QuantusEvaluator, default_explainers
from evaluate_focus import build_focus_mosaics, empty_cache, index_dataset_by_class
from lib.setup import setup_notebook
from lib.surrogates import (
    PVTAttention,
    soften_module_inplace_,
)


@dataclass
class Ablation:
    parameter: str
    value: object
    default_value: object
    explainers: Tuple[str, ...]
    temperature_updates: Optional[Dict[Type[nn.Module], float]] = None
    pga_updates: Dict[str, float] = field(default_factory=dict)
    is_default: bool = False


TAU_ABLATIONS = (
    ("tau_relu", (nn.ReLU,), (0.6, 0.3, 1.0, 0.4, 0.5)),
    ("tau_maxpool", (nn.MaxPool2d,), (0.3, 0.01, 0.5)),
    ("tau_gelu", (nn.GELU,), (1.0, 0.5, 2.0)),
    (
        "tau_attention",
        (nn.MultiheadAttention, PVTAttention),
        (1.0, 0.5, 5.0),
    ),
)

K_ABLATIONS = (5, 1, 2, 3, 10)
ALPHA_ABLATIONS = (20, 5, 10, 40)

DEFAULT_MODELS = {
    "resnet50": "torchvision",
    "vgg11_bn": "torchvision",
    "pvt_v2_b1": "timm",
}


def model_contains_any(model, module_classes):
    return any(isinstance(module, module_classes) for module in model.modules())


def build_ablations(model, default_temperatures):
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

    for value in K_ABLATIONS[1:]:
        ablations.append(
            Ablation(
                parameter="K",
                value=value,
                default_value=K_ABLATIONS[0],
                explainers=("PullbackAscent",),
                pga_updates={"steps": value},
            )
        )

    for value in ALPHA_ABLATIONS[1:]:
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


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def take_focus_batches(dataset, n_batches, batch_size, seed):
    class_to_indices = index_dataset_by_class(dataset)
    rng = np.random.default_rng(seed)
    return [
        build_focus_mosaics(dataset, class_to_indices, batch_size, rng)
        for _ in range(n_batches)
    ]


def evaluate_focus_batches(model, batches, explainers, metric, device):
    results = {"focus": {name: [] for name in explainers}}
    for batch_index, mosaics in enumerate(batches, start=1):
        print(f"Evaluating Focus batch {batch_index}/{len(batches)}")
        for name, (explain_func, explain_kwargs) in explainers.items():
            empty_cache()
            attributions = explain_func(
                model, mosaics.images, mosaics.targets,
                device=device, **explain_kwargs,
            )
            # Match evaluate_focus.py: Focus is evaluated on input * attribution.
            scores = metric(
                model=model,
                x_batch=mosaics.images,
                y_batch=mosaics.targets,
                a_batch=mosaics.images * attributions,
                custom_batch=mosaics.positions,
                channel_first=True,
                device=device,
            )
            results["focus"][name].extend(float(score) for score in scores)
    return results


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


def run_model(args):
    if args.n_batches < 1:
        raise ValueError("--n_batches must be at least 1.")

    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")

    seed_everything(args.seed)
    device, loader, model = setup_notebook(
        batch_size=args.batch_size,
        model_name=args.model_name,
        model_source=args.model_source,
        seed=args.seed,
        dataset="imagenet",
    )
    (
        default_temperatures,
        default_pga_kwargs_counterfactual,
        _,
        _,
    ) = get_default_kwargs()

    ablations = build_ablations(model, default_temperatures)
    ablations = select_ablations(ablations, args.only_ablation)
    # Build once so every ablation uses exactly the same images and positions.
    batches = take_focus_batches(
        loader.dataset, args.n_batches, args.batch_size, args.seed,
    )

    output_file = (
        f"{args.output_file}_model_name={args.model_name}"
        f"_n_batches={args.n_batches}_metric=focus"
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
        metric = quantus.Focus(
            abs=False,
            normalise=False,
            return_aggregate=False,
            disable_warnings=True,
        )
        results = evaluate_focus_batches(
            model, batches, explainers, metric, device,
        )
        all_results.append(format_results(results, ablation, args.model_name))

        # Checkpoint after every (potentially long) configuration.
        pd.concat(all_results, ignore_index=True).to_pickle(f"{output_file}.pkl")

    print(f"Results saved to {output_file}.pkl")


def main(args):
    if args.model_name == "all":
        if args.model_source is not None:
            raise ValueError("--model_source requires a specific --model_name.")
        models = DEFAULT_MODELS.items()
    else:
        source = args.model_source or DEFAULT_MODELS.get(args.model_name, "torchvision")
        models = [(args.model_name, source)]

    for model_name, model_source in models:
        print(f"Running Focus ablations for {model_name} ({model_source})")
        model_args = argparse.Namespace(**vars(args))
        model_args.model_name = model_name
        model_args.model_source = model_source
        run_model(model_args)
        # run_model has released its references before clearing the GPU cache.
        empty_cache()


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run Focus hyperparameter ablations. By default, sweep resnet50, "
            "vgg11_bn and pvt_v2_b1 with all applicable ablations."
        )
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="results/focus/ablations_focus",
        help="Output path without the .pkl extension or model name.",
    )
    parser.add_argument(
        "--model_name", type=str, default="all",
        help="Model name, or 'all' to sweep the three default architectures.",
    )
    parser.add_argument(
        "--model_source",
        type=str,
        default=None,
        choices=["torchvision", "timm"],
        help="Source for a single model; inferred for the default architectures.",
    )
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument(
        "--n_batches", type=int, default=10,
        help="Number of Focus mosaic batches (total mosaics = n_batches * batch_size).",
    )
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument(
        "--only_ablation",
        type=str,
        default=None,
        help=(
            "Run only one ablation. Accepts a 1-based index (e.g. 3), "
            "'default', or an exact label 'parameter=value' (e.g. K=1)."
        ),
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
