import argparse
import os

import pandas as pd
import torch

from lib.defaults import get_default_kwargs
from lib.evaluator import QuantusEvaluator, default_explainers, default_metrics
from lib.setup import setup_notebook
from lib.surrogates import soften_module_inplace_


def main(args):
    # Environmental settings for reproducibility
    # NOTE - this doesn't work for some architectures (e.g. VGG)
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True

    batch_size = args.batch_size
    device, loader, model = setup_notebook(
        batch_size=batch_size,
        model_name=args.model_name,
        model_source=args.model_source,
        seed=args.seed,
    )
    temperatures, pga_kwargs_counterfactual, pga_kwargs_grad, pga_kwargs_adv = (
        get_default_kwargs()
    )

    if args.model_name.startswith("resnet"):
        gc_layer = model[1].layer4[-1].conv3
    elif args.model_name.startswith("vgg"):
        gc_layer = model[1].avgpool
    elif args.model_name.startswith("pvt"):
        # gc_layer = model[1].stages[-1].blocks[-1].norm2  # rises error
        gc_layer = None

    soften_module_inplace_(
        model,
        temperatures=temperatures,
        standard_backward=True,
        fill_default_temperatures=True,
    )

    explainers_filter = None
    metrics_filter = None
    if args.test_mode:
        explainers_filter = ["SoftPullback", "Gradient"]
        metrics_filter = ["monotonicity_correlation", "faithfulness_correlation"]
    # explainers_filter = ["SoftPullback", "Gradient"]
    # metrics_filter = ["monotonicity_correlation", "faithfulness_correlation"]
    # metrics_filter = ["infidelity"]

    explainers = default_explainers(explainers_filter, gc_layer=gc_layer)
    metrics = default_metrics(metrics_filter)

    evaluator = QuantusEvaluator(
        model,
        metrics,
        explainers,
        device=device,
    )
    results = evaluator.evaluate_loader(loader, n_batches=args.n_batches)

    output_file = f"{args.output_file}_model_name={args.model_name}"
    df = evaluator.as_dataframe(results)
    evaluator.save_results(df, output_file)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the results CSV file. Don't include .pkl extension nor the model name.",
    )

    parser.add_argument("--model_name", type=str, required=False, default="resnet50")
    parser.add_argument(
        "--model_source",
        type=str,
        required=False,
        default="torchvision",
        choices=["torchvision", "timm"],
    )
    parser.add_argument("--batch_size", type=int, required=False, default=10)
    parser.add_argument("--n_batches", type=int, required=False, default=2)
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="If set, runs in test mode with fewer explainers and metrics for quick testing.",
    )
    parser.add_argument("--seed", type=int, required=False, default=314)
    args = parser.parse_args()
    main(args)
