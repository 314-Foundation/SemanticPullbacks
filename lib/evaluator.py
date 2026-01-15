import gc
from typing import Dict

import numpy as np
import pandas as pd
import quantus
import torch

from lib.attributions import (
    quantus_double_pullback_ascent_diff_explain_func,
    quantus_pullback_ascent_diff_explain_func,
)
from lib.defaults import get_default_kwargs


def default_explainers(filter_explainers=None):
    temperatures, pga_kwargs_counterfactual, pga_kwargs_grad, pga_kwargs_adv = (
        get_default_kwargs()
    )
    explainers = {
        "SoftPullback": (quantus_pullback_ascent_diff_explain_func, pga_kwargs_grad),
        "DoublePullback": (
            quantus_double_pullback_ascent_diff_explain_func,
            {
                "pga_kwargs_1": pga_kwargs_grad,
                "pga_kwargs_2": pga_kwargs_grad,
            },
        ),
        "DoublePullbackBis": (
            quantus_double_pullback_ascent_diff_explain_func,
            {
                "pga_kwargs_1": {**pga_kwargs_grad, "alpha": 2},
                "pga_kwargs_2": pga_kwargs_grad,
            },
        ),
        "Gradient": (quantus.explain, {"method": "Gradient"}),
        "GradientShap": (quantus.explain, {"method": "GradientShap"}),
        "IntegratedGradients": (quantus.explain, {"method": "IntegratedGradients"}),
        "Saliency": (quantus.explain, {"method": "Saliency"}),
        "DeepLift": (quantus.explain, {"method": "DeepLift"}),
        "InputXGradient": (quantus.explain, {"method": "InputXGradient"}),
        "Deconvolution": (quantus.explain, {"method": "Deconvolution"}),
        "GuidedGradCam": (quantus.explain, {"method": "GuidedGradCam"}),
    }

    if filter_explainers is not None:
        explainers = {
            name: explainers[name] for name in filter_explainers if name in explainers
        }

    return explainers


def default_metrics(filter_metrics=None):
    metrics = {
        "faithfulness_correlation": quantus.FaithfulnessCorrelation(
            nr_runs=100,
            subset_size=12544,
            return_aggregate=False,
            # perturb_baseline="uniform",
        ),
        "monotonicity_correlation": quantus.MonotonicityCorrelation(
            nr_samples=10,
            features_in_step=12544,  # 224*224 / 4
            # perturb_baseline="uniform",
            # abs=False,
            # normalise=False,
            # perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
            # similarity_func=quantus.similarity_func.correlation_spearman,
        ),
        "faithfulness_estimate": quantus.FaithfulnessEstimate(
            # perturb_func=qua ntus.perturb_func.baseline_replacement_by_indices,
            # similarity_func=quantus.similarity_func.correlation_pearson,
            features_in_step=12544 // 4,
            # features_in_step=224,
            # perturb_baseline="black",
            # abs=False,
            # normalise=False,
        ),
        "pixel_flipping": quantus.PixelFlipping(
            features_in_step=12544 // 4,
            # features_in_step=224,
            perturb_baseline="black",
            # perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        ),
        "infidelity": quantus.Infidelity(
            perturb_baseline="uniform",
            # perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
            n_perturb_samples=5,
            perturb_patch_sizes=[56],
            display_progressbar=True,
        ),
        "avg_sensitivity": quantus.AvgSensitivity(
            nr_samples=10,
            # lower_bound=0.2,
            lower_bound=0.02,
            # abs=True,
            # normalise=True,
            norm_numerator=quantus.norm_func.fro_norm,
            norm_denominator=quantus.norm_func.fro_norm,
            # perturb_func=quantus.perturb_func.uniform_noise,
            similarity_func=quantus.similarity_func.difference,
        ),
        "max_sensitivity": quantus.MaxSensitivity(
            nr_samples=10,
            lower_bound=0.02,
            # norm_numerator=quantus.norm_func.fro_norm,
            # norm_denominator=quantus.norm_func.fro_norm,
            # perturb_func=quantus.perturb_func.uniform_noise,
            # similarity_func=quantus.similarity_func.difference,
        ),
        "sparseness": quantus.Sparseness(
            # abs=False,
            # normalise=False,
            # normalise=True,
        ),
        "random_logit": quantus.RandomLogit(
            num_classes=1000,
            # abs=True,
            # seed=42,
            # normalise=False,
            # similarity_func=cosine,
            # similarity_func=quantus.similarity_func.ssim,
            similarity_func=quantus.similarity_func.correlation_pearson,
        ),
    }

    if filter_metrics is not None:
        metrics = {name: metrics[name] for name in filter_metrics if name in metrics}

    return metrics
    # "faithfulness_correlation"  # Medium, Unstable
    # "monotonicity_correlation"  # Very Good
    # "faithfulness_estimate"  # Good, Slow
    # "pixel_flipping"  # Good
    # "infidelity"
    # "avg_sensitivity"  # Very Good
    # "max_sensitivity"
    # "sparseness"
    # "random_logit"


class QuantusEvaluator:
    def __init__(self, model, metrics, explainers, device):
        self.model = model
        self.device = device
        self.metrics = metrics
        self.explainers = explainers

    def precompute_attributions(self, x_batch, y_batch):
        self.attributions = {}
        for explainer_name, (
            explain_func,
            explain_func_kwargs,
        ) in self.explainers.items():
            self.empty_cache()

            self.attributions[explainer_name] = explain_func(
                self.model, x_batch, y_batch, device=self.device, **explain_func_kwargs
            )

    def evaluate_metric(self, metric, explainer_name, x_batch, y_batch) -> list:
        explain_func, explain_func_kwargs = self.explainers[explainer_name]
        a_batch = self.attributions[explainer_name]

        if type(x_batch) is torch.Tensor:
            x_batch = x_batch.cpu().numpy()
        if type(y_batch) is torch.Tensor:
            y_batch = y_batch.cpu().numpy()
        if type(a_batch) is torch.Tensor:
            a_batch = a_batch.cpu().numpy()

        print(
            f"Evaluating metric {metric.__class__.__name__} with explainer {explainer_name}"
        )

        scores = metric(
            model=self.model,
            x_batch=x_batch,
            y_batch=y_batch,
            a_batch=a_batch,
            device=self.device,
            explain_func=explain_func,
            explain_func_kwargs={**explain_func_kwargs},  # metrics may modify kwargs
        )
        # scores = torch.as_tensor(scores).float()
        return scores

    def evaluate_batch(
        self, x_batch, y_batch, precompute_attributions=True
    ) -> Dict[str, Dict[str, list]]:
        results = {}
        if precompute_attributions:
            self.precompute_attributions(x_batch, y_batch)

        for metric_name, metric in self.metrics.items():
            results[metric_name] = {}
            for explainer_name in self.explainers.keys():
                self.empty_cache()

                scores = self.evaluate_metric(metric, explainer_name, x_batch, y_batch)
                results[metric_name][explainer_name] = scores

        self.empty_cache()

        return results

    def evaluate_loader(self, data_loader, n_batches=1) -> Dict[str, Dict[str, list]]:
        all_results = {}
        for batch_idx, (x_batch, y_batch) in enumerate(data_loader):

            print(f"Evaluating batch {batch_idx + 1}/{n_batches}")

            x_batch = x_batch.numpy()
            y_batch = y_batch.numpy()

            batch_results = self.evaluate_batch(
                x_batch, y_batch, precompute_attributions=True
            )

            for metric_name, explainer_results in batch_results.items():
                if metric_name not in all_results:
                    all_results[metric_name] = {}
                for explainer_name, scores in explainer_results.items():
                    if explainer_name not in all_results[metric_name]:
                        all_results[metric_name][explainer_name] = []
                    all_results[metric_name][explainer_name].extend(scores)

            if batch_idx + 1 >= n_batches:
                break

        return all_results

    @staticmethod
    def empty_cache():
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def as_dataframe(all_results: Dict[str, Dict[str, list]]) -> pd.DataFrame:
        return pd.DataFrame.from_dict(all_results)

    @staticmethod
    def save_results(df: pd.DataFrame, filename: str):
        if isinstance(df, dict):
            df = QuantusEvaluator.as_dataframe(df)
        df.to_pickle(f"{filename}.pkl")

    @staticmethod
    def load_results(filename: str) -> pd.DataFrame:
        return pd.read_pickle(f"{filename}.pkl")

    @staticmethod
    def summarize_results(df: pd.DataFrame) -> pd.DataFrame:

        def format_mean_std(cell):
            arr = np.array(cell)
            if len(arr) == 0:
                return ""

            def trim(x):
                s = f"{x:.3f}"
                return s.rstrip("0").rstrip(".") if "." in s else s

            mean = trim(arr.mean())
            std = trim(arr.std())
            return f"{mean}\u00b1{std}"

        return df.map(format_mean_std)
        # return df.applymap(format_mean_std)
