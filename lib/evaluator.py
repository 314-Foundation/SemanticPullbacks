import gc
import time
from typing import Dict

import numpy as np
import pandas as pd
import quantus
import torch

from lib.attributions import (
    fusiongrad_explainer,
    quantus_double_pullback_ascent_diff_explain_func,
    quantus_gradient_ascent_diff_explain_func,
    quantus_pullback_ascent_diff_explain_func,
    smoothgrad_explainer,
)
from lib.defaults import get_default_kwargs
from lib.helpers import l2_normalize_batch_numpy
from lib.metrics import FaithfulnessCorrelationPatches, InfidelityOptimalScaling


def default_explainers(
    filter_explainers=None,
    gc_layer=None,
    pga_kwargs_counterfactual=None,
    reduce_axes=False,
):
    temperatures, default_pga_kwargs_counterfactual, pga_kwargs_grad, pga_kwargs_adv = (
        get_default_kwargs()
    )
    if pga_kwargs_counterfactual is None:
        pga_kwargs_counterfactual = default_pga_kwargs_counterfactual

    if reduce_axes:
        reduce_axes = (1,)
    else:
        reduce_axes = ()

    explainers = {
        "SoftPullback": (quantus_pullback_ascent_diff_explain_func, pga_kwargs_grad),
        # "SoftPullbackMulti": (
        #     quantus_pullback_ascent_diff_explain_func,
        #     {**pga_kwargs_grad, "steps": 5},
        # ),
        # "DoublePullback": (
        #     quantus_double_pullback_ascent_diff_explain_func,
        #     {
        #         "pga_kwargs_1": pga_kwargs_grad,
        #         "pga_kwargs_2": pga_kwargs_grad,
        #     },
        # ),
        # "DoublePullbackBis": (
        #     quantus_double_pullback_ascent_diff_explain_func,
        #     {
        #         "pga_kwargs_1": {
        #             **pga_kwargs_grad,
        #             "alpha": 2,
        #             "steps": 1,
        #             "normalize_step": False,
        #         },
        #         "pga_kwargs_2": pga_kwargs_grad,
        #     },
        # ),
        # "PullbackAscent": (
        #     quantus_pullback_ascent_diff_explain_func,
        #     pga_kwargs_counterfactual,
        # ),
        # "PullbackAscentBis": (
        #     quantus_pullback_ascent_diff_explain_func,
        #     {
        #         **pga_kwargs_counterfactual,
        #         "steps": 10,
        #     },
        # ),
        "PullbackAscent": (
            quantus_pullback_ascent_diff_explain_func,
            {
                **pga_kwargs_counterfactual,
                "clip_margin": None,
            },
        ),
        "PullbackAscentBis": (
            quantus_pullback_ascent_diff_explain_func,
            {
                **pga_kwargs_counterfactual,
                "clip_margin": None,
                "steps": 10,
            },
        ),
        "PullbackAscent3": (
            quantus_pullback_ascent_diff_explain_func,
            {
                **pga_kwargs_counterfactual,
                "clip_margin": None,
                "steps": 3,
            },
        ),
        "PullbackAscentClipLast": (
            quantus_pullback_ascent_diff_explain_func,
            {
                **pga_kwargs_counterfactual,
                "clip_last_only": True,
                # "clip_margin": 0.0,
                # "alpha": 20,
                # "clip_every_n_steps": 5,
                # "steps": 5,
            },
        ),
        # "PullbackAscentBisClipLast": (
        #     quantus_pullback_ascent_diff_explain_func,
        #     {
        #         **pga_kwargs_counterfactual,
        #         "clip_last_only": True,
        #         # "clip_last_only": Tre,
        #         # "clip_margin": 0.0,
        #         # "alpha": 20,
        #         # "clip_every_n_steps": 5,
        #         "steps": 10,
        #     },
        # ),
        # "GradientAscent": (
        #     quantus_gradient_ascent_diff_explain_func,
        #     {
        #         **pga_kwargs_counterfactual,
        #         # "clip_margin": None,
        #     },
        # ),
        "GradientAscent": (
            quantus_gradient_ascent_diff_explain_func,
            {
                **pga_kwargs_counterfactual,
                "clip_margin": None,
            },
        ),
        # "GradientAscentClipLast": (
        #     quantus_gradient_ascent_diff_explain_func,
        #     {
        #         **pga_kwargs_counterfactual,
        #         "clip_last_only": True,
        #     },
        # ),
        # "PullbackAscentBisNoClip": (
        #     quantus_pullback_ascent_diff_explain_func,
        #     {
        #         **pga_kwargs_counterfactual,
        #         "steps": 10,
        #         "clip_margin": None,
        #     },
        # ),
        # "PullbackAscentSmallAlpha": (
        #     quantus_pullback_ascent_diff_explain_func,
        #     {
        #         **pga_kwargs_counterfactual,
        #         "alpha": 5,
        #     },
        # ),
        # "PullbackAscentNoAlpha": (
        #     quantus_pullback_ascent_diff_explain_func,
        #     {
        #         **pga_kwargs_counterfactual,
        #         "alpha": None,
        #         "normalize_step": False,
        #     },
        # ),
        # "PullbackAscentSmallAlphaNoClip": (
        #     quantus_pullback_ascent_diff_explain_func,
        #     {
        #         **pga_kwargs_counterfactual,
        #         "alpha": 5,
        #         "clip_margin": None,
        #     },
        # ),
        "PullbackAscentNoAlpha": (
            quantus_pullback_ascent_diff_explain_func,
            {
                **pga_kwargs_counterfactual,
                "alpha": None,
                "normalize_step": False,
                "clip_margin": None,
                "steps": 5,
            },
        ),
        "SmoothPullback": (smoothgrad_explainer, {"use_pullback": True}),
        "FusionPullback": (fusiongrad_explainer, {"use_pullback": True}),
        "SmoothGrad": (smoothgrad_explainer, {}),
        "FusionGrad": (fusiongrad_explainer, {}),
        # "UnreducedGradient": (
        #     quantus.explain,
        #     {"method": "Gradient", "reduce_axes": ()},
        # ),
        "Gradient": (
            quantus.explain,
            {"method": "Gradient", "reduce_axes": reduce_axes},
        ),
        # "GradientOur": (quantus_gradient_ascent_diff_explain_func, pga_kwargs_grad),
        "GradientShap": (
            quantus.explain,
            {"method": "GradientShap", "reduce_axes": reduce_axes},
        ),
        "IntegratedGradients": (
            quantus.explain,
            {"method": "IntegratedGradients", "reduce_axes": reduce_axes},
        ),
        # "Saliency": (
        #     quantus.explain,
        #     {"method": "Saliency", "reduce_axes": reduce_axes},
        # ),
        "DeepLift": (
            quantus.explain,
            {"method": "DeepLift", "reduce_axes": reduce_axes},
        ),
        # "InputXGradient": (
        #     quantus.explain,
        #     {"method": "InputXGradient", "reduce_axes": reduce_axes},
        # ),
        # "Deconvolution": (
        #     quantus.explain,
        #     {"method": "Deconvolution", "reduce_axes": reduce_axes},
        # ),
        # "Lime": (quantus.explain, {"method": "Lime"}),  # way slower than others
        # # "Occlusion": (quantus.explain, {"method": "Occlusion"}),  # very, very slow
        # "KernelShap": (quantus.explain, {"method": "KernelShap"}),  # slow and bad
        # "LRP": (
        #     quantus.explain,
        #     {"method": "LRP", "reduce_axes": reduce_axes},
        # ),  # requires additional custom rules for many layers (even for nn.LayerNorm)
        # "DeepLiftShap": (
        #     quantus.explain,
        #     {"method": "DeepLiftShap"},
        # ),  # slow, similar to DeepLift
        # "FeatureAblation": (
        #     quantus.explain,
        #     {"method": "FeatureAblation"},
        # ),  # very slow
        # "FeaturePermutation": (
        #     quantus.explain,
        #     {"method": "FeaturePermutation"},
        # ),  # very slow
    }
    if gc_layer is not None:
        explainers.update(
            {
                "GuidedGradCam": (
                    quantus.explain,
                    {
                        "method": "GuidedGradCam",
                        "gc_layer": gc_layer,
                        "reduce_axes": reduce_axes,
                    },
                ),
                # "InternalInfluence": (  # CUDA runs out of memory even for small batches
                #     quantus.explain,
                #     {"method": "InternalInfluence", "gc_layer": gc_layer},
                # ),
                # {
                #     "LayerGradCam": (  # This is just for a single layer
                #         quantus.explain,
                #         {"method": "LayerGradCam", "gc_layer": gc_layer},
                #     ),
                # },
            }
        )

    if filter_explainers is not None:
        explainers = {
            name: explainers[name] for name in filter_explainers if name in explainers
        }

    return explainers


def default_metrics(filter_metrics=None):
    metrics = {
        # "infidelity": quantus.Infidelity(
        "infidelity": InfidelityOptimalScaling(
            perturb_baseline="uniform",
            # perturb_baseline=0.0,
            # perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
            perturb_func_kwargs={"uniform_low": -1.0, "uniform_high": 1.0},
            n_perturb_samples=5,
            # n_perturb_samples=10,
            # perturb_patch_sizes=[56],
            perturb_patch_sizes=[112, 56],  # , 28],
            display_progressbar=True,
            # loss_func=mse_optimal_scaling,
            # normalise=True,
            # normalise_func=l2_normalize_batch_numpy,
            # normalise_func=normalise_by_average_second_moment_estimate,
        ),
        # "faithfulness_correlation": quantus.FaithfulnessCorrelation(
        "faithfulness_correlation": FaithfulnessCorrelationPatches(
            # nr_runs=100,
            # nr_runs=50,
            # subset_size=12544,
            # subset_size=12544 // 4,
            # subset_size=224,
            return_aggregate=False,
            # perturb_baseline="uniform",
            perturb_baseline=0.0,
            # perturb_baseline=-1.0,
            # normalise=False,
            # perturb_patch_sizes=[112, 56, 28],
            perturb_patch_sizes=[56],
            abs=True,
            # abs=False,
            # normalise_func=l2_normalize_batch_numpy,
        ),
        "faithfulness_estimate": quantus.FaithfulnessEstimate(
            # perturb_func=qua ntus.perturb_func.baseline_replacement_by_indices,
            # similarity_func=quantus.similarity_func.correlation_pearson,
            features_in_step=12544 // 4,
            # features_in_step=12544,
            # features_in_step=224,
            # perturb_baseline="black",
            perturb_baseline=0.0,
            # abs=False,
            abs=True,
            # normalise=False,
        ),
        "monotonicity_correlation": quantus.MonotonicityCorrelation(
            nr_samples=10,
            # features_in_step=12544,  # 224*224 / 4
            features_in_step=12544 // 4,
            # features_in_step=224,
            perturb_baseline="uniform",
            perturb_func_kwargs={"uniform_low": -1.0, "uniform_high": 1.0},
            # perturb_baseline=0.0,
            abs=True,
            eps=1e-12,
            # normalise=True,
            # normalise_func=l2_normalize_batch_numpy,
            # perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
            # similarity_func=quantus.similarity_func.correlation_spearman,
        ),
        # "pixel_flipping": quantus.PixelFlipping(
        #     features_in_step=12544 // 4,
        #     # features_in_step=224,
        #     perturb_baseline="black",
        #     # perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        # ),
        # TODO: play around with these metrics later, understand them better
        # "region_perturbation": quantus.RegionPerturbation(
        #     patch_size=14,
        #     regions_evaluation=50,
        #     perturb_baseline="black",
        #     normalise=True,
        # ),
        # "selectivity": quantus.Selectivity(
        #     patch_size=56,
        #     perturb_baseline="black",
        # ),
        # "sensitivity_n": quantus.SensitivityN(
        #     # features_in_step=224,
        #     features_in_step=12544 // 4,
        #     n_max_percentage=0.8,
        #     # similarity_func=quantus.similarity_func.correlation_pearson,
        #     # perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
        #     perturb_baseline="black",
        #     return_aggregate=False,
        #     # abs=True,
        # ),
        # TODO: fix these metrics as they cause errors
        # "IROF": quantus.IROF(),
        # "road": quantus.ROAD(),
        # "sufficiency": quantus.Sufficiency(
        #     threshold=0.6,
        #     return_aggregate=False,
        # ),
        # "avg_sensitivity": quantus.AvgSensitivity(
        #     nr_samples=10,
        #     # lower_bound=0.2,
        #     lower_bound=0.02,
        #     # abs=True,
        #     # normalise=True,
        #     norm_numerator=quantus.norm_func.fro_norm,
        #     norm_denominator=quantus.norm_func.fro_norm,
        #     # perturb_func=quantus.perturb_func.uniform_noise,
        #     similarity_func=quantus.similarity_func.difference,
        # ),
        "max_sensitivity": quantus.MaxSensitivity(
            nr_samples=10,
            lower_bound=0.02,
            # normalise=True,
            # normalise_func=l2_normalize_batch_numpy,
            # norm_numerator=quantus.norm_func.fro_norm,
            # norm_denominator=quantus.norm_func.fro_norm,
            # perturb_func=quantus.perturb_func.uniform_noise,
            # similarity_func=quantus.similarity_func.difference,
        ),
        # "sparseness": quantus.Sparseness(
        #     # abs=False,
        #     # normalise=False,
        #     # normalise=True,
        # ),
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
    # "max_sensitivity"  # Very Good
    # "sparseness"  # Good
    # "random_logit"  # Similar to other explainers


class QuantusEvaluator:
    def __init__(self, model, metrics, explainers, device, random_labels=False):
        self.model = model
        self.device = device
        self.metrics = metrics
        self.explainers = explainers
        self.random_labels = random_labels

    def precompute_attributions(self, x_batch, y_batch):
        self.attributions = {}

        if isinstance(x_batch, torch.Tensor):
            x_batch = x_batch.detach().cpu().numpy()
        if isinstance(y_batch, torch.Tensor):
            y_batch = y_batch.detach().cpu().numpy()

        for explainer_name, (
            explain_func,
            explain_func_kwargs,
        ) in self.explainers.items():
            self.empty_cache()

            print(f"Precomputing attributions for explainer {explainer_name}")
            start_time = time.time()
            self.attributions[explainer_name] = explain_func(
                self.model, x_batch, y_batch, device=self.device, **explain_func_kwargs
            )
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"Explainer {explainer_name} took {elapsed:.3f} seconds.")

    def evaluate_metric(self, metric_name, explainer_name, x_batch, y_batch) -> list:
        metric = self.metrics[metric_name]
        explain_func, explain_func_kwargs = self.explainers[explainer_name]
        a_batch = self.attributions[explainer_name]

        if isinstance(x_batch, torch.Tensor):
            x_batch = x_batch.detach().cpu().numpy()
        if isinstance(y_batch, torch.Tensor):
            y_batch = y_batch.detach().cpu().numpy()
        if isinstance(a_batch, torch.Tensor):
            a_batch = a_batch.detach().cpu().numpy()

        print(f"Evaluating metric {metric_name} with explainer {explainer_name}")

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

                scores = self.evaluate_metric(
                    metric_name, explainer_name, x_batch, y_batch
                )
                results[metric_name][explainer_name] = scores

        self.empty_cache()

        return results

    def evaluate_loader(self, data_loader, n_batches=1) -> Dict[str, Dict[str, list]]:
        all_results = {}
        for batch_idx, (x_batch, y_batch) in enumerate(data_loader):

            print(f"Evaluating batch {batch_idx + 1}/{n_batches}")

            x_batch = x_batch.numpy()
            y_batch = y_batch.numpy()
            if self.random_labels:
                y_batch = np.random.randint(
                    0, 1000, size=y_batch.shape, dtype=y_batch.dtype
                )

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
    def summarize_results(df: pd.DataFrame, quantile=0.0, precision=3) -> pd.DataFrame:
        def format_mean_std(cell):
            arr = np.array(cell)
            if len(arr) == 0:
                return ""

            arr = np.ma.masked_invalid(arr)

            if quantile > 0:
                lower = np.quantile(arr, quantile)
                upper = np.quantile(arr, 1 - quantile)

                arr = np.clip(arr, lower, upper)

            mean = arr.mean()
            std = arr.std()

            def format_val(x, precision=2):
                if np.isnan(x):
                    return ""
                # Format in scientific notation if very large or very small
                if abs(x) >= 1e4 or (abs(x) > 0 and abs(x) < 1e-3):
                    return f"{x:.2e}"
                else:
                    s = f"{x:.{precision}f}"
                    return s.rstrip("0").rstrip(".") if "." in s else s

            mean_str = format_val(mean, precision=precision)
            std_str = format_val(std, precision=precision - 1)
            return f"{mean_str}\u00b1{std_str}"

        return df.map(format_mean_std)
