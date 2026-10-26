"""Loading, summarizing, and comparing saved per-example explainer scores.

Only NumPy and pandas are needed for computation; IPython is imported only
when displaying the summary. Input arrays must share the same example order.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SHORT_METRICS_MAP = {
    "infidelity": "Infidelity",
    "faithfulness_correlation": "Faith.Corr",
    "monotonicity_correlation": "Mono.Corr",
    "max_sensitivity": "Max.Sens",
    "random_logit": "Rand.Logit",
}


def _bootstrap_mean_ci(
    values,
    rng,
    n_resamples=10_000,
    confidence_level=0.95,
    chunk_size=1_000,
):
    """Return a percentile bootstrap CI for the mean of paired differences."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return np.nan, np.nan
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1.")

    bootstrap_means = np.empty(n_resamples, dtype=float)
    for start in range(0, n_resamples, chunk_size):
        stop = min(start + chunk_size, n_resamples)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        bootstrap_means[start:stop] = values[indices].mean(axis=1)

    alpha = (1 - confidence_level) / 2
    return tuple(np.quantile(bootstrap_means, [alpha, 1 - alpha]))


EXPLAINER_PAIRS = (
    ("SoftPullback", "Gradient", "SoftPullback - Gradient"),
    ("PullbackAscent", "GradientAscent", "PullbackAscent - GradientAscent"),
    ("SmoothPullback", "SmoothGrad", "SmoothPullback - SmoothGrad"),
    ("FusionPullback", "FusionGrad", "FusionPullback - FusionGrad"),
)


def paired_explainer_differences(
    results_pkl_path,
    metric_overrides=None,
    pairs=EXPLAINER_PAIRS,
    show=True,
    bootstrap_resamples=10_000,
    bootstrap_seed=314,
):
    """Compute paired Pullback-minus-Gradient differences and bootstrap CIs.

    Read a pickle with explainers as the index, metrics as columns, and
    per-example score arrays as cells. Arrays are paired by position, so both
    explainers must use the same examples in the same order. Non-finite scores
    are removed jointly from each pair. ``metric_overrides`` maps metric names
    to alternative result pickles in the same format.

    Save ``<stem>_paired_explainers.csv`` beside the base pickle and return
    ``(detailed_dataframe, display_dataframe, csv_path)``. Set ``show=False``
    to run without IPython. The seeded percentile bootstrap uses 95% CIs.
    """
    results_pkl_path = Path(results_pkl_path)
    if not results_pkl_path.exists():
        raise FileNotFoundError(f"File not found: {results_pkl_path}")

    base_df = load_results_pickle(results_pkl_path)
    metric_frames = {metric: base_df for metric in base_df.columns}
    metric_sources = {metric: results_pkl_path for metric in base_df.columns}

    for metric, override_path in (metric_overrides or {}).items():
        override_path = Path(override_path)
        if not override_path.exists():
            raise FileNotFoundError(f"File not found: {override_path}")
        override_df = load_results_pickle(override_path)
        if metric not in override_df.columns:
            raise ValueError(f"Metric {metric!r} is missing from {override_path}.")
        metric_frames[metric] = override_df
        metric_sources[metric] = override_path

    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    for pullback_name, gradient_name, pair_label in pairs:
        for metric, metric_df in metric_frames.items():
            missing = {pullback_name, gradient_name}.difference(metric_df.index)
            if missing:
                raise ValueError(
                    f"Missing explainers {sorted(missing)} for metric={metric!r} "
                    f"in {metric_sources[metric]}."
                )

            pullback_values = np.asarray(
                metric_df.at[pullback_name, metric], dtype=float
            ).reshape(-1)
            gradient_values = np.asarray(
                metric_df.at[gradient_name, metric], dtype=float
            ).reshape(-1)
            if pullback_values.shape != gradient_values.shape:
                raise ValueError(
                    f"Cannot pair {pair_label}, metric={metric!r}: Pullback has "
                    f"{pullback_values.size} values and Gradient has "
                    f"{gradient_values.size}."
                )

            valid = np.isfinite(pullback_values) & np.isfinite(gradient_values)
            paired_pullback = pullback_values[valid]
            paired_gradient = gradient_values[valid]
            differences = paired_pullback - paired_gradient
            if differences.size == 0:
                continue

            ci95_low, ci95_high = _bootstrap_mean_ci(
                differences,
                rng=rng,
                n_resamples=bootstrap_resamples,
            )
            rows.append({
                "result_set": results_pkl_path.stem,
                "pair": pair_label,
                "pullback_explainer": pullback_name,
                "gradient_explainer": gradient_name,
                "metric": metric,
                "metric_source": metric_sources[metric].name,
                "pullback_mean": float(paired_pullback.mean()),
                "gradient_mean": float(paired_gradient.mean()),
                "delta_mean": float(differences.mean()),
                "delta_std": (
                    float(differences.std(ddof=1))
                    if differences.size > 1 else np.nan
                ),
                "ci95_low": float(ci95_low),
                "ci95_high": float(ci95_high),
                "paired_count": int(differences.size),
            })

    paired_df = pd.DataFrame(rows).sort_values(["pair", "metric"]).reset_index(drop=True)
    paired_csv_path = results_pkl_path.with_name(
        results_pkl_path.stem + "_paired_explainers.csv"
    )
    paired_df.to_csv(paired_csv_path, index=False)

    metric_directions = {
        "infidelity": "↓",
        "faithfulness_correlation": "↑",
        "monotonicity_correlation": "↑",
        "max_sensitivity": "↓",
        "random_logit": "↓",
    }
    metric_labels = {
        metric: f"{SHORT_METRICS_MAP.get(metric, metric)} {metric_directions.get(metric, '')}".rstrip()
        for metric in paired_df["metric"].unique()
    }
    paired_display = paired_df.copy()
    paired_display["metric"] = paired_display["metric"].map(metric_labels)
    paired_display["delta_ci95"] = paired_display.apply(
        lambda row: (
            f"{row['delta_mean']:.4f} "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
        ),
        axis=1,
    )
    paired_pivot = paired_display.pivot(
        index="pair",
        columns="metric",
        values="delta_ci95",
    ).reset_index()
    paired_pivot.columns.name = None

    print(f"Loaded base results: {results_pkl_path}")
    for metric, source_path in metric_sources.items():
        print(f"Metric source [{metric}]: {source_path}")
    print(f"Saved paired explainer CSV: {paired_csv_path}")
    print(
        f"Delta format: Pullback - Gradient, mean [95% percentile bootstrap CI] "
        f"using {bootstrap_resamples:,} resamples (seed={bootstrap_seed})."
    )
    print(f"Paired counts: {sorted(paired_df['paired_count'].unique())}")
    if show:
        from IPython.display import display

        display(paired_pivot)
    return paired_df, paired_pivot, paired_csv_path


short_titles_map = {
    "Image": "Image",
    "Gradient": "Grad",
    "SoftPullback": "SoftPb",
    "GradientAscent": "GA",
    "PullbackAscent": "PA",
    "SmoothGrad": "SmoothG",
    "SmoothPullback": "SmoothPb",
    "FusionGrad": "FusionG",
    "FusionPullback": "FusionPb",
    "IntegratedGradients": "IG",
    "GradientShap": "GradShap",
    "DeepLift": "DeepLift",
}

DEFAULT_EXPLAINERS = (
    "Gradient", "SoftPullback", "GradientAscent", "PullbackAscent",
    "SmoothGrad", "SmoothPullback", "FusionGrad", "FusionPullback",
    "IntegratedGradients", "GradientShap", "DeepLift",
)

DEFAULT_METRICS = SHORT_METRICS_MAP


def load_results_pickle(path):
    """Read saved results, including NumPy 2 pickles under NumPy 1."""
    import sys

    if int(np.__version__.split(".", maxsplit=1)[0]) < 2:
        import numpy.core as core
        import numpy.core.multiarray as multiarray
        import numpy.core.numeric as numeric

        sys.modules.setdefault("numpy._core", core)
        sys.modules.setdefault("numpy._core.multiarray", multiarray)
        sys.modules.setdefault("numpy._core.numeric", numeric)
    return pd.read_pickle(path)


def _as_float_array(value):
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        arr = np.asarray(value, dtype=float).reshape(-1)
    else:
        arr = np.asarray([value], dtype=float)
    return arr[np.isfinite(arr)]


def _format_val(x, precision=3):
    if np.isnan(x):
        return ""
    if abs(x) >= 1e4 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.2e}"
    s = f"{x:.{precision}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def _format_mean_std(arr, precision=3):
    if arr.size == 0:
        return ""
    mean = arr.mean()
    std = arr.std()
    mean_str = _format_val(mean, precision=precision)
    std_str = _format_val(std, precision=max(1, precision - 1))
    return f"{mean_str}±{std_str}"


def summarize_ablations_pickle(pkl_path, show=True, *, metric_names=None):
    short_metrics_map = DEFAULT_METRICS if metric_names is None else metric_names
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"File not found: {pkl_path}")

    df = load_results_pickle(pkl_path)
    metric_cols = [col for col in short_metrics_map if col in df.columns]
    group_cols = ["model_name", "parameter", "value", "default_value", "explainer"]

    summary_rows = []
    for _, row in df.iterrows():
        base = {col: row[col] for col in group_cols}
        for metric in metric_cols:
            arr = _as_float_array(row[metric])
            base[short_metrics_map[metric]] = _format_mean_std(arr, precision=3)
        summary_rows.append(base)

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["model_name", "parameter", "value", "explainer"])
        .reset_index(drop=True)
    )

    csv_path = pkl_path.with_name(pkl_path.stem + "_summary.csv")
    summary.to_csv(csv_path, index=False)

    print(f"Loaded: {pkl_path}")
    print(f"Saved CSV: {csv_path}")
    if show:
        from IPython.display import display

        display(summary)
    return summary, csv_path


def merge_with_priority(df_big, df_small, key_col="Faith.Corr"):
    # we merge results as we re-genenerated the faithfulness_correlation metric for different set of patches
    # the previous selection had too large patches, which inflated the correlation score
    result = df_big.loc[df_small.index].copy()
    result[key_col] = df_small[key_col]
    return result


def process_df(
    df_path,
    precision=3,
    uncertainty="std",
    confidence_level=0.95,
    bootstrap_resamples=10_000,
    bootstrap_seed=314,
    *,
    metric_names=None,
    explainers=DEFAULT_EXPLAINERS,
    explainer_names=None,
):
    """Use uncertainty='bootstrap_ci' to report mean [95% CI] instead of mean±std."""
    short_metrics_map = DEFAULT_METRICS if metric_names is None else metric_names
    names = short_titles_map if explainer_names is None else explainer_names
    selected_columns = list(short_metrics_map.keys())
    
    path = Path(df_path)
    if path.suffix != ".pkl":
        path = Path(f"{path}.pkl")
    df = load_results_pickle(path)
    df = df[
        [col for col in selected_columns if col in df.columns]
    ].rename(columns=short_metrics_map)
    df = df.loc[[idx for idx in explainers if idx in df.index]]
    df = df.rename(index=names)
    
    print("num_samples:", len(df.iloc[0].iloc[0]))
    
    df = summarize_results(
        df,
        precision=precision,
        uncertainty=uncertainty,
        confidence_level=confidence_level,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    return df


def add_focus_to_df(df, focus_pkl_path):
    focus_df = load_results_pickle(focus_pkl_path)

    focus_summary = focus_df.groupby("explainer")["score"].apply(
        lambda scores: (
            f"{scores.mean():.3f}±{scores.std():.2f}"
            if len(scores) > 1 else f"{scores.mean():.2f}±nan"
        )
    )
    focus_summary.index = focus_summary.index.map(short_titles_map)

    result = df.drop(columns="Focus", errors="ignore").copy()
    result.insert(1, "Focus", focus_summary.reindex(result.index))
    return result


def summarize_results(
    df: pd.DataFrame,
    quantile=0.0,
    precision=3,
    std_precision=2,
    uncertainty="std",
    confidence_level=0.95,
    bootstrap_resamples=10_000,
    bootstrap_seed=314,
) -> pd.DataFrame:
    """Format cell-wise means with either std or a bootstrap CI of the mean."""
    if uncertainty not in {"std", "bootstrap_ci"}:
        raise ValueError("uncertainty must be either 'std' or 'bootstrap_ci'.")
    if not 0 <= quantile < 0.5:
        raise ValueError("quantile must be in [0, 0.5).")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be at least 1.")

    rng = np.random.default_rng(bootstrap_seed)

    def format_val(x, value_precision=2):
        if np.isnan(x):
            return ""
        value = f"{x:.{value_precision}f}"
        return value
        # return value.rstrip("0").rstrip(".") if "." in value else value

    def format_summary(cell):
        arr = np.asarray(cell, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return ""

        if quantile > 0:
            lower = np.quantile(arr, quantile)
            upper = np.quantile(arr, 1 - quantile)
            arr = np.clip(arr, lower, upper)

        mean_str = format_val(arr.mean(), value_precision=precision)
        if uncertainty == "std":
            std_str = format_val(arr.std(), value_precision=std_precision)
            return f"{mean_str}\u00b1{std_str}"

        ci_low, ci_high = _bootstrap_mean_ci(
            arr, rng, n_resamples=bootstrap_resamples,
            confidence_level=confidence_level,
        )
        ci_low_str = format_val(ci_low, value_precision=std_precision)
        ci_high_str = format_val(ci_high, value_precision=std_precision)
        return f"{mean_str} [{ci_low_str}, {ci_high_str}]"

    return df.map(format_summary)
