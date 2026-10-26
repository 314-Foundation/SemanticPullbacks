# Semantic Pullbacks

This repository contains the source code for the paper *Pulling Back the
Curtain on Deep Networks*.

The repository includes the Semantic Pullback attribution implementations,
model and dataset utilities, quantitative evaluation code, and notebooks for
reproducing the qualitative examples from the paper. The main implementation
is in `lib/`: attribution methods are defined in `lib/attributions.py`, the
surrogate backward rules in `lib/surrogates.py`, and the evaluation pipeline in
`lib/evaluator.py`. Loading, summarising, and comparing saved results is
implemented in `lib/results.py`.

## Installation

Install compatible versions of PyTorch (`torch`) and `torchvision` for your
platform, then install the remaining dependencies. Both PyTorch packages are
commented out in `requirements.txt` so that you can select the appropriate build:

```bash
pip install -r requirements.txt
```

## Data and execution

Run the commands below from the repository root.

- **ImageNet:** provide the validation images under `../data/imagenet/val`,
  with one subdirectory per class in ImageFolder format. The class ordering
  must match the ImageNet labels expected by the pretrained models.
- **Imagenette:** the loader downloads the validation dataset into `../data`
  when needed and maps its labels to ImageNet class IDs.

The code selects CUDA when available, then Apple MPS, otherwise CPU. Pretrained
model weights may be downloaded on first use.

| Architecture | `--model_name` | `--model_source` |
| --- | --- | --- |
| ResNet | `resnet50` | `torchvision` |
| VGG | `vgg11_bn` | `torchvision` |
| PVT | `pvt_v2_b1` | `timm` |

## Main quantitative evaluation

Evaluate 1000 ImageNet validation images (50 batches of 20):

```bash
python evaluate_quantus.py --output_file results/quantus --dataset imagenet --model_name resnet50 --model_source torchvision --batch_size 20 --n_batches 50
```

For a small smoke test on Imagenette:

```bash
python evaluate_quantus.py --output_file results/quantus_test --dataset imagenette --model_name resnet50 --batch_size 2 --n_batches 1 --test_mode
```

Without explicit batch settings, this script evaluates only 20 images
(`--batch_size 10 --n_batches 2`). `--test_mode` selects fewer explainers and
metrics; it does not change the number of examples.

The script computes attribution maps for the configured explainers, evaluates
the quantitative metrics used in the main comparison, and saves the results as
a pandas pickle file named `<output_file>_model_name=<model_name>.pkl`.
Pass `--output_file` without an extension and ensure its parent directory exists.

The notebooks in `notebooks/` provide smaller interactive examples, explainer
comparisons, counterfactuals, and feature accentuations. Figures generated from
these experiments are stored in `media/`. Use `notebooks/verify_results.ipynb`
to inspect the saved results and reproduce the summary tables.

## Additional experiments

### Hyperparameter ablations

Run one-at-a-time hyperparameter ablations on ImageNet with:

```bash
python evaluate_ablations.py --output_file results/quantus_ablations --model_name resnet50 --model_source torchvision --batch_size 20 --n_batches 50
```

The script evaluates `PullbackAscent` and `SoftPullback` for the temperature
ablations. The `K` and `alpha` ablations apply only to `PullbackAscent`. It also
runs one explicit default configuration for each explainer. Results are
checkpointed after every configuration in a single pickle file whose rows
identify the varied parameter, its value, and the explainer.

Use `--only_ablation default`, an exact label such as `--only_ablation K=1`,
or a 1-based index to run a single configuration. `--test_mode` reduces the
metric set. Without explicit batch settings, this script uses 10 batches of
20 images. Temperature ablations are skipped when the model has no matching
modules.

### Focus localisation

Evaluate localisation with Quantus Focus mosaics using:

```bash
python evaluate_focus.py --output_file results/focus/quantus_focus --dataset imagenet --n_mosaics 1000 --model_name resnet50 --model_source torchvision
```

`--n_mosaics` sets the total number of mosaics independently of the evaluation
batch size. For every mosaic, the script samples one target class and two
distractor classes that differ from the target, repeats the same target image
in two randomly chosen quadrants, and places one distractor image in each
remaining quadrant. Each source image is resized to `112x112` before the four
tiles are combined into a `224x224` model input.

Because the attribution maps represent signed changes in an input space
normalised to `[-1, 1]`, they are first converted into relevance maps using the
element-wise input-attribution product. The resulting relevance maps are passed
to the standard Quantus Focus metric with `abs=False` and `normalise=False`.
Focus retains only positive input-attribution relevance and measures the share
located in the two target quadrants, so the score remains in `[0, 1]`.

For a visual smoke test, run `notebooks/focus.ipynb`. It prints the target
classes, quadrant positions, and Focus scores, and displays the mosaics, target
masks, and relevance maps.

The script saves per-mosaic results to `<output_file>_model_name=<model_name>.pkl`
and a mean/std/count summary to the corresponding `_summary.csv` file.

### Focus hyperparameter ablations

Run the complete Focus sweep across ResNet, VGG, and PVT with:

```bash
python evaluate_ablations_focus.py
```

This runs all applicable temperature, `K`, and `alpha` ablations, plus the
default configuration, for each architecture. Defaults are 50 batches of 20
mosaics: **1000 mosaics per configuration**. Within each architecture, all
configurations reuse the same mosaics. Only Focus is evaluated.

Results are checkpointed after each configuration, separately for each model,
under `results/focus/`, using the filename pattern
`ablations_focus_model_name=<model_name>_n_batches=50_metric=focus.pkl`.
Use `--output_file` to change the output prefix.

For a two-mosaic test of one configuration on ResNet:

```bash
python evaluate_ablations_focus.py --output_file results/focus/ablations_test --model_name resnet50 --batch_size 2 --n_batches 1 --only_ablation default
```

Use `--only_ablation K=1` to test that ablation. Use different output prefixes
for separate runs that should be retained. `--model_name` restricts the sweep
to one architecture; its source is inferred for the three models listed above.
An explicit `--model_source` is supported only with a single model.

## Inspecting and comparing saved results

Run `notebooks/verify_results.ipynb` to create the comparison tables and paired
analyses from the saved files. Its input paths are explicit in the cells;
update them when analysing a new evaluation run.

The reusable helpers in `lib/results.py` include:

- `process_df`: select and rename metrics and explainers, and format means
  with standard deviations or bootstrap confidence intervals.
- `merge_with_priority`: replace a metric using regenerated results, as used
  for Faith.Corr in the PVT and ResNet comparisons.
- `add_focus_to_df`: add Focus summaries from per-mosaic results.
- `summarize_ablations_pickle`: export ablation summaries to `_summary.csv`.
- `paired_explainer_differences`: compute per-example pullback-minus-gradient
  differences for the four explainer pairs, with 95% percentile bootstrap CIs
  (10,000 resamples and seed 314 by default).

For example:

```python
from lib.results import paired_explainer_differences

paired, table, csv_path = paired_explainer_differences(
    "results/quantus_resnet_it_nips_20_50_resnet50.pkl",
    metric_overrides={
        "faithfulness_correlation": "results/quantus_resnet_fc_nips_20_50_resnet50.pkl",
    },
    show=False,
)
```

Paired comparisons require score arrays for the same examples in the same
order. Non-finite values are removed jointly within each pair. The function
writes `<input_stem>_paired_explainers.csv` and returns the detailed dataframe,
a display table, and the CSV path. `show=False` disables notebook display.

The notebook also writes `results/pvt_df_20_50.csv`,
`results/resnet_df_20_50.csv`, and `results/vgg_df_20_50.csv`, plus ablation
summaries and paired-comparison CSVs.

## Targeted accentuations

Use `notebooks/targeted_accentuations.ipynb` to compare accentuations across
PVT, ResNet, and VGG. Configure `METHOD`, `K`, `alpha`, `N_EXAMPLES`, and:

```python
TARGET_MODE = "fixed"       # Common target_class for all images.
# TARGET_MODE = "true_label"  # Each image's ground-truth dataset label.
target_class = 497

MODEL_LAYOUT = "columns"   # Original images, PVT, ResNet, VGG as columns.
# MODEL_LAYOUT = "rows"     # Original images and models as separate rows.
```

After changing the target mode or optimisation settings, rerun the configuration
and subsequent cells to recompute results. To change only the layout, update
`MODEL_LAYOUT` and rerun the final plotting cell; no model recomputation is needed.
Figures are saved under `media/targeted_accentuations/`, with the target and
layout recorded in the filename.
