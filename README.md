# Semantic Pullbacks

This repository contains the source code for the paper *Pulling Back the
Curtain on Deep Networks*. It provides a unifying view of neural computation
that connects gradient smoothing, B-cos-style alignment, feature accentuation,
robust optimisation, and normalisation effects. Based on this view, the paper
introduces Semantic Pullbacks as an explanation method for deep neural
networks.

The repository includes the Semantic Pullback attribution implementations,
model and dataset utilities, quantitative evaluation code, and notebooks for
reproducing the qualitative examples from the paper. The main implementation
is in `lib/`: attribution methods are defined in `lib/attributions.py`, the
surrogate backward rules in `lib/surrogates.py`, and the evaluation pipeline in
`lib/evaluator.py`.

## Installation

Install PyTorch for your platform and then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Main quantitative evaluation

Run the Quantus evaluation on ImageNet with:

```bash
python evaluate_quantus.py \
  --output_file results/quantus \
  --dataset imagenet \
  --model_name resnet50 \
  --model_source torchvision
```

The script computes attribution maps for the configured explainers, evaluates
the quantitative metrics used in the main comparison, and saves the results as
a pandas pickle file under the requested output path.

The notebooks in `notebooks/` provide smaller interactive examples, explainer
comparisons, counterfactuals, and feature accentuations. Figures generated from
these experiments are stored in `media/`. Use `notebooks/verify_results.ipynb`
to inspect the saved results and reproduce the summary tables.

## Additional experiments

### Hyperparameter ablations

Run one-at-a-time hyperparameter ablations on ImageNet with:

```bash
python evaluate_ablations.py \
  --output_file results/quantus_ablations \
  --model_name resnet50 \
  --model_source torchvision
```

The script evaluates `PullbackAscent` and `SoftPullback` for the temperature
ablations. The `K` and `alpha` ablations apply only to `PullbackAscent`. It also
runs one explicit default configuration for each explainer. Results are
checkpointed after every configuration in a single pickle file whose rows
identify the varied parameter, its value, and the explainer.

### Focus localisation

Evaluate localisation with Quantus Focus mosaics using:

```bash
python evaluate_focus.py \
  --output_file results/focus \
  --dataset imagenet \
  --n_mosaics 80 \
  --model_name resnet50 \
  --model_source torchvision
```

`--n_mosaics` sets the total number of mosaics independently of the evaluation
batch size. For every mosaic, the script samples one target class and two
distractor classes that differ from the target, repeats the same target image
in two randomly chosen quadrants, and places one distractor image in each
remaining quadrant.

For a visual smoke test, run `notebooks/focus.ipynb`. It prints the target
classes and quadrant positions, checks the Quantus scores against a direct
implementation of the Focus formula, and displays the mosaics, target masks,
and attribution maps.
