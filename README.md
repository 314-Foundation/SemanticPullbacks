# This repository contains the source code for the paper "Pulling Back the Curtain on Deep Networks"

We provide a unifying view on neural computation, bridging ideas like gradient smoothing, B-cos-style alignment, feature accentuation, robust optimization and normalization effects. We introduce Semantic Pullbacks as an effective explanation method for deep models.

## Installation

Install torch environment according to the `requirements.txt` file.

## Usage

Run the `evaluate_quantus.py` script to recreate the numerical results.

Run one-at-a-time hyperparameter ablations with:

```bash
python evaluate_ablations.py \
  --output_file results/quantus_ablations \
  --model_name resnet50 \
  --model_source torchvision
```

The script evaluates `PullbackAscent` and `SoftPullback` for temperature
ablations. The `K` and `alpha` ablations are evaluated only for
`PullbackAscent`. Results are checkpointed after every configuration in a
single pickle file whose rows identify the parameter, value, and explainer.

Play around with Semantic Pullbacks in juputer notebooks in the `notebooks` folder. This is how the images in `media` folder have been generated.

Read the results from the `results` folder using the `notebooks/verify_results` notebook.
