import argparse
import gc
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import quantus
import torch
from quantus.functions.mosaic_func import build_single_mosaic

from lib.defaults import get_default_kwargs
from lib.evaluator import default_explainers
from lib.setup import setup_notebook
from lib.surrogates import soften_module_inplace_

EXPLAINER_NAMES = (
    "SoftPullback",
    "PullbackAscent",
    "SmoothPullback",
    "FusionPullback",
    "Gradient",
    "GradientAscent",
    "SmoothGrad",
    "FusionGrad",
    "GradientShap",
    "IntegratedGradients",
    "DeepLift",
    "GuidedGradCam",
)
QUADRANT_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")


@dataclass
class FocusMosaics:
    images: np.ndarray
    targets: np.ndarray
    positions: np.ndarray
    source_indices: list
    component_labels: list


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def dataset_targets(dataset):
    if hasattr(dataset, "targets"):
        targets = dataset.targets
    elif hasattr(dataset, "samples"):
        targets = [sample[1] for sample in dataset.samples]
    elif hasattr(dataset, "_samples"):
        targets = [sample[1] for sample in dataset._samples]
    else:
        raise TypeError(
            "The dataset must expose targets, samples, or _samples to build "
            "class-balanced Focus mosaics."
        )

    target_transform = getattr(dataset, "target_transform", None)
    if target_transform is not None:
        targets = [target_transform(int(target)) for target in targets]

    return np.asarray(targets, dtype=np.int64)


def index_dataset_by_class(dataset):
    class_to_indices = {}
    for index, target in enumerate(dataset_targets(dataset)):
        class_to_indices.setdefault(int(target), []).append(index)

    if len(class_to_indices) < 2:
        raise ValueError("Focus mosaic generation requires at least two classes.")

    return class_to_indices


def validate_mosaics(source_indices, component_labels, positions, targets):
    for indices, labels, target_positions, target in zip(
        source_indices,
        component_labels,
        positions,
        targets,
    ):
        target_indices = [
            image_index
            for image_index, is_target in zip(indices, target_positions)
            if is_target
        ]
        distractor_labels = [
            label for label, is_target in zip(labels, target_positions) if not is_target
        ]
        if (
            len(target_indices) != 2
            or target_indices[0] != target_indices[1]
            or any(
                label != target
                for label, is_target in zip(labels, target_positions)
                if is_target
            )
            or any(label == target for label in distractor_labels)
        ):
            raise RuntimeError(
                "Each Focus mosaic must contain the same target image exactly "
                "twice and two images whose labels differ from the target."
            )


def load_image(dataset, index, expected_label):
    image, label = dataset[index]
    if int(label) != expected_label:
        raise RuntimeError(
            f"Dataset label mismatch at index {index}: expected {expected_label}, "
            f"got {int(label)}."
        )
    if not torch.is_tensor(image):
        raise TypeError("The dataset transform must return image tensors.")
    return image.detach().cpu().numpy()


def build_focus_mosaics(dataset, class_to_indices, n_mosaics, rng):
    classes = np.asarray(sorted(class_to_indices), dtype=np.int64)
    mosaic_images = []
    targets = []
    positions = []
    source_indices = []
    component_labels = []

    for _ in range(n_mosaics):
        target = int(rng.choice(classes))
        distractor_classes = classes[classes != target]
        distractor_labels = tuple(
            int(label)
            for label in rng.choice(
                distractor_classes,
                size=2,
                replace=True,
            )
        )

        target_index = int(rng.choice(class_to_indices[target]))
        distractor_indices = tuple(
            int(rng.choice(class_to_indices[label])) for label in distractor_labels
        )
        target_image = load_image(dataset, target_index, target)
        distractor_images = [
            load_image(dataset, index, label)
            for index, label in zip(distractor_indices, distractor_labels)
        ]

        target_quadrants = set(rng.choice(4, size=2, replace=False).tolist())
        target_positions = []
        quadrant_images = []
        quadrant_indices = []
        quadrant_labels = []
        distractor_index = 0

        for quadrant in range(4):
            is_target = quadrant in target_quadrants
            target_positions.append(int(is_target))
            if is_target:
                quadrant_images.append(target_image)
                quadrant_indices.append(target_index)
                quadrant_labels.append(target)
            else:
                quadrant_images.append(distractor_images[distractor_index])
                quadrant_indices.append(distractor_indices[distractor_index])
                quadrant_labels.append(distractor_labels[distractor_index])
                distractor_index += 1

        mosaic_images.append(build_single_mosaic(quadrant_images))
        targets.append(target)
        positions.append(target_positions)
        source_indices.append(tuple(quadrant_indices))
        component_labels.append(tuple(quadrant_labels))

    mosaic_images = np.stack(mosaic_images)
    positions = np.asarray(positions, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    validate_mosaics(source_indices, component_labels, positions, targets)

    return FocusMosaics(
        images=mosaic_images,
        targets=targets,
        positions=positions,
        source_indices=source_indices,
        component_labels=component_labels,
    )


def positions_to_masks(positions, image_shape):
    height, width = image_shape[-2:]
    half_height = height // 2
    half_width = width // 2
    masks = np.zeros((len(positions), 1, height, width), dtype=np.float32)

    for index, target_positions in enumerate(positions):
        masks[index, :, :half_height, :half_width] = target_positions[0]
        masks[index, :, :half_height, half_width:] = target_positions[1]
        masks[index, :, half_height:, :half_width] = target_positions[2]
        masks[index, :, half_height:, half_width:] = target_positions[3]

    return masks


def format_positions(positions):
    return ",".join(
        name for name, is_target in zip(QUADRANT_NAMES, positions) if is_target
    )


def evaluate_mosaics(
    model,
    dataset,
    explainers,
    metric,
    device,
    args,
    output_file,
):
    rows = []
    class_to_indices = index_dataset_by_class(dataset)
    rng = np.random.default_rng(args.seed)
    n_batches = math.ceil(args.n_mosaics / args.batch_size)

    for batch_index in range(n_batches):
        batch_start = batch_index * args.batch_size
        batch_size = min(args.batch_size, args.n_mosaics - batch_start)
        print(f"Evaluating Focus batch {batch_index + 1}/{n_batches}")
        mosaics = build_focus_mosaics(
            dataset,
            class_to_indices,
            n_mosaics=batch_size,
            rng=rng,
        )

        for explainer_name, (explain_func, explain_kwargs) in explainers.items():
            empty_cache()
            print(f"Computing {explainer_name} attributions")
            attributions = explain_func(
                model,
                mosaics.images,
                mosaics.targets,
                device=device,
                **explain_kwargs,
            )
            relevances = mosaics.images * attributions
            scores = metric(
                model=model,
                x_batch=mosaics.images,
                y_batch=mosaics.targets,
                a_batch=relevances,
                custom_batch=mosaics.positions,
                channel_first=True,
                device=device,
            )

            for batch_mosaic_index, score in enumerate(scores):
                rows.append(
                    {
                        "model_name": args.model_name,
                        "dataset": args.dataset,
                        "explainer": explainer_name,
                        "evaluation_batch": batch_index,
                        "mosaic_index": batch_start + batch_mosaic_index,
                        "target_label": int(mosaics.targets[batch_mosaic_index]),
                        "target_positions": format_positions(
                            mosaics.positions[batch_mosaic_index]
                        ),
                        "component_labels": tuple(
                            mosaics.component_labels[batch_mosaic_index]
                        ),
                        "source_indices": tuple(
                            mosaics.source_indices[batch_mosaic_index]
                        ),
                        "score": float(score),
                    }
                )

        # Focus runs can be long, so checkpoint after each evaluation batch.
        pd.DataFrame(rows).to_pickle(f"{output_file}.pkl")

    return pd.DataFrame(rows)


def main(args):
    if args.n_mosaics < 1:
        raise ValueError("--n_mosaics must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be at least 1.")

    seed_everything(args.seed)
    device, loader, model = setup_notebook(
        batch_size=args.batch_size,
        model_name=args.model_name,
        model_source=args.model_source,
        seed=args.seed,
        dataset=args.dataset,
    )
    temperatures, _, _, _ = get_default_kwargs()
    soften_module_inplace_(
        model,
        temperatures=temperatures,
        standard_backward=False,
        fill_default_temperatures=True,
    )

    if args.model_name.startswith("resnet"):
        gc_layer = model[1].layer4[-1].conv3
    elif args.model_name.startswith("vgg"):
        gc_layer = model[1].avgpool
    else:
        gc_layer = None

    explainers = default_explainers(EXPLAINER_NAMES, gc_layer=gc_layer)
    metric = quantus.Focus(
        abs=False,
        normalise=False,
        return_aggregate=False,
        disable_warnings=True,
    )

    output_file = f"{args.output_file}_model_name={args.model_name}"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    results = evaluate_mosaics(
        model=model,
        dataset=loader.dataset,
        explainers=explainers,
        metric=metric,
        device=device,
        args=args,
        output_file=output_file,
    )
    summary = results.groupby("explainer")["score"].agg(["mean", "std", "count"])
    summary.to_csv(f"{output_file}_summary.csv")

    print(summary)
    print(f"Detailed results saved to {output_file}.pkl")
    print(f"Summary saved to {output_file}_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate attribution methods with input-attribution Focus."
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output path without the extension or model name.",
    )
    parser.add_argument("--model_name", default="resnet50")
    parser.add_argument(
        "--model_source",
        choices=["torchvision", "timm"],
        default="torchvision",
    )
    parser.add_argument(
        "--n_mosaics",
        type=int,
        default=80,
        help="Total number of randomly sampled Focus mosaics.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of mosaics evaluated together.",
    )
    parser.add_argument("--seed", type=int, default=314)
    parser.add_argument(
        "--dataset",
        choices=["imagenette", "imagenet"],
        default="imagenet",
    )
    main(parser.parse_args())
