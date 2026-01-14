from typing import Optional

import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torch import nn


def plot_example_grid(
    X,
    nrow=4,
    column_titles=None,
    cmap=None,
    scale_each: bool = True,
    normalize=True,
    value_range: Optional[tuple[int, int]] = None,
    renormalize_fn=None,
    title=None,
    figsize=None,
    save_path=None,
    dpi=80,
    heatmap=False,
):
    """
    X: tensor (B, C, H, W)
    nrow: number of images in each row
    column_titles: list of str - only if B is a multiple of nrow
    """
    if heatmap:
        X = squeeze_channels(X, mode="mean")
        X = normalise_by_negative_batch(X)

    X = X.detach().cpu()

    # work around the make_grid expanding the image to 3 channels
    map_to_one_channel = (X.shape[1] == 1) and (cmap is not None)

    if renormalize_fn:
        X = renormalize_fn(X)
        normalize = False

    grid = vutils.make_grid(
        X,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
        scale_each=scale_each,
        padding=2,
    )

    npimg = grid.permute(1, 2, 0)
    if map_to_one_channel:
        npimg = npimg.mean(dim=-1, keepdim=True)
    npimg = npimg.numpy()

    H, W = npimg.shape[:2]
    img_h = H // (len(X) // nrow)
    img_w = W // nrow

    if figsize is not None:
        plt.figure(figsize=figsize, dpi=dpi)
    else:
        plt.figure(figsize=(min(20, 3.5 * nrow), 3.5 * (len(X) // nrow + 1)), dpi=dpi)

    plt.imshow(npimg, cmap=cmap or ("gray" if npimg.shape[-1] == 1 else None))

    # Add column titles
    if column_titles is not None:
        for i, col_title in enumerate(column_titles):
            x_center = i * img_w + img_w / 2
            plt.text(x_center, y=-5, s=col_title, fontsize=12, ha="center", va="bottom")

    if title:
        plt.title(title)

    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()


def squeeze_channels(
    attributions: torch.Tensor,
    mode="mean",
) -> torch.Tensor:
    if mode == "mean":
        attributions = torch.mean(attributions, dim=1, keepdim=True)
    elif mode == "sum":
        attributions = torch.sum(attributions, dim=1, keepdim=True)
    elif mode == "abs_mean":
        attributions = torch.mean(torch.abs(attributions), dim=1, keepdim=True)
    elif mode == "norm":
        attributions = torch.norm(attributions, dim=1, keepdim=True)
    else:
        raise ValueError(f"Unknown mode {mode} for squeeze_channels")

    return attributions


def normalise_by_negative_batch(a: torch.Tensor) -> torch.Tensor:
    """
    Normalise a batch of images/tensors between [-1, 1] per image using flatten/unflatten.

    Parameters
    ----------
    a: torch.Tensor
        Batch of images/tensors, e.g., shape (B, C, H, W).

    Returns
    -------
    torch.Tensor
        Normalized batch, same shape as input.
    """
    shape = a.shape
    B = shape[0]
    rest_shape = shape[1:]
    a_flat = a.flatten(1)  # shape: (B, N)

    a_max = a_flat.max(dim=1, keepdim=True)[0]
    a_min = a_flat.min(dim=1, keepdim=True)[0]

    mask_zero = (a_max == 0) & (a_min == 0)
    mask_pos = a_min > 0
    mask_neg = a_max < 0
    mask_mix = (a_min < 0) & (a_max > 0)

    out_flat = torch.zeros_like(a_flat)
    out_flat = torch.where(mask_pos, a_flat / a_max.clamp(min=1e-8), out_flat)
    out_flat = torch.where(mask_neg, -a_flat / a_min.clamp(max=-1e-8), out_flat)

    # Mixed sign case: normalize positive and negative parts separately
    if mask_mix.any():
        # For each batch element, apply only where mask_mix is True
        pos = (a_flat > 0).float()
        neg = (a_flat < 0).float()
        # Avoid division by zero
        a_max_safe = a_max.clone()
        a_max_safe[a_max_safe == 0] = 1e-8
        a_min_safe = a_min.clone()
        a_min_safe[a_min_safe == 0] = -1e-8
        mixed_val = pos * (a_flat / a_max_safe) - neg * (a_flat / a_min_safe)
        out_flat = torch.where(mask_mix, mixed_val, out_flat)

    out = out_flat.unflatten(1, rest_shape)
    return out


def plot_function(
    f,
    x_range=(-5, 5),
    num_points=1000,
    title="Function plot",
    xlabel="x",
    ylabel="f(x)",
    dpi=80,
):
    x = torch.linspace(x_range[0], x_range[1], num_points)
    y = f(x)

    y = y.detach().cpu().numpy()
    x = x.detach().cpu().numpy()

    plt.figure(figsize=(8, 4), dpi=dpi)
    plt.plot(x, y, label="f(x)")
    plt.axhline(0, color="black", linewidth=1, linestyle="--")  # y=0 axis
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_images(images, adv_images, k=5):
    uimages = images.unflatten(0, (5, 1))
    uadv_images = adv_images.unflatten(0, (5, k))

    udiff = uadv_images - uimages

    show_adv = torch.cat([uimages, uadv_images], dim=1).flatten(0, 1)
    show_diff = torch.cat([uimages, udiff], dim=1).flatten(0, 1)

    return show_adv, show_diff
