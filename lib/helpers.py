import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torch import nn


def plot_example_grid(
    X,
    nrow=4,
    column_titles=None,
    cmap=None,
    normalize=True,
    renormalize_fn=None,
    title=None,
    figsize=None,
    save_path=None,
    dpi=80,
):
    """
    X: tensor (B, C, H, W)
    nrow: number of images in each row
    column_titles: list of str - only if B is a multiple of nrow
    """
    X = X.detach().cpu()

    if renormalize_fn:
        X = renormalize_fn(X)
        normalize = False

    grid = vutils.make_grid(
        X, nrow=nrow, normalize=normalize, scale_each=True, padding=2
    )
    npimg = grid.permute(1, 2, 0).numpy()

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
