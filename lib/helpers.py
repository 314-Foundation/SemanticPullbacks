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


def maxpool2d_param_extractor(child):
    return {
        "kernel_size": child.kernel_size,
        "stride": child.stride,
        "padding": child.padding,
        "dilation": child.dilation,
        "return_indices": child.return_indices,
        "ceil_mode": child.ceil_mode,
    }


def batchnorm_param_extractor(child):
    return {
        "num_features": child.num_features,
        "eps": child.eps,
        "momentum": child.momentum,
        "affine": child.affine,
        "track_running_stats": child.track_running_stats,
        "weight": None if child.weight is None else child.weight.clone(),
        "bias": None if child.bias is None else child.bias.clone(),
        "running_mean": child.running_mean.clone(),
        "running_var": child.running_var.clone(),
    }


def layernorm_param_extractor(child):
    return {
        "normalized_shape": tuple(child.normalized_shape),
        "eps": child.eps,
        "elementwise_affine": child.elementwise_affine,
        # affine parameters (may be None if elementwise_affine=False)
        "weight": None if child.weight is None else child.weight.clone(),
        "bias": None if child.bias is None else child.bias.clone(),
    }


# def multihead_attention_param_extractor(child):
#     return {
#         # --- konstruktor / config ---
#         "embed_dim": child.embed_dim,
#         "num_heads": child.num_heads,
#         "dropout": child.dropout,
#         "bias": child.in_proj_bias is not None,
#         "add_bias_kv": child.bias_k is not None and child.bias_v is not None,
#         "add_zero_attn": child.add_zero_attn,
#         "kdim": child.kdim,
#         "vdim": child.vdim,
#         "batch_first": getattr(child, "batch_first", False),
#         # --- parametry projekcji wejściowej (Q,K,V) ---
#         "in_proj_weight": (
#             child.in_proj_weight.clone() if child.in_proj_weight is not None else None
#         ),
#         "in_proj_bias": (
#             child.in_proj_bias.clone() if child.in_proj_bias is not None else None
#         ),
#         # --- parametry projekcji wyjściowej ---
#         "out_proj_weight": child.out_proj.weight.clone(),
#         "out_proj_bias": (
#             child.out_proj.bias.clone() if child.out_proj.bias is not None else None
#         ),
#         # --- ewentualne bias_k / bias_v ---
#         "bias_k": child.bias_k.clone() if child.bias_k is not None else None,
#         "bias_v": child.bias_v.clone() if child.bias_v is not None else None,
#     }


def replace_module_with_custom_(
    module,
    custom_cls,
    original_cls=None,
    by_name=None,
    param_extractor=None,
    replace_class=False,
):
    """
    Recursively replaces submodules:
    - if param_extractor is None → in-place replacement: child.__class__ = custom_cls
    - if param_extractor is not None → creates a new module via custom_cls(**params)
    """
    for name, child in module.named_children():
        should_replace = (by_name is not None and name == by_name) or (
            original_cls is not None and isinstance(child, original_cls)
        )

        if should_replace:
            if replace_class:
                # replace class in-place
                # if not issubclass(custom_cls, child.__class__):
                #     raise TypeError(
                #         f"custom_cls ({custom_cls.__name__}) must be a subclass of "
                #         f"{child.__class__.__name__} when using in-place replacement"
                #     )
                child.__class__ = custom_cls
            else:
                # create new module and replace
                params = param_extractor(child) if param_extractor is not None else {}
                setattr(module, name, custom_cls(**params))
        else:
            replace_module_with_custom_(
                child,
                custom_cls,
                original_cls=original_cls,
                by_name=by_name,
                param_extractor=param_extractor,
                replace_class=replace_class,
            )


def show_images(images, adv_images, k=5):
    uimages = images.unflatten(0, (5, 1))
    uadv_images = adv_images.unflatten(0, (5, k))

    udiff = uadv_images - uimages

    show_adv = torch.cat([uimages, uadv_images], dim=1).flatten(0, 1)
    show_diff = torch.cat([uimages, udiff], dim=1).flatten(0, 1)

    return show_adv, show_diff
