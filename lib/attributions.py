import gc

import numpy as np
import quantus
import torch
from captum.attr import Saliency

from lib.helpers import squeeze_channels
from lib.pga import PGA
from lib.surrogates import set_module_standard_backward_, soften_module_inplace_


class GradientAscentDiff:
    def __init__(
        self,
        model,
        squeeze_channel_mode=None,
        **pga_kwargs,
    ):
        self.model = model
        self.atk = PGA(
            self.model,
            **pga_kwargs,
        )
        self.atk.set_mode_targeted_by_label(quiet=True)
        self.squeeze_channel_mode = squeeze_channel_mode

    def attribute(self, inputs, target):
        if isinstance(inputs, np.ndarray):
            device = next(self.model.parameters()).device
            inputs = torch.as_tensor(inputs, device=device)
            target = torch.as_tensor(target, device=device)
        else:
            inputs = inputs.to(self.atk.device)
            target = target.to(self.atk.device)

        adv_inputs = self.atk(inputs, target)

        attributions = (
            adv_inputs - inputs
        )  # if clip_margin is not None, then usually grad != (adv_images - images) due to the clipping!

        if self.squeeze_channel_mode is not None:
            attributions = squeeze_channels(
                attributions,
                mode=self.squeeze_channel_mode,
            )

        return attributions


class PullbackAscentDiff(GradientAscentDiff):
    def __init__(
        self,
        model,
        temperatures=None,
        squeeze_channel_mode=None,
        **pga_kwargs,
    ):
        super().__init__(
            model,
            squeeze_channel_mode=squeeze_channel_mode,
            **pga_kwargs,
        )
        self.temperatures = temperatures

    def attribute(self, inputs, target):
        if self.temperatures is not None:
            # NOTE: This modifies the model IN PLACE,
            # but should not affect forward nor backward passes,
            # as we restore standard_backward later.
            soften_module_inplace_(
                self.model,
                temperatures=self.temperatures,
                standard_backward=False,
                fill_default_temperatures=False,
            )
        else:
            set_module_standard_backward_(self.model, standard_backward=False)

        attributions = super().attribute(inputs, target)

        set_module_standard_backward_(self.model, standard_backward=True)

        return attributions


class DoublePullbackAscentDiff:
    def __init__(self, model, pga_kwargs_1, pga_kwargs_2, squeeze_channel_mode=None):
        self.pad1 = PullbackAscentDiff(
            model,
            **pga_kwargs_1,
        )
        self.pad2 = PullbackAscentDiff(
            model,
            squeeze_channel_mode=squeeze_channel_mode,
            **pga_kwargs_2,
        )

    def attribute(self, inputs, target):
        inter_attributions = self.pad1.attribute(inputs, target)
        final_attributions = self.pad2.attribute(inputs + inter_attributions, target)
        return final_attributions


# QUANTUS ADAPTERS
# TODO: PGA assumes images are in [-1,1], so we may need to add normalization here?


def quantus_gradient_ascent_diff_explain_func(
    model,
    inputs,
    targets,
    squeeze_channel_mode=None,
    device=None,
    **pga_kwargs,
):
    """
    Quantus-compatible explain_func for LocalGradientAscent.
    Args:
        model: PyTorch model
        inputs: torch.Tensor or np.ndarray, shape (B, C, H, W)
        targets: torch.Tensor or np.ndarray, shape (B,)
        alpha, steps, eps: hyperparameters for LocalGradientAscent
    Returns:
        attributions: np.ndarray, shape (B, C, H, W)
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)

    inputs = torch.as_tensor(inputs, device=device)
    targets = torch.as_tensor(targets, device=device)

    gad = GradientAscentDiff(
        model,
        squeeze_channel_mode=squeeze_channel_mode,
        **pga_kwargs,
    )
    attributions = gad.attribute(inputs, targets)
    return attributions.detach().cpu().numpy()


def quantus_pullback_ascent_diff_explain_func(
    model,
    inputs,
    targets,
    temperatures=None,
    squeeze_channel_mode=None,
    device=None,
    **pga_kwargs,
):
    """
    Quantus-compatible explain_func for LocalGradientAscent.
    Args:
        model: PyTorch model
        inputs: torch.Tensor or np.ndarray, shape (B, C, H, W)
        targets: torch.Tensor or np.ndarray, shape (B,)
        temperatures: dict[str, float], temperatures for SurrogateModules
        alpha, steps, eps: hyperparameters for LocalGradientAscent
    Returns:
        attributions: np.ndarray, shape (B, C, H, W)
    """
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)

    inputs = torch.as_tensor(inputs, device=device)
    targets = torch.as_tensor(targets, device=device)

    pad = PullbackAscentDiff(
        model,
        temperatures=temperatures,
        squeeze_channel_mode=squeeze_channel_mode,
        **pga_kwargs,
    )
    attributions = pad.attribute(inputs, targets)
    return attributions.detach().cpu().numpy()


def quantus_double_pullback_ascent_diff_explain_func(
    model,
    inputs,
    targets,
    pga_kwargs_1,
    pga_kwargs_2,
    squeeze_channel_mode=None,
    device=None,
):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)

    inputs = torch.as_tensor(inputs, device=device)
    targets = torch.as_tensor(targets, device=device)

    dpad = DoublePullbackAscentDiff(
        model,
        squeeze_channel_mode=squeeze_channel_mode,
        pga_kwargs_1=pga_kwargs_1,
        pga_kwargs_2=pga_kwargs_2,
    )
    attributions = dpad.attribute(inputs, targets)

    return attributions.detach().cpu().numpy()


def fusiongrad_explainer(
    model, inputs, targets, abs=False, normalise=False, *args, **kwargs
) -> np.array:
    """Wrapper aorund captum's FusionGrad implementation."""

    std = kwargs.get("std", 0.5)
    mean = kwargs.get("mean", 1.0)
    n = kwargs.get("n", 10)
    m = kwargs.get("m", 10)
    sg_std = kwargs.get("sg_std", 0.5)
    sg_mean = kwargs.get("sg_mean", 0.0)
    posterior_mean = kwargs.get("posterior_mean", None)
    noise_type = kwargs.get("noise_type", "multiplicative")
    clip = kwargs.get("clip", False)

    def _sample(
        model, posterior_mean, std, distribution=None, noise_type="multiplicative"
    ):
        """Implmentation to sample a model."""

        # Load model params.
        model.load_state_dict(posterior_mean)

        # If std is not zero, loop over each layer and add Gaussian noise.
        if not std == 0.0:
            with torch.no_grad():
                for layer in model.parameters():
                    if noise_type == "additive":
                        layer.add_(distribution.sample(layer.size()).to(layer.device))
                    elif noise_type == "multiplicative":
                        layer.mul_(distribution.sample(layer.size()).to(layer.device))
                    else:
                        print(
                            "Set NoiseGrad attribute 'noise_type' to either 'additive' or 'multiplicative' (str)."
                        )

        return model

    # Creates a normal (also called Gaussian) distribution.
    distribution = torch.distributions.normal.Normal(
        loc=torch.as_tensor(mean, dtype=torch.float),
        scale=torch.as_tensor(std, dtype=torch.float),
    )

    # Set model in evaluate mode.
    model.to(kwargs.get("device", None))
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = (
            torch.Tensor(inputs)
            .reshape(
                -1,
                kwargs.get("nr_channels", 3),
                kwargs.get("img_size", 224),
                kwargs.get("img_size", 224),
            )
            .to(kwargs.get("device", None))
        )
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).long().to(kwargs.get("device", None))

    assert (
        len(np.shape(inputs)) == 4
    ), "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size) e.g., (1, 3, 224, 224)."

    if inputs.shape[0] > 1:
        explanation = torch.zeros(
            (
                n,
                m,
                inputs.shape[0],
                kwargs.get("img_size", 224),
                kwargs.get("img_size", 224),
            )
        )
    else:
        explanation = torch.zeros(
            (n, m, kwargs.get("img_size", 224), kwargs.get("img_size", 224))
        )

    for i in range(n):
        model = _sample(
            model=model,
            posterior_mean=posterior_mean,
            std=std,
            distribution=distribution,
            noise_type=noise_type,
        )
        for j in range(m):
            inputs_noisy = inputs + torch.randn_like(inputs) * sg_std + sg_mean
            if clip:
                inputs_noisy = torch.clip(inputs_noisy, min=0.0, max=1.0)

            explanation[i][j] = (
                Saliency(model)
                .attribute(inputs_noisy, targets, abs=abs)
                .sum(axis=1)
                .reshape(-1, kwargs.get("img_size", 224), kwargs.get("img_size", 224))
                .cpu()
                .data
            )

    explanation = explanation.mean(axis=(0, 1))

    gc.collect()
    torch.cuda.empty_cache()

    if normalise:
        explanation = quantus.normalise_func.normalise_by_negative(explanation)

    if isinstance(explanation, torch.Tensor):
        if explanation.requires_grad:
            return explanation.cpu().detach().numpy()
        return explanation.cpu().numpy()

    return explanation


def smoothgrad_explainer(
    model, inputs, targets, abs=False, normalise=False, *args, **kwargs
) -> np.ndarray:
    sg_std = kwargs.get("sg_std", 0.15)
    sg_mean = kwargs.get("sg_mean", 0.0)
    n = kwargs.get("n", 50)
    clip = kwargs.get("clip", False)
    device = kwargs.get("device", None)

    model.to(device)
    model.eval()

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.Tensor(inputs).to(device)
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).long().to(device)

    # if kwargs.get("channels_first", True) is False and inputs.ndim == 4:
    #     inputs = inputs.permute(0, 3, 1, 2)

    # if len(inputs.shape) != 4:
    #     inputs = inputs.reshape(
    #         -1,
    #         kwargs.get("nr_channels", 3),
    #         kwargs.get("img_size", 224),
    #         kwargs.get("img_size", 224),
    #     )

    b, c, h, w = inputs.shape
    reduce_channels = kwargs.get("reduce_channels", False)

    if reduce_channels:
        explanation = torch.zeros((n, b, h, w))
    else:
        explanation = torch.zeros((n, b, c, h, w))
    # explanation = torch.zeros((n, b, c, h, w))

    saliency = Saliency(model)

    for i in range(n):
        inputs_noisy = inputs + torch.randn_like(inputs) * sg_std + sg_mean
        if clip:
            inputs_noisy = torch.clamp(inputs_noisy, 0.0, 1.0)

        attrs = saliency.attribute(inputs_noisy, target=targets, abs=abs)

        if reduce_channels:
            attrs = attrs.sum(dim=1)

        explanation[i] = attrs.detach().cpu()

    explanation = explanation.mean(dim=0)

    # gc.collect()
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()

    if normalise:
        # import quantus

        explanation = quantus.normalise_func.normalise_by_negative(explanation)

    if isinstance(explanation, torch.Tensor):
        return explanation.detach().cpu().numpy()

    return explanation
