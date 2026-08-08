import copy
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


class PullbackWrapper:
    def __init__(self, func, temperatures=None):
        self.func = func
        self.temperatures = temperatures

    def __call__(self, model, inputs, targets, **kwargs):
        # NOTE: This modifies the model IN PLACE,
        # but should not affect later forward nor backward passes,
        # as we restore standard_backward in the finally block.
        if self.temperatures is not None:
            soften_module_inplace_(
                model,
                temperatures=self.temperatures,
                standard_backward=False,
                fill_default_temperatures=False,
            )
        else:
            set_module_standard_backward_(model, standard_backward=False)

        try:
            return self.func(model, inputs, targets, **kwargs)
        finally:
            set_module_standard_backward_(model, standard_backward=True)


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
            soften_module_inplace_(
                self.model,
                temperatures=self.temperatures,
                standard_backward=False,
                fill_default_temperatures=False,
            )
        else:
            set_module_standard_backward_(self.model, standard_backward=False)

        try:
            attributions = super().attribute(inputs, target)
        finally:
            # Reset standard backward after pullback attribution call.
            set_module_standard_backward_(self.model, standard_backward=True)

        return attributions


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


def smoothgrad_explainer(
    model, inputs, targets, abs=False, normalise=False, *args, **kwargs
) -> np.ndarray:
    sg_std = kwargs.get("sg_std", 0.15)
    sg_mean = kwargs.get("sg_mean", 0.0)
    n = kwargs.get("n", 50)
    clip = kwargs.get("clip", False)
    device = kwargs.get("device", None)
    use_pullback = kwargs.get("use_pullback", False)

    model.to(device)
    model.eval()

    if use_pullback:
        set_module_standard_backward_(model, standard_backward=False)

    if not isinstance(inputs, torch.Tensor):
        inputs = torch.Tensor(inputs).to(device)
    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).long().to(device)

    b, c, h, w = inputs.shape

    explanation = torch.zeros((n, b, c, h, w))

    saliency = Saliency(model)

    for i in range(n):
        inputs_noisy = inputs + torch.randn_like(inputs) * sg_std + sg_mean
        if clip:
            inputs_noisy = torch.clamp(inputs_noisy, -1.0, 1.0)

        attrs = saliency.attribute(inputs_noisy, target=targets, abs=abs)

        explanation[i] = attrs.detach().cpu()

    explanation = explanation.mean(dim=0)

    if normalise:
        explanation = quantus.normalise_func.normalise_by_negative(explanation)

    if isinstance(explanation, torch.Tensor):
        return explanation.detach().cpu().numpy()

    if use_pullback:
        set_module_standard_backward_(model, standard_backward=True)

    return explanation


def fusiongrad_explainer(
    model, inputs, targets, abs=False, normalise=False, *args, **kwargs
) -> np.ndarray:
    std = kwargs.get("std", 0.1)
    mean = kwargs.get("mean", 1.0)
    n = kwargs.get("n", 10)
    m = kwargs.get("m", 10)
    sg_std = kwargs.get("sg_std", 0.15)
    sg_mean = kwargs.get("sg_mean", 0.0)
    posterior_mean = kwargs.get("posterior_mean", None)
    noise_type = kwargs.get("noise_type", "multiplicative")
    clip = kwargs.get("clip", False)
    device = kwargs.get("device", None)
    use_pullback = kwargs.get("use_pullback", False)

    model.to(device)
    model.eval()

    if use_pullback:
        set_module_standard_backward_(model, standard_backward=False)

    original_state = copy.deepcopy(model.state_dict())

    try:
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.Tensor(inputs).to(device)
        else:
            inputs = inputs.to(device)

        if not isinstance(targets, torch.Tensor):
            targets = torch.as_tensor(targets).long().to(device)
        else:
            targets = targets.long().to(device)

        b, c, h, w = inputs.shape

        explanation = torch.zeros((n, m, b, c, h, w))

        if posterior_mean is None:
            posterior_mean = copy.deepcopy(model.state_dict())

        distribution = torch.distributions.normal.Normal(
            loc=torch.as_tensor(mean, dtype=torch.float32, device=device),
            scale=torch.as_tensor(std + 1e-6, dtype=torch.float32, device=device),
        )

        def _sample(
            model, posterior_mean, distribution, std, noise_type="multiplicative"
        ):
            model.load_state_dict(posterior_mean)

            if std != 0.0:
                with torch.no_grad():
                    for param in model.parameters():
                        noise = distribution.sample(param.shape).to(param.device)

                        if noise_type == "additive":
                            param.add_(noise)
                        elif noise_type == "multiplicative":
                            param.mul_(noise)
                        else:
                            raise ValueError(
                                "noise_type must be 'additive' or 'multiplicative'."
                            )

            return model

        for i in range(n):
            model = _sample(
                model=model,
                posterior_mean=posterior_mean,
                distribution=distribution,
                std=std,
                noise_type=noise_type,
            )

            saliency = Saliency(model)

            for j in range(m):
                inputs_noisy = inputs + torch.randn_like(inputs) * sg_std + sg_mean

                if clip:
                    inputs_noisy = torch.clamp(inputs_noisy, -1.0, 1.0)

                attrs = saliency.attribute(inputs_noisy, target=targets, abs=abs)

                explanation[i, j] = attrs.detach().cpu()

        explanation = explanation.mean(dim=(0, 1))

        if normalise:
            explanation = quantus.normalise_func.normalise_by_negative(explanation)

        if isinstance(explanation, torch.Tensor):
            return explanation.detach().cpu().numpy()

        return explanation

    finally:
        model.load_state_dict(original_state)

        if use_pullback:
            set_module_standard_backward_(model, standard_backward=True)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
