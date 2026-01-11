import numpy as np
import torch

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
        self.atk.set_mode_targeted_by_label()
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
            # as we leave standard_backward=True
            soften_module_inplace_(
                self.model,
                temperatures=self.temperatures,
                standard_backward=False,
            )
        else:
            set_module_standard_backward_(self.model, standard_backward=False)

        attributions = super().attribute(inputs, target)

        set_module_standard_backward_(self.model, standard_backward=True)

        return attributions


# QUANTUS ADAPTERS
# TODO: PGA assumes images are in [-1,1], so we may need to add normalization here?


def quantus_gradient_ascent_diff_explain_func(
    model,
    inputs,
    targets,
    sequeeze_channel_mode=None,
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

    if isinstance(inputs, np.ndarray):
        inputs = torch.as_tensor(inputs, device=device)
    if isinstance(targets, np.ndarray):
        targets = torch.as_tensor(targets, device=device)

    lga = GradientAscentDiff(
        model,
        sequeeze_channel_mode=sequeeze_channel_mode,
        **pga_kwargs,
    )
    attributions = lga.attribute(inputs, targets)
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

    if isinstance(inputs, np.ndarray):
        inputs = torch.as_tensor(inputs, device=device)
    if isinstance(targets, np.ndarray):
        targets = torch.as_tensor(targets, device=device)

    lga = PullbackAscentDiff(
        model,
        temperatures=temperatures,
        squeeze_channel_mode=squeeze_channel_mode,
        **pga_kwargs,
    )
    attributions = lga.attribute(inputs, targets)
    return attributions.detach().cpu().numpy()
