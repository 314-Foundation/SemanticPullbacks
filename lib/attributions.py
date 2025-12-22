import numpy as np
import torch

from lib.pga import PGA
from lib.surrogates import set_module_standard_backward_, soften_module_inplace_


class LocalGradientAscent:
    def __init__(self, model, alpha=None, steps=1, eps=None):
        self.model = model
        self.atk = PGA(self.model, alpha=alpha, steps=steps, eps=eps)
        self.atk.set_mode_targeted_by_label()

    def attribute(self, inputs, target):
        inputs = inputs.to(self.atk.device)
        target = target.to(self.atk.device)

        adv_inputs = self.atk(inputs, target)

        attributions = adv_inputs - inputs

        return attributions


class LocalRelevanceAscent(LocalGradientAscent):
    # NOTE: This modifies the model IN PLACE, but should not affect forward nor backward passes, as we leave standard_backward=True
    def __init__(self, model, temperatures, alpha=None, steps=1, eps=None):
        super().__init__(model, alpha=alpha, steps=steps, eps=eps)

        self.temperatures = temperatures
        soften_module_inplace_(self.model, temperatures=self.temperatures)
        set_module_standard_backward_(
            self.model, standard_backward=True
        )  # don't change the model behavior!

    def attribute(self, inputs, target):
        set_module_standard_backward_(self.model, standard_backward=False)

        attributions = super().attribute(inputs, target)

        set_module_standard_backward_(self.model, standard_backward=True)

        return attributions


# QUANTUS ADAPTERS
# TODO: PGA assumes images are in [-1,1], so we may need to add normalization here?


def quantus_local_gradient_ascent_explain_func(
    model,
    inputs,
    targets,
    alpha=None,
    steps=1,
    eps=None,
    device=None,
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

    lga = LocalGradientAscent(model, alpha=alpha, steps=steps, eps=eps)
    attributions = lga.attribute(inputs, targets)
    return attributions.detach().cpu().numpy()


def quantus_local_relevance_ascent_explain_func(
    model, inputs, targets, temperatures, alpha=None, steps=1, eps=None
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

    lga = LocalRelevanceAscent(
        model, temperatures=temperatures, alpha=alpha, steps=steps, eps=eps
    )
    attributions = lga.attribute(inputs, targets)
    return attributions.detach().cpu().numpy()
