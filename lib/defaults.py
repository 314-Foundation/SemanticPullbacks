from torch import nn

from lib.surrogates import LayerNorm2d, PVTAttention


def get_default_kwargs():
    temperatures = {
        nn.ReLU: 0.3,
        nn.SiLU: 1.6,
        nn.GELU: 1.0,
        nn.MaxPool2d: 0.3,
        # nn.LayerNorm: None,
        # LayerNorm2d: None,
        nn.MultiheadAttention: 1.0,
        PVTAttention: 1.5,
    }
    pga_kwargs_multi = {
        # "alpha": None,
        # "alpha": 0.1,
        "alpha": 20,
        # "steps": 1,
        "steps": 10,
        "eps": None,
        # "eps": 100,
        "normalize_step": True,
        # "normalize_step": False,
        "relative_alpha": False,
        # "relative_alpha": True,
        # "clip_margin": None,
        "clip_margin": 0.0,
    }
    pga_kwargs_grad = {
        "alpha": None,
        "steps": 1,
        "eps": None,
        "normalize_step": False,
        "relative_alpha": False,
        # "relative_alpha": True,
        "clip_margin": None,
    }
    pga_kwargs_adv = {
        "eps": 6.0,
        "alpha": 1.0,
        "steps": 10,
        "use_cross_entropy_loss": True,
    }

    return temperatures, pga_kwargs_multi, pga_kwargs_grad, pga_kwargs_adv
