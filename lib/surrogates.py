import math
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def normal_cdf(x, temp=1.0):
    return 0.5 * (1 + torch.erf(x / (math.sqrt(2) * temp)))


class ModuleWithTemperature(nn.Module):
    def __init__(self, *args, temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def extra_repr(self):
        ret = super().extra_repr()
        if ret:
            ret += ", "
        return f"{ret}temperature={self.temperature}"


class SurrogateModule(ModuleWithTemperature):
    def backward_gradient(self, x):
        raise NotImplementedError

    # https://github.com/AdaptiveAILab/fgi
    def forward(self, x):
        step = super().forward(x)
        grad = self.backward_gradient(x)

        mul = x * grad.detach()
        y = mul - mul.detach() + step.detach()

        return y


class SurrogateReLU(SurrogateModule, nn.ReLU):
    def backward_gradient(self, x):
        return F.sigmoid(x / self.temperature)


class SurrogateSiLU(SurrogateModule, nn.SiLU):
    def backward_gradient(self, x):
        return F.sigmoid(x / self.temperature)


class SurrogateGELU(SurrogateModule, nn.GELU):
    def backward_gradient(self, x):
        return normal_cdf(x, self.temperature)


class SoftMaxPool2d(ModuleWithTemperature, nn.MaxPool2d):
    def forward(self, x):
        B, C, H, W = x.shape
        kH, kW = self.kernel_size, self.kernel_size

        # Unfold input to patches
        x_unf = F.unfold(
            x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding
        )
        x_unf = x_unf.view(B, C, kH * kW, -1)

        # Softmax pooling over spatial positions
        weights = F.softmax(x_unf / self.temperature, dim=2)
        weights = (
            weights.detach()
        )  # prevent gradients through weights, generally works better
        pooled = (x_unf * weights).sum(dim=2)

        # Reshape back to image
        out_H = (H + 2 * self.padding - kH) // self.stride + 1
        out_W = (W + 2 * self.padding - kW) // self.stride + 1
        return pooled.view(B, C, out_H, out_W)


class SurrogateMaxPool2d(SoftMaxPool2d):
    def forward(self, x):
        soft = super().forward(x)
        hard = F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )

        return hard.detach() + (soft - soft.detach())


# class SurrogateLayerNorm(nn.Module):
class SurrogateLayerNorm(ModuleWithTemperature, nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        # Normalize over the last D dimensions, where
        # D = len(normalized_shape), exactly like nn.LayerNorm. :contentReference[oaicite:0]{index=0}
        D = len(self.normalized_shape)
        dims = tuple(range(x.dim() - D, x.dim()))

        mean = x.mean(dim=dims, keepdim=True)
        # IMPORTANT: use biased variance (unbiased=False), same as LayerNorm. :contentReference[oaicite:1]{index=1}
        var = x.var(dim=dims, keepdim=True, unbiased=False)

        # don't compute gradients through mean and var during evaluation
        if not self.training:
            mean = mean.detach()
            var = var.detach()

        inv_std = torch.rsqrt(var + self.eps)
        x_hat = (x - mean) * inv_std

        if self.elementwise_affine:
            if self.weight is not None:
                x_hat = x_hat * self.weight
            if self.bias is not None:
                x_hat = x_hat + self.bias

        return x_hat


class SurrogateLayerNorm2d(SurrogateLayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x


class SurrogateMultiheadAttention(ModuleWithTemperature, nn.MultiheadAttention):
    def forward(self, query, key, value, *args, **kwargs):
        if not self.training:
            # we need to pass graidients through all the QKV terms to properly backpropagate the relevance
            # query = query.detach()
            # key = key.detach()
            # value = value.detach()
            orig, attn_weights = super().forward(query, key, value, *args, **kwargs)

            squery = query / self.temperature
            skey = key / self.temperature
            # dont pass gradients through temperature scaling
            squery = squery.detach() + (query - query.detach())
            skey = skey.detach() + (key - key.detach())

            softer, _ = super().forward(squery, skey, value, *args, **kwargs)

            ret = softer - softer.detach() + orig.detach()
            return ret, attn_weights

        return super().forward(query, key, value, *args, **kwargs)


SURROGATE_CLASS_MAP = {
    "ReLU": (SurrogateReLU, 0.3),
    "SiLU": (SurrogateSiLU, 0.6),
    "GELU": (SurrogateGELU, 1.0),
    "MaxPool2d": (SurrogateMaxPool2d, 0.2),
    "LayerNorm": (SurrogateLayerNorm, 1.0),
    "LayerNorm2d": (SurrogateLayerNorm2d, 1.0),
    "MultiheadAttention": (SurrogateMultiheadAttention, 1.2),
}


def replace_modules_with_surrogates_(
    module,
    temperatures,
    surrogate_prefix="Surrogate",
):
    for name, child in module.named_children():
        cls_name = child.__class__.__name__
        key = (
            cls_name[len(surrogate_prefix) :]
            if cls_name.startswith(surrogate_prefix)
            else cls_name
        )

        if key in temperatures:
            child.temperature = temperatures[key]

            if cls_name in SURROGATE_CLASS_MAP:
                child.__class__ = SURROGATE_CLASS_MAP[cls_name][0]
        else:
            replace_modules_with_surrogates_(
                child,
                temperatures=temperatures,
                surrogate_prefix=surrogate_prefix,
            )


def soften_module_inplace_(
    module,
    temperatures=None,
    surrogate_prefix="Surrogate",
):
    base_temperatures = {
        key: SURROGATE_CLASS_MAP[key][1] for key in SURROGATE_CLASS_MAP.keys()
    }
    if temperatures is not None:
        base_temperatures.update(temperatures)

    replace_modules_with_surrogates_(
        module,
        temperatures=base_temperatures,
        surrogate_prefix=surrogate_prefix,
    )
