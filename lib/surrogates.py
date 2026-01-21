import math
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F

# from timm.layers.norm import LayerNorm as TimmLayerNorm
from timm.models.pvt_v2 import Attention as PVTAttention
from torch import Tensor, nn
from torchvision.models.convnext import LayerNorm2d


def normal_cdf(x, temp=1.0):
    return 0.5 * (1 + torch.erf(x / (math.sqrt(2) * temp)))


def softmax_detach_norm_stable(x, dim=-1, detach=True):
    m = x.max(dim=dim, keepdim=True).values

    if detach:
        m = m.detach()

    ex = torch.exp(x - m)
    denom = ex.sum(dim=dim, keepdim=True)

    if detach:
        denom = denom.detach()

    return ex / denom


class SurrogateModule(nn.Module):
    def __init__(self, *args, temperature=1.0, standard_backward=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.standard_backward = standard_backward

    def extra_repr(self):
        ret = super().extra_repr()
        if ret:
            ret += ", "
        return f"{ret}standard_backward={self.standard_backward}, temperature={self.temperature}"

    @classmethod
    def replace_class_with_surrogate(
        cls, module, temperature=1.0, standard_backward=False
    ):
        module.__class__ = cls
        module.temperature = temperature
        module.standard_backward = standard_backward


class FGIModule(SurrogateModule):
    # https://github.com/AdaptiveAILab/fgi
    def backward_gradient(self, x):
        raise NotImplementedError

    def forward(self, x):
        step = super().forward(x)

        if self.standard_backward:
            return step

        grad = self.backward_gradient(x)

        mul = x * grad.detach()
        y = mul - mul.detach() + step.detach()

        return y


class SurrogateActivation(FGIModule):
    @classmethod
    def replace_class_with_surrogate(cls, module, *args, **kwargs):
        super().replace_class_with_surrogate(module, *args, **kwargs)
        module.inplace = False


class SurrogateReLU(SurrogateActivation, nn.ReLU):
    def backward_gradient(self, x):
        # return F.sigmoid(x / self.temperature)

        # approximate expected gating
        return normal_cdf(x, self.temperature)


class SurrogateSiLU(SurrogateActivation, nn.SiLU):
    def backward_gradient(self, x):
        return F.sigmoid(x / self.temperature)

        # approximate expected gating
        kappa = torch.tensor(1.702, device=x.device)
        denom = torch.sqrt(kappa**2 + self.temperature**2)

        return normal_cdf(x, denom)


class SurrogateGELU(SurrogateActivation, nn.GELU):
    def backward_gradient(self, x):
        return normal_cdf(x, self.temperature)
        # return F.sigmoid(x / self.temperature)

        # approximate expected gating
        one = torch.tensor(1.0, device=x.device)
        denom = torch.sqrt(one + self.temperature**2)

        return normal_cdf(x, denom)


class SoftMaxPool2d(SurrogateModule, nn.MaxPool2d):
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

        if not self.standard_backward:
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
        hard = F.max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )

        if self.standard_backward:
            return hard

        soft = super().forward(x)

        return hard.detach() + (soft - soft.detach())


class SurrogateLayerNorm(SurrogateModule, nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        # Normalize over the last D dimensions, where
        # D = len(normalized_shape), exactly like nn.LayerNorm. :contentReference[oaicite:0]{index=0}
        D = len(self.normalized_shape)
        dims = tuple(range(x.dim() - D, x.dim()))

        mean = x.mean(dim=dims, keepdim=True)
        # IMPORTANT: use biased variance (unbiased=False), same as LayerNorm. :contentReference[oaicite:1]{index=1}
        var = x.var(dim=dims, keepdim=True, unbiased=False)

        if not self.standard_backward:
            # don't compute gradients through mean and var
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
    # used in e.g. torchvision.models.convnext_tiny
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x


class SurrogateMultiheadAttention(SurrogateModule, nn.MultiheadAttention):
    # TODO: maybe this can be improved further?
    def forward(self, query, key, value, *args, **kwargs):
        orig, attn_weights = super().forward(query, key, value, *args, **kwargs)

        if self.standard_backward:
            return orig, attn_weights

        # query = query.detach()
        # key = key.detach()
        # value = value.detach()

        squery = query / self.temperature
        skey = key / self.temperature
        # dont pass gradients through temperature scaling
        squery = squery.detach() + (query - query.detach())
        skey = skey.detach() + (key - key.detach())

        softer, _ = super().forward(squery, skey, value, *args, **kwargs)

        ret = softer - softer.detach() + orig.detach()
        return ret, attn_weights


class SurrogatePVTAttention(SurrogateModule, PVTAttention):
    def forward(self, x, feat_size):
        B, N, C = x.shape
        H, W = feat_size
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        if self.pool is not None:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            x = self.sr(self.pool(x)).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            x = self.act(x)
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, self.head_dim)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            if self.sr is not None:
                x = x.permute(0, 2, 1).reshape(B, C, H, W)
                x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
                x = self.norm(x)
                kv = (
                    self.kv(x)
                    .reshape(B, -1, 2, self.num_heads, self.head_dim)
                    .permute(2, 0, 3, 1, 4)
                )
            else:
                kv = (
                    self.kv(x)
                    .reshape(B, -1, 2, self.num_heads, self.head_dim)
                    .permute(2, 0, 3, 1, 4)
                )
        k, v = kv.unbind(0)

        if self.standard_backward:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)

            soft_attn = attn / self.temperature
            # soft_attn = soft_attn.detach() + (attn - attn.detach())
            # attn = attn.detach() + (soft_attn - soft_attn.detach())

            # attn = softmax_detach_norm_stable(attn, dim=-1, detach=True)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # soft_attn = softmax_detach_norm_stable(soft_attn, dim=-1, detach=True)
            soft_attn = soft_attn.softmax(dim=-1)
            soft_attn = self.attn_drop(soft_attn)

            attn = attn.detach()
            soft_attn = soft_attn.detach()

            x = attn @ v
            soft_x = soft_attn @ v

            x = soft_x - soft_x.detach() + x.detach()

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


SURROGATE_CLASS_MAP = {
    nn.ReLU: (SurrogateReLU, 0.6),
    nn.SiLU: (SurrogateSiLU, 1.6),
    nn.GELU: (SurrogateGELU, 1.0),
    nn.MaxPool2d: (SurrogateMaxPool2d, 0.3),
    LayerNorm2d: (SurrogateLayerNorm2d, None),
    nn.LayerNorm: (SurrogateLayerNorm, None),
    PVTAttention: (SurrogatePVTAttention, 1.0),
    nn.MultiheadAttention: (SurrogateMultiheadAttention, 1.0),
}
SURROGATE_BASE_CLASSES = tuple(SURROGATE_CLASS_MAP.keys())


def extract_base_class(module):
    # module can be a surrogate child class already
    for cls in SURROGATE_BASE_CLASSES:
        if isinstance(module, cls):
            return cls


def replace_modules_with_surrogates_(
    module,
    temperatures,
    standard_backward=False,
):
    for name, child in module.named_children():
        base_cls = extract_base_class(child)

        if base_cls in temperatures:
            if isinstance(child, SurrogateModule):
                child.temperature = temperatures[base_cls]
            else:
                surrogate_cls = SURROGATE_CLASS_MAP[base_cls][0]
                surrogate_cls.replace_class_with_surrogate(
                    child,
                    temperature=temperatures[base_cls],
                    standard_backward=standard_backward,
                )

        replace_modules_with_surrogates_(
            child,
            temperatures=temperatures,
            standard_backward=standard_backward,
        )


def set_standard_backward_in_surrogates_(
    module,
    classes,
    standard_backward=True,
):
    for name, child in module.named_children():
        base_cls = extract_base_class(child)

        if base_cls in classes:
            if isinstance(child, SurrogateModule):
                child.standard_backward = standard_backward

        set_standard_backward_in_surrogates_(
            child,
            classes=classes,
            standard_backward=standard_backward,
        )


def soften_module_inplace_(
    module,
    temperatures=None,
    standard_backward=True,
    fill_default_temperatures=True,
):
    if temperatures is None:
        temperatures = {}

    if fill_default_temperatures:
        temperatures = {
            key: val[1] for key, val in SURROGATE_CLASS_MAP.items()
        } | temperatures

    replace_modules_with_surrogates_(
        module,
        temperatures=temperatures,
        standard_backward=standard_backward,
    )


def set_module_standard_backward_(
    module,
    standard_backward=True,
):
    set_standard_backward_in_surrogates_(
        module,
        classes=SURROGATE_BASE_CLASSES,
        standard_backward=standard_backward,
    )
