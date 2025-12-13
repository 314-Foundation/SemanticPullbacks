import math
from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TwoWayReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, temperature=1.0, act="relu"):
        ctx.save_for_backward(z)
        ctx.temperature = temperature
        ctx.act = act

        if act == "relu":
            return F.relu(z)
        elif act == "gelu":
            return F.gelu(z)
        elif act == "silu":
            return F.silu(z)
        else:
            raise Exception(f"Wrong act: {act}")

    @staticmethod
    def backward(ctx, grad_output):
        (z,) = ctx.saved_tensors
        temp = ctx.temperature
        act = ctx.act

        if act == "gelu":
            make_gate = lambda temp: 0.5 * (1 + torch.erf(z / (math.sqrt(2) * temp)))
            # gate = 0.5 * (1 + torch.erf(z / (math.sqrt(2) * temp)))
            # gate = torch.distributions.Normal(0, 1).cdf(z / temp)
        else:
            make_gate = lambda temp: F.sigmoid(z / temp)
            # gate = F.sigmoid(z / temp)

        # make_gate = lambda temp: 0.5 * (1 + torch.erf(z / (math.sqrt(2) * temp)))
        # gate = torch.where(z > 0, make_gate(temp), make_gate(1.0))
        gate = make_gate(temp)

        return grad_output * gate, None, None


class TwoWayReLU(nn.Module):
    def __init__(self, temperature=1.0, act="relu"):
        super().__init__()
        self.temperature = temperature
        self.act = act

    def forward(self, x):
        return TwoWayReLUFunction.apply(x, self.temperature, self.act)

    def extra_repr(self):
        return f"temperature={self.temperature}, act={self.act}"


class SoftMaxPool2d(nn.MaxPool2d):
    def __init__(self, *args, temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

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
        pooled = (x_unf * weights).sum(dim=2)

        # Reshape back to image
        out_H = (H + 2 * self.padding - kH) // self.stride + 1
        out_W = (W + 2 * self.padding - kW) // self.stride + 1
        return pooled.view(B, C, out_H, out_W)

    def extra_repr(self):
        ret = super().extra_repr()
        return f"{ret}, temperature={self.temperature}"


class SurrogateSoftMaxPool2d(SoftMaxPool2d):
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


class NormalCDF(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        return 0.5 * (1 + torch.erf(x / (math.sqrt(2) * self.temperature)))

    def extra_repr(self):
        return f"temperature={self.temperature}"


class SurrogateIdentity(nn.Module):
    # https://github.com/AdaptiveAILab/fgi
    def __init__(self, backward_module=None):
        super().__init__()
        if backward_module is None:
            backward_module = NormalCDF()

        self.backward_module = backward_module

    def forward(self, x):
        mul = x * self.backward_module(x).detach()
        y = mul - mul.detach() + x.detach()

        return y


class SurrogateLayerNorm(nn.Module):
    # class SurrogateLayerNorm(nn.LayerNorm):
    """
    Pure-Python drop-in replacement for torch.nn.LayerNorm.

    Matches:
    - API:  normalized_shape, eps, elementwise_affine, bias, device, dtype
    - Behavior: mean/var over last D dims, biased variance (unbiased=False)
    - Parameter names/shapes: weight, bias with shape `normalized_shape`
    """

    def __init__(
        self,
        normalized_shape: Union[int, Iterable[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        weight=None,
        bias=None,
        # device: Optional[torch.device] = None,
        # dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        elif isinstance(normalized_shape, torch.Size):
            normalized_shape = tuple(normalized_shape)
        else:
            normalized_shape = tuple(normalized_shape)

        self.normalized_shape: Tuple[int, ...] = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self._use_bias = bias is not None  # only relevant if elementwise_affine=True

        self.weight = weight
        self.bias = bias
        # self.register_parameter("weight", weight)
        # self.register_parameter("bias", bias)

    def forward(self, x: Tensor) -> Tensor:
        # Normalize over the last D dimensions, where
        # D = len(normalized_shape), exactly like nn.LayerNorm. :contentReference[oaicite:0]{index=0}
        D = len(self.normalized_shape)
        dims = tuple(range(x.dim() - D, x.dim()))

        mean = x.mean(dim=dims, keepdim=True)
        # IMPORTANT: use biased variance (unbiased=False), same as LayerNorm. :contentReference[oaicite:1]{index=1}
        var = x.var(dim=dims, keepdim=True, unbiased=False)

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

    def extra_repr(self) -> str:
        s = (
            f"normalized_shape={self.normalized_shape}, "
            f"eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )
        # Match nn.LayerNorm’s constructor options (bias only matters if affine)
        if self.elementwise_affine:
            s += f", bias={self._use_bias}"
        return s


class SurrogateLayerNorm2d(SurrogateLayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x


class DetachedAttention(nn.MultiheadAttention):
    def forward(self, query, key, value, *args, **kwargs):
        if not self.training:
            # pass
            # query = query.detach()
            # key = key.detach()
            # value = value.detach()
            orig, attn_weights = super().forward(query, key, value, *args, **kwargs)

            temp_query = 1.3
            temp_key = 1.3
            squery = query / temp_query
            skey = key / temp_key
            squery = squery.detach() + (query - query.detach())
            skey = skey.detach() + (key - key.detach())

            softer, _ = super().forward(squery, skey, value, *args, **kwargs)

            # mul = query / temp  # .detach()
            ret = softer - softer.detach() + orig.detach()
            return ret, attn_weights
            # query = mul - mul.detach() + query.detach()
            # query = mul

            # mul = key / temp  # .detach()
            # key = mul - mul.detach() + key.detach()
            # key = mul

        return super().forward(query, key, value, *args, **kwargs)
