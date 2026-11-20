"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # weight before bias: mugrade's expected values assume that draw order
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype)
        )
        if bias:
            # note fan_in is out_features here, not in_features
            self.bias = Parameter(
                init.kaiming_uniform(
                    out_features, 1, device=device, dtype=dtype
                ).reshape((out_features,))
            )
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = X @ self.weight
        if self.bias is not None:
            # needle has no implicit broadcasting, so do it explicitly
            out = out + self.bias.broadcast_to(out.shape)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        size = 1
        for dim in X.shape[1:]:
            size *= dim
        return X.reshape((X.shape[0], size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, num_classes = logits.shape
        # pick out z_y with a one-hot mask, since needle has no gather op
        y_one_hot = init.one_hot(num_classes, y, device=logits.device, dtype="float32")
        z_y = ops.summation(logits * y_one_hot, axes=(1,))
        losses = ops.logsumexp(logits, axes=(1,)) - z_y
        return ops.summation(losses) / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        # running stats are buffers, not Parameters: optimizers must not see them
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, feat = x.shape
        weight = self.weight.reshape((1, feat)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, feat)).broadcast_to(x.shape)

        if self.training:
            # statistics over the batch dimension
            mean = ops.summation(x, axes=(0,)) / batch_size
            centered = x - mean.reshape((1, feat)).broadcast_to(x.shape)
            var = ops.summation(centered * centered, axes=(0,)) / batch_size

            # update the running estimates off-graph so the optimizer memory
            # checks don't see an ever-growing computational graph
            self.running_mean = (
                (1 - self.momentum) * self.running_mean.data
                + self.momentum * mean.data
            ).data
            self.running_var = (
                (1 - self.momentum) * self.running_var.data
                + self.momentum * var.data
            ).data

            denom = ops.power_scalar(
                var.reshape((1, feat)).broadcast_to(x.shape) + self.eps, 0.5
            )
            normed = centered / denom
        else:
            mean = self.running_mean.reshape((1, feat)).broadcast_to(x.shape)
            var = self.running_var.reshape((1, feat)).broadcast_to(x.shape)
            normed = (x - mean) / ops.power_scalar(var + self.eps, 0.5)

        return weight * normed + bias
        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, feat = x.shape
        # statistics over the feature dimension, one set per sample
        mean = (ops.summation(x, axes=(1,)) / feat).reshape((batch_size, 1))
        centered = x - mean.broadcast_to(x.shape)
        # biased variance: divide by N, not N - 1
        var = (ops.summation(centered * centered, axes=(1,)) / feat).reshape(
            (batch_size, 1)
        )
        normed = centered / ops.power_scalar(var.broadcast_to(x.shape) + self.eps, 0.5)

        weight = self.weight.reshape((1, feat)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, feat)).broadcast_to(x.shape)
        return weight * normed + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x
        # randb(p=1 - self.p) marks the units that are *kept*; scaling by
        # 1/(1 - p) keeps the expected activation unchanged
        mask = init.randb(*x.shape, p=1 - self.p, device=x.device, dtype="float32")
        return x * mask / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
