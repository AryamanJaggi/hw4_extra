"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


def _prod(xs):
    out = 1
    for x in xs:
        out *= x
    return out


def _normalize_axes(axes, ndim):
    """Return a sorted tuple of non-negative axes (all axes if axes is None)."""
    if axes is None:
        return tuple(range(ndim))
    if isinstance(axes, int):
        axes = (axes,)
    return tuple(sorted(set(a % ndim for a in axes)))


def _reduce(a: NDArray, axes, op: str, keepdims: bool = False) -> NDArray:
    """Reduce (sum / max) an NDArray over an arbitrary set of axes."""
    axes = _normalize_axes(axes, a.ndim)
    if BACKEND == "np":
        fn = array_api.sum if op == "sum" else array_api.max
        return fn(a, axis=axes, keepdims=keepdims)
    if len(axes) == 0:
        return a
    kept = tuple(i for i in range(a.ndim) if i not in axes)
    kept_shape = tuple(a.shape[i] for i in kept)
    red = _prod(a.shape[i] for i in axes)
    if len(axes) == 1:
        view = a
        axis = axes[0]
    else:
        view = a.permute(kept + axes).compact().reshape(kept_shape + (red,))
        axis = len(kept_shape)
    out = view.sum(axis=axis) if op == "sum" else view.max(axis=axis)
    if keepdims:
        out = out.reshape(tuple(1 if i in axes else s for i, s in enumerate(a.shape)))
    else:
        out = out.reshape(kept_shape)
    return out


def _sum_to_shape(t: Tensor, shape) -> Tensor:
    """Sum a (broadcasted) tensor back down to `shape`."""
    shape = tuple(shape)
    if t.shape == shape:
        return t
    lead = len(t.shape) - len(shape)
    axes = list(range(lead))
    for i, s in enumerate(shape):
        if s == 1 and t.shape[lead + i] != 1:
            axes.append(lead + i)
    if axes:
        t = summation(t, axes=tuple(axes))
    return reshape(t, shape)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        if BACKEND == "np":
            return a ** b
        return array_api.exp(array_api.log(a) * b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad * b * power(a, b - 1), out_grad * node * log(a)
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        if self.scalar == 0:
            return out_grad * 0
        return out_grad * self.scalar * power_scalar(a, self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, -out_grad * a / (b * b)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def _perm(self, ndim):
        perm = list(range(ndim))
        if self.axes is None:
            i, j = ndim - 2, ndim - 1
        else:
            i, j = self.axes
        perm[i], perm[j] = perm[j], perm[i]
        return tuple(perm)

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        perm = self._perm(a.ndim)
        if BACKEND == "np":
            return array_api.transpose(a, perm)
        return a.permute(perm)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if BACKEND == "np":
            return array_api.reshape(a, self.shape)
        return a.compact().reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if BACKEND == "np":
            return array_api.broadcast_to(a, self.shape)
        return a.broadcast_to(self.shape).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return _sum_to_shape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return _reduce(a, self.axes, "sum")
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        axes = _normalize_axes(self.axes, len(in_shape))
        keep_shape = tuple(1 if i in axes else s for i, s in enumerate(in_shape))
        return broadcast_to(reshape(out_grad, keep_shape), in_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        if BACKEND == "np" or (a.ndim == 2 and b.ndim == 2):
            return a @ b
        if b.ndim == 2:
            # (..., m, n) @ (n, p): fold batch into rows
            out = a.compact().reshape((_prod(a.shape[:-1]), a.shape[-1])) @ b
            return out.reshape(a.shape[:-1] + (b.shape[-1],))
        # general batched matmul with broadcasting of batch dims
        batch = tuple(numpy.broadcast_shapes(a.shape[:-2], b.shape[:-2]))
        m, n, p = a.shape[-2], a.shape[-1], b.shape[-1]
        a_b = a.broadcast_to(batch + (m, n))
        b_b = b.broadcast_to(batch + (n, p))
        nb = _prod(batch)
        a_b = a_b.compact().reshape((nb, m, n))
        b_b = b_b.compact().reshape((nb, n, p))
        out = array_api.empty((nb, m, p), device=a.device)
        for i in range(nb):
            out[i, :, :] = (a_b[i, :, :].compact().reshape((m, n)) @
                            b_b[i, :, :].compact().reshape((n, p)))
        return out.reshape(batch + (m, p))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        ga = matmul(out_grad, transpose(b))
        gb = matmul(transpose(a), out_grad)
        return _sum_to_shape(ga, a.shape), _sum_to_shape(gb, b.shape)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * node
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        mask = node.realize_cached_data() > 0
        return out_grad * Tensor.make_const(mask)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (-(node * node) + 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Sigmoid(TensorOp):
    def compute(self, a):
        return (array_api.exp(-a) + 1) ** -1

    def gradient(self, out_grad, node):
        return out_grad * node * (-node + 1)


def sigmoid(a):
    return Sigmoid()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if BACKEND == "np":
            return array_api.stack(args, axis=self.axis)
        n = len(args)
        shape = args[0].shape
        for x in args:
            assert x.shape == shape, "All arrays need to be of the same size"
        new_shape = shape[: self.axis] + (n,) + shape[self.axis:]
        out = array_api.empty(new_shape, device=args[0].device)
        idx = [slice(None)] * len(new_shape)
        for i, x in enumerate(args):
            idx[self.axis] = i
            out[tuple(idx)] = x
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        n = A.shape[self.axis]
        new_shape = A.shape[: self.axis] + A.shape[self.axis + 1:]
        if BACKEND == "np":
            return tuple(array_api.take(A, i, axis=self.axis) for i in range(n))
        idx = [slice(None)] * A.ndim
        outs = []
        for i in range(n):
            idx[self.axis] = i
            outs.append(A[tuple(idx)].compact().reshape(new_shape))
        return tuple(outs)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if BACKEND == "np":
            return array_api.flip(a, self.axes)
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a
        axes = [ax for ax in self.axes if ax < a.ndim]
        new_shape = list(a.shape)
        for ax in axes:
            new_shape[ax] *= self.dilation + 1
        out = array_api.full(tuple(new_shape), 0.0, device=a.device)
        idx = [slice(None)] * a.ndim
        for ax in axes:
            idx[ax] = slice(0, new_shape[ax], self.dilation + 1)
        out[tuple(idx)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a
        idx = [slice(None)] * a.ndim
        for ax in self.axes:
            if ax < a.ndim:
                idx[ax] = slice(0, a.shape[ax], self.dilation + 1)
        return a[tuple(idx)].compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class _Slice(TensorOp):
    """Take a contiguous slice [start, stop) along each axis (used internally by conv)."""

    def __init__(self, bounds: tuple):
        self.bounds = tuple(bounds)

    def compute(self, a):
        return a[tuple(slice(lo, hi) for lo, hi in self.bounds)].compact()

    def gradient(self, out_grad, node):
        in_shape = node.inputs[0].shape
        return _Pad(tuple((lo, s - hi) for (lo, hi), s in zip(self.bounds, in_shape)))(out_grad)


class _Pad(TensorOp):
    def __init__(self, pads: tuple):
        self.pads = tuple(pads)

    def compute(self, a):
        return a.pad(self.pads)

    def gradient(self, out_grad, node):
        return _Slice(tuple((lo, lo + s) for (lo, _), s in zip(self.pads, node.inputs[0].shape)))(
            out_grad
        )


def _crop_to(t: Tensor, shape) -> Tensor:
    if tuple(t.shape) == tuple(shape):
        return t
    return _Slice(tuple((0, s) for s in shape))(t)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # A: N x H x W x C_in (NHWC);  B: K x K x C_in x C_out
        p, s = self.padding, self.stride
        if p > 0:
            A = A.pad(((0, 0), (p, p), (p, p), (0, 0)))
        A = A.compact()
        N, H, W, C = A.shape
        K, K2, C_in, C_out = B.shape
        assert C == C_in, "input channels mismatch"
        Ho = (H - K) // s + 1
        Wo = (W - K2) // s + 1
        Ns, Hs, Ws, Cs = A.strides
        inner = K * K2 * C_in
        Z = A.as_strided((N, Ho, Wo, K, K2, C_in), (Ns, Hs * s, Ws * s, Hs, Ws, Cs))
        Z = Z.compact().reshape((N * Ho * Wo, inner))
        out = Z @ B.compact().reshape((inner, C_out))
        return out.reshape((N, Ho, Wo, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, Wt = node.inputs
        N, H, W, C_in = X.shape
        K, _, _, C_out = Wt.shape
        p, s = self.padding, self.stride
        # undo the striding, and crop to the "stride-1" output size H + 2p - K + 1
        G = out_grad
        if s > 1:
            G = dilate(G, (1, 2), s - 1)
        G = _crop_to(G, (N, H + 2 * p - K + 1, W + 2 * p - K + 1, C_out))

        # dX = conv(G, flip(W) with in/out channels swapped), padding K - 1 - p
        W_flip = transpose(flip(Wt, (0, 1)), (2, 3))  # K x K x C_out x C_in
        X_grad = conv(G, W_flip, stride=1, padding=K - 1 - p)

        # dW = conv(X^T, G^T): treat C_in as batch, N as channels
        X_t = transpose(X, (0, 3))  # C_in x H x W x N
        G_t = transpose(transpose(G, (0, 1)), (1, 2))  # H' x W' x N x C_out
        W_grad = conv(X_t, G_t, stride=1, padding=p)  # C_in x K x K x C_out
        W_grad = transpose(transpose(W_grad, (0, 1)), (1, 2))  # K x K x C_in x C_out
        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
