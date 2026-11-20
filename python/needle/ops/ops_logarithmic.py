from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *
from .ops_mathematic import _reduce, _normalize_axes

from ..backend_selection import array_api, BACKEND

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # log-softmax over the last axis
        m = _reduce(Z, (Z.ndim - 1,), "max", keepdims=True)
        shifted = Z - array_api.broadcast_to(m, Z.shape)
        lse = array_api.log(_reduce(array_api.exp(shifted), (Z.ndim - 1,), "sum", keepdims=True))
        return shifted - array_api.broadcast_to(lse, Z.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        shape = node.shape
        keep = shape[:-1] + (1,)
        softmax = exp(node)
        s = summation(out_grad, axes=(len(shape) - 1,)).reshape(keep).broadcast_to(shape)
        return out_grad - softmax * s
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        m_keep = _reduce(Z, self.axes, "max", keepdims=True)
        m = _reduce(Z, self.axes, "max", keepdims=False)
        s = _reduce(array_api.exp(Z - array_api.broadcast_to(m_keep, Z.shape)), self.axes, "sum")
        return array_api.log(s) + m
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        shape = Z.shape
        axes = _normalize_axes(self.axes, len(shape))
        keep = tuple(1 if i in axes else s for i, s in enumerate(shape))
        lse = node.realize_cached_data().reshape(keep)
        softmax = array_api.exp(Z.realize_cached_data() - array_api.broadcast_to(lse, shape))
        return out_grad.reshape(keep).broadcast_to(shape) * Tensor.make_const(softmax)
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)
