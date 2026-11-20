"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for p in self.params:
            if p.grad is None:
                continue
            # .data throughout: the update must not extend the graph that
            # produced the gradients, or memory grows every step
            grad = p.grad.data + self.weight_decay * p.data

            if p not in self.u:
                self.u[p] = ndl.init.zeros(
                    *p.shape, device=p.device, dtype=p.dtype, requires_grad=False
                )
            # u_{t+1} = beta * u_t + (1 - beta) * grad
            self.u[p] = (
                self.momentum * self.u[p] + (1 - self.momentum) * grad
            ).data

            p.data = p.data - self.lr * self.u[p]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        grads = [p.grad for p in self.params if p.grad is not None]
        if not grads:
            return
        total_norm = np.sqrt(sum(float((g.numpy().astype(np.float64) ** 2).sum()) for g in grads))
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in self.params:
                if p.grad is not None:
                    p.grad = (p.grad.data * clip_coef).data
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.data + self.weight_decay * p.data

            if p not in self.m:
                self.m[p] = ndl.init.zeros(
                    *p.shape, device=p.device, dtype=p.dtype, requires_grad=False
                )
                self.v[p] = ndl.init.zeros(
                    *p.shape, device=p.device, dtype=p.dtype, requires_grad=False
                )

            # first and second moment estimates
            self.m[p] = (self.beta1 * self.m[p] + (1 - self.beta1) * grad).data
            self.v[p] = (
                self.beta2 * self.v[p] + (1 - self.beta2) * grad * grad
            ).data

            # bias correction, using the 1-indexed step count
            m_hat = (self.m[p] / (1 - self.beta1 ** self.t)).data
            v_hat = (self.v[p] / (1 - self.beta2 ** self.t)).data

            update = m_hat / (ndl.ops.power_scalar(v_hat, 0.5) + self.eps)
            p.data = p.data - self.lr * update.data
        ### END YOUR SOLUTION
