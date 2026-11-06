"""Micro-autograd engine. Reverse-mode autodiff over NumPy arrays.

Only the scaffolding and ONE worked example (__add__) are provided.
Your job: implement every operator marked TODO so that gradients match torch.autograd.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Tensor"]


def _as_tensor(value) -> "Tensor":
    """Wrap plain numbers/arrays so that `x + 1` and `1 + x` both work."""
    if isinstance(value, Tensor):
        return value
    return Tensor(value, requires_grad=False)


class Tensor:
    """A node in the computation graph.

    Attributes
    ----------
    data : np.ndarray, float64
    grad : np.ndarray, float64, same shape as data. Accumulated, never assigned.
    requires_grad : bool
    """

    # Keeps numpy from hijacking `np_array + our_tensor` and returning an ndarray.
    __array_priority__ = 1000
    # Float64 by default: 1e-6 parity tolerances are meaningless in float32.
    __array_ufunc__ = None

    def __init__(self, data, requires_grad: bool = True, _children=(), _op: str = ""):
        self.data = np.asarray(data, dtype=np.float64)
        self.grad = np.zeros_like(self.data)
        self.requires_grad = requires_grad
        # Placeholder: each op fills this in with a closure that pushes grad into parents.
        self._backward = lambda: None
        # Parents in the graph. A set is fine here; backward() dedupes via `visited`.
        self._prev = set(_children)
        self._op = _op

    # ---------------------------------------------------------------- utilities

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self) -> str:
        g = "None" if self.grad is None else np.array2string(self.grad, precision=2)
        return f"Tensor(shape={self.shape}, op={self._op!r}, data={self.data}, grad={g})"

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    @staticmethod
    def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
        """Undo NumPy broadcasting: sum `grad` back down to `shape`.

        Forward: (3,1) + (1,4) -> (3,4). Backward: each input needs its grad summed over
        the axes it was broadcast along, keeping a length-1 axis as length 1.
        """
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)
        for axis, dim in enumerate(shape):
            if dim == 1 and grad.shape[axis] != 1:
                grad = grad.sum(axis=axis, keepdims=True)
        return grad

    # ----------------------------------------------------------------- backward

    def backward(self) -> None:
        """Seed d(self)/d(self) = 1 and propagate gradients in reverse topological order."""
        # Build a topological order of the graph reachable from `self` (iterative DFS).
        topo, visited, stack = [], set(), [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._prev:
                if id(parent) not in visited:
                    stack.append((parent, False))

        # Seed: if `self` is a scalar loss, d loss / d loss = 1.
        self.grad = np.ones_like(self.data)

        # Walk in reverse order: every node's grad is complete by the time we call its closure.
        for node in reversed(topo):
            node._backward()

    # --------------------------------------------------------- worked example

    def __add__(self, other) -> "Tensor":
        """out = self + other.

        THE PATTERN (read once, then apply everywhere):

        1. compute the forward value, recording the parents;
        2. write a closure `_backward` that reads `out.grad` and pushes the local
           derivative times the incoming gradient into each parent with `+=`;
        3. store it on `out`.

        The local derivative of addition is 1 with respect to each input, so each parent
        receives exactly the incoming gradient -- but unbroadcast back to its own shape.
        """
        other = _as_tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.grad += Tensor._unbroadcast(out.grad, self.data.shape)
            other.grad += Tensor._unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other) -> "Tensor":
        return self + other

    # ------------------------------------------------------------ TODO: ops

    def __mul__(self, other) -> "Tensor":
        # TODO: local derivative is the *other* operand (then unbroadcast).
        raise NotImplementedError("Tensor.__mul__")

    def __rmul__(self, other) -> "Tensor":
        return self * other

    def __sub__(self, other) -> "Tensor":
        # TODO: hint -- write it as self + (-other), or derive it directly.
        raise NotImplementedError("Tensor.__sub__")

    def __rsub__(self, other) -> "Tensor":
        return _as_tensor(other) - self

    def __truediv__(self, other) -> "Tensor":
        # TODO: d(a/b)/da = 1/b, d(a/b)/db = -a/b**2
        raise NotImplementedError("Tensor.__truediv__")

    def __rtruediv__(self, other) -> "Tensor":
        return _as_tensor(other) / self

    def __pow__(self, exponent) -> "Tensor":
        # TODO: only scalar exponents are required. d/dx x**k = k * x**(k-1)
        raise NotImplementedError("Tensor.__pow__")

    def __neg__(self) -> "Tensor":
        # TODO
        raise NotImplementedError("Tensor.__neg__")

    def __matmul__(self, other) -> "Tensor":
        # TODO: out = A @ B. dA = out.grad @ B.T ; dB = A.T @ out.grad
        # (For 2-D only. If you want batch dims, that's optional extra.)
        raise NotImplementedError("Tensor.__matmul__")

    def exp(self) -> "Tensor":
        # TODO: d/dx e**x = e**x -- reuse the forward result, don't recompute.
        raise NotImplementedError("Tensor.exp")

    def log(self) -> "Tensor":
        # TODO: d/dx ln(x) = 1/x
        raise NotImplementedError("Tensor.log")

    def relu(self) -> "Tensor":
        # TODO: local derivative is 1 where input > 0, else 0.
        raise NotImplementedError("Tensor.relu")

    def sum(self) -> "Tensor":
        # TODO: reduce to a scalar (shape ()). Every input element receives the same grad.
        raise NotImplementedError("Tensor.sum")

    def mean(self) -> "Tensor":
        # TODO: like sum(), but scaled by 1/N where N = data.size.
        raise NotImplementedError("Tensor.mean")

    def reshape(self, *shape) -> "Tensor":
        # TODO: accept reshape(2, 3) and reshape((2, 3)). Backward is the inverse view.
        raise NotImplementedError("Tensor.reshape")

    @property
    def T(self) -> "Tensor":
        # TODO: transpose (swap the last two axes). Backward is the same transpose.
        raise NotImplementedError("Tensor.T")
