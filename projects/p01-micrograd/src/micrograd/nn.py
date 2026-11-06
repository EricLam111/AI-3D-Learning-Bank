"""Micro neural-net layers built on the engine in tensor.py.

Signatures only. You implement the bodies using ONLY operations your Tensor supports.
"""
from __future__ import annotations

import numpy as np

from .tensor import Tensor

__all__ = ["Module", "Linear", "MLP"]


class Module:
    """Minimal container so you can iterate over parameters and zero their grads."""

    def parameters(self) -> list[Tensor]:
        return [p for _, p in self.named_parameters()]

    def named_parameters(self) -> list[tuple[str, Tensor]]:
        raise NotImplementedError("Module.named_parameters")

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Module):
    """y = x @ W + b

    W: (in_features, out_features), b: (out_features,)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, seed: int | None = None):
        # TODO: initialise W and b as Tensor leaf nodes wrapped in nn.Parameter-free storage,
        # i.e. just Tensor objects with requires_grad=True.
        # Recommended init: uniform(-1/sqrt(in_features), +1/sqrt(in_features)).
        # Keep the arrays as `self.W`, `self.b` (Tensor) and remember the shapes.
        raise NotImplementedError("Linear.__init__")

    def forward(self, x: Tensor) -> Tensor:
        # TODO: x @ W, then add b (broadcasting does this for free -- and _unbroadcast
        # in Tensor.__add__ is what makes b's gradient correct).
        raise NotImplementedError("Linear.forward")

    def named_parameters(self):
        return [("W", self.W), ("b", self.b)]


class MLP(Module):
    """A stack of Linear layers with ReLU between them and a raw linear output.

    Used by train_xor.py. For 2-D XOR data: 2 -> hidden -> 1.
    """

    def __init__(self, sizes: list[int], seed: int | None = None):
        # TODO: build self.layers = [Linear(sizes[i], sizes[i+1]) for i in range(len(sizes) - 1)]
        raise NotImplementedError("MLP.__init__")

    def forward(self, x: Tensor) -> Tensor:
        # TODO: relu after every layer except the last, in order.
        raise NotImplementedError

    def named_parameters(self):
        # TODO: flatten the layer parameters with unique names, e.g. "layer0.W".
        raise NotImplementedError
