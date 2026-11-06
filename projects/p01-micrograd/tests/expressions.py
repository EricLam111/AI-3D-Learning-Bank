"""Shared expression set used by BOTH the gradcheck and the torch-parity suites.

Every expression must be written using only the op surface in README.md, so the exact same
lambda can be evaluated by micrograd, by torch, and by central finite differences.
"""
import numpy as np

X_SHAPE = (3, 4)
Y_SHAPE = (4, 3)


def make_inputs(seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=X_SHAPE), rng.normal(size=Y_SHAPE)


EXPRESSIONS = {
    "relu_add": lambda x, y: ((x @ y).relu() + x @ y).sum(),
    "exp_product": lambda x, y: (x.exp() * x + x).mean(),
    "exp_log_chain": lambda x, y: (x.exp().log() ** 2).sum(),
    "relu_times": lambda x, y: ((x @ y).relu() * (x @ y)).sum(),
    "division": lambda x, y: (x / (x.exp() + 1.0)).sum(),
    "log_square": lambda x, y: ((x ** 2 + 1.0).log() * y.sum()).sum(),
    "transpose_mm": lambda x, y: (x.T @ y.T).relu().sum(),
    "reshape_view": lambda x, y: (x.reshape(2, 6).relu().sum() + y.mean()),
}
