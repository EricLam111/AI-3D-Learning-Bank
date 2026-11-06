"""Finite-difference gradient check. No torch, no labels -- just calculus as ground truth.

For every expression in expressions.py, each element of x and y is perturbed by +/- eps and the
central difference (f(+eps) - f(-eps)) / (2*eps) is compared to your analytic gradient.

At eps=1e-6 you should agree to about 1e-5. If you see agreement to 1e-3 but not 1e-5, suspect a
transpose or an missing unbroadcast, not floating point.
"""
import numpy as np
import pytest

from expressions import EXPRESSIONS, make_inputs
from micrograd import Tensor

EPS = 1e-6
RTOL = 1e-5
ATOL = 1e-6


def _forward(fn, x_np, y_np) -> float:
    out = fn(Tensor(x_np.copy()), Tensor(y_np.copy()))
    assert out.shape == (), f"expression must reduce to a scalar, got shape {out.shape}"
    return float(out.data)


@pytest.mark.parametrize("name", list(EXPRESSIONS))
def test_analytic_matches_numeric(name):
    fn = EXPRESSIONS[name]
    x_np, y_np = make_inputs(seed=3)

    x, y = Tensor(x_np.copy()), Tensor(y_np.copy())
    fn(x, y).backward()

    for label, base, tensor in (("x", x_np, x), ("y", y_np, y)):
        numeric = np.zeros_like(base)
        for idx in np.ndindex(base.shape):
            orig = base[idx]

            base[idx] = orig + EPS
            plus = _forward(fn, x_np, y_np)

            base[idx] = orig - EPS
            minus = _forward(fn, x_np, y_np)

            base[idx] = orig
            numeric[idx] = (plus - minus) / (2 * EPS)

        assert tensor.grad.shape == base.shape, f"{label}.grad has the wrong shape"
        np.testing.assert_allclose(
            tensor.grad,
            numeric,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"gradient mismatch for {label} in expression {name!r}",
        )
