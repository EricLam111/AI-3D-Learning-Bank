"""The real acceptance test: your gradients must match torch.autograd to 1e-6.

The same lambda from expressions.py is evaluated twice -- once on your Tensor, once on a torch
tensor -- so any arithmetic discrepancy is unambiguously your bug.
"""
import numpy as np
import pytest

from expressions import EXPRESSIONS, make_inputs
from micrograd import Tensor

torch = pytest.importorskip("torch", reason="install torch to run the parity suite")


@pytest.mark.parametrize("name", list(EXPRESSIONS))
def test_forward_and_backward_match_torch(name):
    fn = EXPRESSIONS[name]
    x_np, y_np = make_inputs(seed=1)

    # --- your engine
    x, y = Tensor(x_np.copy()), Tensor(y_np.copy())
    out = fn(x, y)
    mine_value = float(out.data)
    out.backward()

    # --- torch, float64 so the 1e-6 tolerance is a fair comparison
    tx = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    ty = torch.tensor(y_np, dtype=torch.float64, requires_grad=True)
    tout = fn(tx, ty)
    torch_value = float(tout.item())
    tout.backward()

    assert np.isclose(mine_value, torch_value, rtol=1e-9, atol=1e-9), (
        f"forward value differs for {name!r}: mine={mine_value!r} torch={torch_value!r}"
    )
    # Some expressions ignore `y` entirely -- torch reports grad=None for unused inputs, so treat
    # "unused" as "gradient must be exactly zero" rather than skipping the check.
    for label, mine, reference in (("x", x, tx), ("y", y, ty)):
        if reference.grad is None:
            assert np.allclose(mine.grad, 0.0), f"{label}.grad should stay zero in {name!r}"
            continue
        np.testing.assert_allclose(
            mine.grad,
            reference.grad.numpy(),
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"{label}.grad mismatch in {name!r}",
        )
