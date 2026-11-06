"""Spec tests: shape, dtype, broadcasting and accumulation semantics.

No torch needed -- the reference here is plain NumPy arithmetic you can check by hand.
These tests will fail with NotImplementedError until you implement the operators.
"""
import numpy as np
import pytest

from micrograd import Tensor


# ------------------------------------------------------------------ basics


def test_dtype_shape_and_zero_grad():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t.shape == (2, 2)
    assert t.data.dtype == np.float64
    assert np.allclose(t.grad, 0.0)
    t.grad = np.ones((2, 2))
    t.zero_grad()
    assert np.allclose(t.grad, 0.0)


def test_scalar_and_reflected_operators():
    x = Tensor([1.0, 2.0])

    out = x + 1.0
    assert isinstance(out, Tensor)
    assert np.allclose(out.data, [2.0, 3.0])

    assert np.allclose((1.0 + x).data, [2.0, 3.0])
    assert np.allclose((1.0 - x).data, [0.0, -1.0])
    assert np.allclose((2.0 * x).data, [2.0, 4.0])
    assert np.allclose((2.0 / x).data, [2.0, 1.0])


# ------------------------------------------------------------- broadcasting


def test_broadcast_forward_and_backward():
    x = Tensor(np.arange(3.0).reshape(3, 1))  # (3,1)
    y = Tensor(np.arange(4.0).reshape(1, 4))  # (1,4)

    out = x + y
    assert out.shape == (3, 4)

    out.mean().backward()

    # Each of the 12 inputs gets 1/12 of the mean's gradient, summed over the broadcast axis.
    assert x.grad.shape == (3, 1)
    assert y.grad.shape == (1, 4)
    assert np.allclose(x.grad, 4 / 12)
    assert np.allclose(y.grad, 3 / 12)


def test_grad_accumulates_when_a_tensor_is_reused():
    x = Tensor([2.0, -3.0])
    (x * x + x).sum().backward()
    # d/dx (x^2 + x) = 2x + 1
    assert np.allclose(x.grad, 2 * np.array([2.0, -3.0]) + 1)


def test_long_chain_accumulates():
    x = Tensor([1.0])
    y = x
    for _ in range(5):
        y = y + x  # x appears 6 times in total
    y.sum().backward()
    assert np.allclose(x.grad, 6.0)


# ---------------------------------------------------------------- reductions


def test_reductions_return_scalars():
    x = Tensor([[1.0, 2.0], [3.0, 4.0]])

    s = x.sum()
    assert s.shape == ()
    assert np.isclose(float(s.data), 10.0)

    m = x.mean()
    assert m.shape == ()
    assert np.isclose(float(m.data), 2.5)

    m.backward()
    assert np.allclose(x.grad, 1.0 / 4.0)


# ------------------------------------------------------------- view ops


def test_reshape_and_transpose_match_numpy():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(2, 6))
    t = Tensor(a)

    assert np.allclose(t.reshape(3, 4).data, a.reshape(3, 4))
    assert np.allclose(t.reshape((3, 4)).data, a.reshape(3, 4))
    assert np.allclose(t.T.data, a.T)


def test_view_ops_backward():
    a = np.array([[1.0, -1.0], [2.0, -2.0]])

    t = Tensor(a)
    t.T.sum().backward()
    assert np.allclose(t.grad, 1.0)

    u = Tensor(a)
    u.reshape(4).relu().sum().backward()
    assert np.allclose(u.grad.reshape(4), (a.reshape(4) > 0).astype(float))


# ------------------------------------------------------- elementwise ops


def test_relu_masks_negatives():
    x = Tensor([-2.0, 0.5])
    x.relu().sum().backward()
    assert np.allclose(x.grad, [0.0, 1.0])


def test_exp_log_roundtrip_gradient():
    x = Tensor([0.3, 1.2])
    x.exp().log().sum().backward()
    assert np.allclose(x.grad, 1.0)


def test_pow_and_neg():
    x = Tensor([2.0, -3.0])
    (x**2).sum().backward()
    assert np.allclose(x.grad, [4.0, -6.0])

    y = Tensor([1.0, -1.0])
    (-y).sum().backward()
    assert np.allclose(y.grad, [-1.0, -1.0])


def test_division():
    a = Tensor([4.0, 9.0])
    b = Tensor([2.0, 3.0])
    (a / b).sum().backward()

    assert np.allclose(a.grad, 1 / np.array([2.0, 3.0]))
    assert np.allclose(b.grad, -np.array([4.0, 9.0]) / np.array([2.0, 3.0]) ** 2)


# --------------------------------------------------------------- matmul


def test_matmul_backward_shapes_and_values():
    rng = np.random.default_rng(1)
    a_np = rng.normal(size=(3, 4))
    b_np = rng.normal(size=(4, 5))

    a, b = Tensor(a_np), Tensor(b_np)
    (a @ b).sum().backward()

    assert a.grad.shape == (3, 4)
    assert b.grad.shape == (4, 5)
    # d sum(A@B) / dA = ones(3,5) @ B.T   and   dB = A.T @ ones(3,5)
    assert np.allclose(a.grad, np.ones((3, 5)) @ b_np.T)
    assert np.allclose(b.grad, a_np.T @ np.ones((3, 5)))


# ------------------------------------------------------------- nn layers


def test_linear_layer_shapes_and_grads():
    from micrograd.nn import Linear

    layer = Linear(4, 2, seed=0)
    x = Tensor(np.random.default_rng(2).normal(size=(5, 4)))

    out = layer(x)
    assert out.shape == (5, 2)

    out.sum().backward()
    assert x.grad.shape == (5, 4)
    assert layer.W.grad.shape == (4, 2)
    assert layer.b.grad.shape == (2,)
    # Bias grad must be summed over the batch -- this is what _unbroadcast is for.
    assert np.allclose(layer.b.grad, 5.0)


# ------------------------------------------------------------ gradient descent


def test_gradient_descent_reaches_the_target():
    """If the gradients point downhill, plain GD must converge. No torch needed."""
    rng = np.random.default_rng(0)
    w = Tensor(rng.normal(size=(3, 1)))
    target = np.array([[1.0], [-2.0], [0.5]])

    losses = []
    for _ in range(200):
        w.zero_grad()
        loss = ((w - target) ** 2).mean()
        losses.append(float(loss.data))
        loss.backward()
        w.data -= 0.3 * w.grad

    assert losses[-1] < losses[0] * 1e-4, f"loss barely moved: {losses[0]} -> {losses[-1]}"
    assert np.allclose(w.data, target, atol=1e-2)
