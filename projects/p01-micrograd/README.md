# P1.1 — Micro-autograd engine (~10 h)

**Goal:** you build a reverse-mode autodiff engine small enough to hold in your head, then prove
it is correct by matching `torch.autograd` to 1e-6.

**Rule:** you may read this README, PyTorch docs, and Karpathy's micrograd as *reference reading*.
You may not copy working code into `src/micrograd/`. The `__add__` operator below is given to you
as a worked example of the pattern — every other op is yours to derive.

---

## What is given to you

| File | Status |
|---|---|
| `src/micrograd/tensor.py` | `Tensor.__init__`, `.shape`, `.backward()`, `._unbroadcast()`, `__add__` implemented. **Every other op raises `NotImplementedError`.** |
| `src/micrograd/nn.py` | Class signatures only (`Module`, `Linear`, `MLP`). Bodies are yours. |
| `tests/test_api.py` | Spec: shape/broadcast semantics vs pure NumPy. No torch needed. |
| `tests/test_gradcheck.py` | Spec: your analytic grads vs central finite differences (1e-5). |
| `tests/test_torch_parity.py` | Spec: your grads vs `torch.autograd` on the same graphs (1e-6). |
| `eval/acceptance.py` | The acceptance runner: runs all three suites, then trains XOR. Green here = project done. |

## The op surface you must implement

Same names as PyTorch, so the identical expression runs on both backends in the parity test.

```
dunder:  +  -  *  /  @  **  -x
methods: .exp()  .log()  .relu()  .sum()  .mean()  .reshape(*shape)  .T
attribute: .T  (swap last two axes)
```

Semantics required:
- float64 throughout (`np.float64`) — the 1e-6 tolerances are meaningless in float32.
- **Gradients accumulate with `+=`**, they never overwrite. This is what makes `x` used twice in an
  expression work, and the test suite checks it explicitly.
- Broadcasting is allowed forward (NumPy rules) and must be *reduced* in backward — that's what
  `_unbroadcast` is for.
- `.sum()` and `.mean()` reduce everything to a scalar (shape `()`).

## Acceptance criteria (from ROADMAP.md)

1. `python eval/acceptance.py` exits 0: all three test suites pass.
2. XOR is learned by an MLP built on **your** engine: final loss < 0.01, 4/4 predictions correct.
3. `NOTES.md` contains: what backprop is (plain-language paragraph), what surprised you, what broke.

## Definition of done — self-check before you tick it off

You should be able to answer these without looking anything up:

- Why does `backward()` need a topological order instead of just recursing?
- Why is `+=` in `_backward` and not `=`?
- In `x @ y`, why is `x.grad = out.grad @ y.data.T` and not the other way round?
- Why does `mean()` scale the incoming gradient by `1/N` while `sum()` does not?
- Where would you add `tanh`? What is its local gradient, and what does it saturate to?

## Suggested order (don't try to do all ops at once)

1. `*`, `-`, `**`, `neg` (trivial, gets you familiar with the closure pattern)
2. `exp`, `log`, `relu` (still elementwise, but non-polynomial local derivatives)
3. `sum`, `mean` (reduction: gradient has to be *broadcast back up*)
4. `reshape`, `.T` (view ops: gradient is the inverse view)
5. `@` (the only one with a real linear-algebra transpose argument)
6. `nn.py`: `Linear`, `MLP` — then watch XOR train

## Environment and commands (Windows / Ubuntu)

From the repository root, create the environment once with
`conda env create -f environment.yml`, then `conda activate ai-learn`.
See the root README for setup and GPU notes. Run these from this project directory:

```bash
python -m pytest -q tests/test_api.py       # API semantics
python -m pytest -q tests/test_gradcheck.py # finite differences
python -m pytest -q tests/test_torch_parity.py # PyTorch comparison
python eval/acceptance.py                   # complete acceptance
```

These commands work in Windows and Ubuntu without `make`. On systems with GNU Make,
the existing `make api`, `make gradcheck`, `make parity` and `make accept` shortcuts also work.
The unimplemented exercises intentionally fail until you complete them.


## Status when this scaffold was created

`python eval/acceptance.py` → exit 1, 14/15 api tests failing with `NotImplementedError`.
That is correct: the tests are the spec, and they only go green when your engine is right.
The harness itself was verified against a reference implementation (all 31 tests green, XOR at
MSE ~1e-27), so a red test means a real bug in your code, not in the spec.
