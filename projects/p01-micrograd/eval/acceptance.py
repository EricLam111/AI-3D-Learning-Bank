#!/usr/bin/env python3
"""P1.1 acceptance runner. Exit code 0 means the project is done.

Checks, in order:
  1. tests/test_api.py         -- semantics (shapes, broadcasting, accumulation)
  2. tests/test_gradcheck.py   -- analytic gradients vs central finite differences
  3. tests/test_torch_parity.py-- gradients vs torch.autograd to 1e-6
  4. XOR                       -- an MLP on YOUR engine learns XOR (MSE < 0.01, 4/4 correct)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OK = "\033[32m ok \033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

SUITES = [
    ("api semantics", "tests/test_api.py"),
    ("finite-difference gradcheck", "tests/test_gradcheck.py"),
    ("torch.autograd parity", "tests/test_torch_parity.py"),
]


def run_suite(path: str) -> tuple[str, str]:
    """Return (status, detail)."""
    if not (ROOT / path).exists():
        return "FAIL", f"missing {path}"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", path, "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    tail = (proc.stdout + proc.stderr).strip().splitlines()
    summary = tail[-1] if tail else ""
    if proc.returncode == 0:
        if "skipped" in summary and "passed" not in summary:
            return "SKIP", summary
        return "ok", summary
    return "FAIL", summary


def train_xor(steps: int = 4000, hidden: int = 8):
    """Train an MLP on XOR with plain gradient descent over a few seeds/lrs."""
    import numpy as np

    from micrograd import Tensor
    from micrograd.nn import MLP

    x_np = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    t_np = np.array([[0.0], [1.0], [1.0], [0.0]])

    results = []
    for seed in (0, 1, 2):
        for lr in (0.1, 0.3, 1.0):
            net = MLP([2, hidden, 1], seed=seed)
            inputs = Tensor(x_np)
            target = Tensor(t_np)
            losses = []
            for _ in range(steps):
                net.zero_grad()
                pred = net(inputs)
                loss = ((pred - target) ** 2).mean()
                loss.backward()
                for p in net.parameters():
                    p.data -= lr * p.grad
                losses.append(float(loss.data))

            final = net(inputs).data.ravel()
            correct = int(np.sum((final > 0.5) == (t_np.ravel() > 0.5)))
            results.append(
                {
                    "seed": seed,
                    "lr": lr,
                    "loss": losses[-1],
                    "correct": correct,
                    "pred": final,
                }
            )
    results.sort(key=lambda r: (r["loss"], -r["correct"]))
    return results


def main() -> int:
    print("=" * 68)
    print("P1.1 acceptance -- micro-autograd engine")
    print("=" * 68)

    statuses: dict[str, str] = {}

    for label, path in SUITES:
        print(f"\n[suite] {label}  ({path})")
        status, detail = run_suite(path)
        statuses[label] = status
        print(f"  {OK if status == 'ok' else SKIP if status == 'SKIP' else FAIL}  {detail}")

    print("\n[xor] MLP([2,8,1]) on your engine, plain GD")
    try:
        results = train_xor()
    except NotImplementedError as exc:
        print(f"  {FAIL}  not implemented yet: {exc}")
        statuses["xor"] = "FAIL"
    except Exception:
        import traceback

        print(f"  {FAIL}  crashed:")
        traceback.print_exc()
        statuses["xor"] = "FAIL"
    else:
        print(f"  {'seed':>4} {'lr':>5} {'final mse':>10} {'correct':>8}   predictions")
        for r in results[:5]:
            preds = " ".join(f"{v:.2f}" for v in r["pred"])
            print(
                f"  {r['seed']:>4} {r['lr']:>5} {r['loss']:>10.5f} "
                f"{r['correct']:>5}/4   [{preds}]"
            )
        best = results[0]
        passed = best["loss"] < 0.01 and best["correct"] == 4
        statuses["xor"] = "ok" if passed else "FAIL"
        print(
            f"  {OK if passed else FAIL}  best mse={best['loss']:.5f} "
            f"(need < 0.01), correct={best['correct']}/4 (need 4/4)"
        )

    print("\n" + "-" * 68)
    all_ok = all(s == "ok" for s in statuses.values())
    for label, status in statuses.items():
        mark = OK if status == "ok" else SKIP if status == "SKIP" else FAIL
        print(f"  [{mark}] {label}")

    if all_ok:
        print("\nACCEPTED. Now fill in NOTES.md -- especially the plain-language paragraph.")
    else:
        print("\nNOT ACCEPTED yet. Red items above are the spec: make them green.")
        if statuses.get("torch.autograd parity") == "SKIP":
            print("  (parity needs torch: conda activate the project env, then pip install torch)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
