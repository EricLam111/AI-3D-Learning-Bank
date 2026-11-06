# From YOLOv8 User to AI Builder — Project-Based Roadmap

**Your starting point:** you can train YOLOv8 for RoboMaster, but the internals are opaque.
**Your goal:** understand AI by building it, then learn to *improve* models with measurements instead of vibes.
**Budget:** ~2 h/day focused work (~14 h/week) + overnight/weekend training slots (~12 h unattended).
**Hardware:** Ryzen AI Max APU (unified LPDDR5X, large RAM) + RTX 3060 12 GB over Thunderbolt 4, Linux.

Expected duration: **~6 months part-time**, 13 projects. Every project ends in a measurable
acceptance test — if you can't measure it, you didn't learn it.

---

## The learning contract (read this before every session)

1. **No tutorial copy-paste.** You may read papers, blog posts, and PyTorch docs. You may not paste
   working code you didn't derive. The point is the derivation.
2. **Every project has an acceptance test.** Green test = project done. Be honest about red.
3. **Cross-check against the reference.** After building from scratch, compare numerically with
   PyTorch/Ultralytics. Matching a reference to 1e-6 is how you prove you understood it.
4. **One thing changed per experiment.** Write the hypothesis *before* the run.
5. **Write it down.** Each project gets a short `NOTES.md`: what you built, what surprised you,
   what broke, the numbers, and one paragraph explaining the concept in plain language.
   If you can't write the paragraph, you don't understand it yet.
6. **Reading:** one paper per 2 weeks, always paired with an implementation project below.

Experiment log format (keep it boring and consistent):

```
## 2025-01-14 P5-detector: focal loss on/off
Hypothesis: focal loss γ=2 raises mAP@0.5 by >2 pts at 50 epochs on the 3k-image subset.
Change: obj loss BCE -> focal(γ=2). Everything else identical, seed 1337.
Result: mAP 0.412 -> 0.447 (+3.5). Val loss stops over-penalising easy bg.
Conclusion: confirmed. Next: does it still help at 100 epochs?
```

---

## Step 0 — Environment (half a day, do this first)

Create the shared learning environment separately on each computer using
`conda env create -f environment.yml` from the repository root, then activate `ai-learn`.
See the root README for Windows and Ubuntu instructions. The first exercises can run on CPU;
connecting the external RTX 3060 and configuring a GPU-enabled PyTorch installation are optional
until you choose a training project that needs them. Consult the current PyTorch installation guide
for the specific OS and hardware rather than assuming a fixed CUDA wheel.

- [ ] Create the Conda environment and verify the CPU-based project tests run.
- [ ] When needed, connect the eGPU and configure a compatible GPU runtime for that machine.
- [ ] Keep datasets and model weights outside Git; compare experiments on fixed splits.

- [ ] Install: `torch`, `numpy`, `matplotlib`, `tqdm`, `pytest`, `jupyterlab`, `tensorboard`.
      Keep `ultralytics` in a **separate** env — it's your reference implementation, not your crutch.
- [ ] Repo layout:
      `projects/pNN-name/{README.md, NOTES.md, src/, tests/, runs/}`; commit before every experiment.
- [ ] Smoke test: train a 2-layer MLP on 1000 MNIST images, confirm `nvidia-smi` shows GPU
      memory in use and the loss decreases.

**Fallback if the eGPU stays flaky:** free Colab/Kaggle T4/P100 for the heavy training runs,
local CPU for the from-scratch phases (they're tiny by design), and a small paid GPU budget for
Phase 5 only. Do not let hardware stall the learning.

---

## Phase 1 — Become the autograd engine (weeks 1–4, ~45 h)

Goal: stop treating "the model learns" as magic.

### P1.1 — Micro-autograd engine (~10 h)
Build a `Tensor` class with `+ - * / @ exp log relu sum mean reshape`, broadcasting, and `.backward()`
that fills `.grad` by walking a topological sort. No PyTorch autograd allowed.
**Acceptance:** for 100 random computation graphs, your gradients match `torch.autograd` to 1e-6
(float64). Then build an MLP on top of it and train XOR.
**You should be able to explain:** why backprop is just the chain rule + memoisation.

### P1.2 — MNIST by hand, pure NumPy (~14 h)
2-layer MLP, Fashion-MNIST, manual forward *and* manual backward (no autograd at all), minibatch SGD.
**Acceptance:** ≥ 88 % test accuracy (Fashion-MNIST), and your analytic gradients match finite
differences to 1e-5 on a small batch.
Then rewrite it on P1.1's engine and reproduce the same accuracy — now you know what autograd buys you.

### P1.3 — Optimisers & the loss-curve diagnostic (~10 h)
Implement SGD, momentum, Nesterov, RMSProp, Adam, AdamW from scratch. Run a learning-rate sweep
(1e-4 … 1e0) on the same task; plot loss vs step for each.
**Acceptance:** you can look at an unseen loss curve and correctly name the failure mode —
LR too high (spikes/NaN), too low (flat), overfit (train↓ val↑), underfit (both flat),
bad init (dead at start), batch too small (noisy), data bug (loss won't move at all).
Write the diagnosis cheat-sheet into `NOTES.md`. **This is the single most useful skill in the roadmap.**

### P1.4 — Convolution, unfogged (~12 h)
Implement `conv2d` in NumPy via `im2col` + matmul, plus max-pool, batchnorm (train & eval modes),
and backward passes for all three. Recreate LeNet on MNIST.
**Acceptance:** your conv forward matches `torch.nn.functional.conv2d` to 1e-6 and your weight grads
match `torch.autograd` to 1e-5 on random inputs. ≥ 99 % MNIST test accuracy.
**You should be able to explain:** receptive field, stride/dilation, why batchnorm has two modes,
and where the FLOPs in a CNN actually go.

---

## Phase 2 — Open the black box you already use: detection (weeks 5–9, ~55 h)

This is the phase that converts your RoboMaster experience into understanding.
Use **your own RoboMaster armour-plate dataset** — you already have data, labels and intuition,
so all your attention goes to the mechanics.

### P2.1 — Detection primitives from scratch (~12 h)
IoU / GIoU / DIoU / CIoU, NMS (and soft-NMS), box encode/decode, anchor generation, and the
matching/assignment problem (IoU-threshold assignment, then a simplified task-aligned assigner).
**Acceptance:** your IoU and NMS match `torchvision.ops` outputs exactly on 10k random boxes;
your mAP matches `torchmetrics` on the same predictions. Pure NumPy/Torch, no library calls inside.

### P2.2 — Train a tiny single-stage detector (~20 h)
Backbone (your P1.4 conv blocks) + multi-scale head, BCE for objectness, focal loss for class,
CIoU for box. Train at 416 px on a 3–5k image subset — comfortable on 12 GB.
**Acceptance:** your own evaluator reports mAP@0.5 **and** mAP@0.5:0.95; you beat a stated baseline
you set *before* training (write the number down first); inference visualisations look sane on 10
hand-picked hard frames (motion blur, partial occlusion, small distant plates).

### P2.3 — Reproduce YOLOv8's loss and beat it once (~13 h)
Read the v8 head: decoupled branches, DFL (distribution focal loss) over 16 bins, task-aligned
assigner. Implement DFL yourself and verify it numerically. Then:
**Acceptance:** a written, derivation-level explanation of every term in the Ultralytics v8 loss
(with your own formulas, no copy-paste); plus one experiment where you change something in the
pipeline and *show the delta* against `yolov8n` trained on the same split under the same budget.

### P2.4 — Ablation protocol: how to "improve a model" (~10 h)
Run a 2×5 grid on the detector: {no augmentation, mosaic only, mosaic+HSV+flip} × {baseline,
+CIoU, +focal, +EMA, +cosine LR}. Same seed, same epochs, same eval.
**Acceptance:** a results table plus a one-page conclusion ranking which lever mattered most and
why. This is the deliverable that teaches you *improvement* — data and augmentation usually beat
architecture, and you should be able to prove it on your own numbers.

---

## Phase 3 — Language models: build one you can explain (weeks 10–17, ~65 h)

### P3.1 — BPE tokenizer from scratch (~8 h)
Byte-level BPE, train on a 100 MB text mix, encode/decode, special tokens.
**Acceptance:** round-trip is lossless on 1 M random unicode docs; compression ratio is within 5 %
of `tiktoken` on the same corpus. Explain why tokenisation causes "it can't spell / can't count" bugs.

### P3.2 — GPT from scratch (~25 h)
Bigram → MLP char model → full decoder-only transformer: token+positional embeddings, causal
multi-head attention, LayerNorm, residuals, weight tying, cross-entropy. Train on TinyStories.
**Acceptance:** val loss curve matches a hand-computed sanity estimate at init
(−ln(1/vocab)); samples become grammatical TinyStories; you can state parameter count and
per-step FLOPs for your config yourself. **Hard rule: write every line; nanoGPT is reference reading only.**

### P3.3 — The improvement ladder, measured (~22 h)
One change at a time, each with its own curve and a fixed token budget:
LR warmup + cosine → weight tying → pre-LN vs post-LN → RMSNorm + SwiGLU →
RoPE → GQA (fewer KV heads) → gradient clipping / grad accumulation.
**Acceptance:** a table of final val loss per change, and 3 sentences per row on *why* it helped
(or hurt). Then scale the winning recipe to the largest model that trains overnight on 12 GB and
report the loss-vs-compute (Chinchilla-style) point you landed on.

### P3.4 — Own evaluation harness (~10 h)
Task-based evals you write yourself: exact-match recall on a held-out Q&A set, a small
multiple-choice benchmark, and 20 adversarial prompts of your own. Plus a prompt-regression suite.
**Acceptance:** `python -m evals.run` prints a table for any checkpoint and is stable across reruns
(seeded). Everything after this point is judged by this harness, not by vibes.

---

## Phase 4 — Improve an existing model: fine-tuning (weeks 18–22, ~40 h)

### P4.1 — LoRA from scratch (~14 h)
Implement low-rank adapters yourself (both in a from-scratch block and injected into a HF model):
rank/alpha, target modules, merged-vs-unmerged inference.
**Acceptance:** your LoRA matches `peft` LoRA output to 1e-4 with identical weights; you can explain
rank vs expressivity vs VRAM trade-off, and show it with measurements.

### P4.2 — QLoRA fine-tune on the 3060 (~14 h)
4-bit quantised base (7–8B class fits in 12 GB QLoRA) + your LoRA + a small instruction dataset
you curate (start from your own notes/questions, clean it deliberately).
**Acceptance:** base vs fine-tuned scored on **your P3.4 harness**, with a table showing where it
improved and where it **regressed** (catastrophic forgetting is a finding, not a failure — report it).

### P4.3 — Data beats parameters (~12 h)
Three dataset variants: raw scrape / dedup+filtered / dedup+filtered+hard-negative examples.
Same base, same LoRA config, same budget.
**Acceptance:** a curve or table proving which data intervention paid off, plus the exact filter
rules you used. **This is the most transferable lesson in the entire roadmap.**

---

## Phase 5 — Capstone: reproduce, then extend (weeks 23–28)

Pick **one**:

- **A. Detection paper reproduction.** Reproduce the key ablation table of a detector paper
  (e.g. the DFL / assigner component) at reduced scale on your RoboMaster data; state where your
  curve diverges from the paper and hypothesise why.
- **B. SLM from scratch, end to end.** Tokenizer → pretrain (~100–300 M params) → SFT → your eval
  harness, everything on the 3060 with overnight runs. Publish the loss curves and the eval table.
- **C. RL for autonomy** (fits RoboMaster). Implement REINFORCE → DQN → PPO from scratch on
  CartPole, verify against stable-baselines3 scores, then apply one to a simplified
  target-tracking/gimbal-aiming sim.

**Acceptance:** a public repo with README, reproducible commands, the numbers, and a 1-page
"what I'd do with 10× compute" analysis. This is your portfolio artifact.

---

## Weekly rhythm (2 h/day)

| Day | Focus |
|---|---|
| Mon–Thu | 2 h build + write `NOTES.md`. Stop mid-problem if needed; log where you stopped. |
| Fri | 2 h: tests + acceptance check against the reference implementation. |
| Sat | Launch the long training run in the evening (`nohup`, checkpointing, 12 h budget). |
| Sun | 2 h: read the curves, run the ablation, write the log entry. Plan next week. |

Rules for unattended runs: checkpoint every epoch, log to TensorBoard, `nohup`/`tmux`,
verify the GPU is actually busy for the first 5 minutes before you walk away.

---

## Milestones

- **Week 4:** autograd + conv reimplemented and verified against PyTorch. You can diagnose a loss curve.
- **Week 9:** your own detector with your own mAP, plus a real ablation table against YOLOv8n.
- **Week 17:** a transformer you wrote, whose improvements you measured one at a time.
- **Week 22:** a fine-tuned model judged by your own eval harness, with data-ablation evidence.
- **Week 28:** one paper/claim reproduced and extended, publicly documented.

## Anti-patterns to avoid

- Reading a fourth "transformers explained" blog post instead of fixing the shape bug in your code.
- Measuring with `print(loss)` and calling it evaluation.
- Changing three things and attributing the win to the one you liked.
- Fine-tuning before you have an eval set — you'll optimise noise.
- Chasing architecture before data quality and augmentation (prove this to yourself in P2.4).