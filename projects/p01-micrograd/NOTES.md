# NOTES — P1.1 Micro-autograd

Fill this in as you go. Empty sections are fine; empty at the *end* means the project isn't done.

## What backprop actually is (write this in your own words, no jargon)

<!-- One paragraph. If it needs the word "obviously", rewrite it. -->

## Implementation log

| Date | What I did | What broke | How I found it |
|---|---|---|---|
| | | | |

## Surprises

-

## Open questions

-

## Self-check answers

1. Why topological order instead of plain recursion?
2. Why `+=` for gradients?
3. In `x @ y`, why `x.grad = out.grad @ y.data.T`?
4. Why does `mean()` scale the gradient by 1/N?
5. Adding `tanh`: local gradient and saturation?
