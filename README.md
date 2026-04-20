# TorchJIT Bugs

Minimal standalone reproducers for three root causes found in the PyTorch JIT pipeline
(`torch.jit.trace` → `torch.jit.freeze` → `torch.jit.optimize_for_inference`).

Each script is fully self-contained — no external weight files, no nnsmith dependency.

**Affected version:** `torch 2.11.0+cu128`

---

## RC1 — `torch.jit.freeze` incorrect BatchNorm fusion

**File:** `rc1_freeze_bn_linear_fusion.py`

`torch.jit.freeze` folds `BatchNorm2d` into a preceding `Linear` layer when the input
is 4D with shape `[N, C, H, 1]`.  The fusion is wrong because the BN channel dimension
(dim 1, size `C`) does not correspond to the Linear output dimension (last dim, also
size `C`).  Freeze applies the BN scale/shift along the wrong axis, corrupting the output.

**Pipeline step:** `torch.jit.freeze`

**Model:** `Linear(1 → 22) → BatchNorm2d(22)` on input `[N, 22, H, 1]`

```
$ python rc1_freeze_bn_linear_fusion.py
PyTorch version: 2.11.0+cu128
max absolute difference (eager vs freeze): 0.134915
BUG CONFIRMED: torch.jit.freeze gives wrong output (diff=0.1349)
  eager mean : -0.049042
  frozen mean: -0.049755
```

---

## RC2 — `optimize_for_inference` linear decomposition + tan amplification

**File:** `rc2_ofi_linear_tan.py`

`optimize_for_inference` decomposes `aten::linear(x, W, b)` into
`aten::matmul(x, W_T) + b` and applies additional BLAS/kernel optimizations
(oneDNN, FBGEMM, prepacked weights).  The optimized kernel accumulates
floating-point rounding errors differently from eager `aten::linear` (difference ≈ 1e-6).
When the linear output feeds into `torch.tan()` near a singularity (π/2 + k·π),
this tiny rounding difference is amplified into a large output divergence.

**Pipeline step:** `torch.jit.optimize_for_inference`

**Model:** `Linear(39 → 39) → tan` on input `[1, 1, 5, 10, 39]` with values in `[3, 7]`

```
$ python rc2_ofi_linear_tan.py
PyTorch version: 2.11.0+cu128
max |eager - jit| on finite elements : 0.6689
elements that change NaN/Inf status  : 0
valid elements compared              : 1950 / 1950

BUG CONFIRMED: optimize_for_inference gives wrong tan output (diff=0.6689)
Root cause: linear->matmul+add decomposition + tan singularity amplification
```

---

## RC3 — `optimize_for_inference` linear decomposition + argmax sensitivity

**File:** `rc3_ofi_linear_argmax.py`

Same root cause as RC2 (OFI's optimized matmul uses a different FP accumulation path),
but the amplification mechanism is argmax sensitivity.  When `Linear(43 → 43)` is applied
to `K = 20` identical input rows, the output should be equal for all `K` candidates.
OFI's internal batching produces slightly different (≈ 1e-5) values for different rows,
which determines which candidate wins `argmax`.  Eager and OFI disagree on 140 of 1204
argmax outputs.

**Pipeline step:** `torch.jit.optimize_for_inference`

**Model:** `Linear(43 → 43) → argmax(dim=1)` on `K = 20` identical rows

```
$ python rc3_ofi_linear_argmax.py
PyTorch version: 2.11.0+cu128
argmax mismatches (eager vs ofi): 140 / 1204
eager unique vals : [0, 1, 2, 3]
ofi   unique vals : [0, 1]

BUG CONFIRMED: optimize_for_inference gives wrong argmax (140 mismatches)
Root cause: linear→matmul+add decomposition + argmax sensitivity to near-equal values
```
