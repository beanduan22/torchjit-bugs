# TorchJIT Numerical Inconsistency Bugs — torch 2.11.0+cu128

**27 confirmed bugs** where `torch.jit.optimize_for_inference` produces numerically different results from eager mode on **torch 2.11.0+cu128**.

Discovered using [NNSmith](https://github.com/ise-uiuc/nnsmith) differential fuzzing on an NVIDIA RTX 6000 Ada GPU.

## Root Cause

`torch.jit.trace` + `torch.jit.freeze` + `torch.jit.optimize_for_inference` rewrites the computation graph in a way that changes numerical results when evaluation inputs differ from the tracing inputs.

## Environment

```
torch==2.11.0+cu128
CUDA 12.8
Python 3.13
NVIDIA RTX 6000 Ada Generation
```

## How to Reproduce

```bash
pip install nnsmith torch==2.11.0
python3 bug_N_repro.py
```

---

## Bug Outputs

### Bug 0 — ReduceMax + ReLU (`max_diff=0.01675`)
```
Bug 0: torch 2.11.0+cu128
  output[0]: shape=(1, 1) dtype=torch.float32 max_diff=0
  output[1]: shape=(9, 21) dtype=torch.float32 max_diff=0
  output[2]: shape=(9, 21) dtype=torch.float32 max_diff=0.01675
  *** BUG: eager != JIT  (max_diff=0.01675) ***
```

### Bug 1 — NCHWConv2d (`max_diff=17.68`)
```
Bug 1: torch 2.11.0+cu128
  output[0]: shape=(1, 2, 1, 1) dtype=torch.float32 max_diff=7.629e-06
  output[1]: shape=(1, 1, 1, 1) dtype=torch.float32 max_diff=0
  output[2]: shape=(1, 1, 1) dtype=torch.float32 max_diff=1.907e-06
  output[3]: shape=(1, 1, 1) dtype=torch.float32 max_diff=1.907e-06
  output[4]: shape=(39, 48, 1, 1, 1) dtype=torch.float32 max_diff=17.68
  *** BUG: eager != JIT  (max_diff=17.68) ***
```

### Bug 2 — Sin (`max_diff=208.2`)
```
Bug 2: torch 2.11.0+cu128
  output[0]: shape=(1, 53, 2, 2) dtype=torch.float32 max_diff=0
  output[1]: shape=(53, 1, 41, 1) dtype=torch.float32 max_diff=208.2
  *** BUG: eager != JIT  (max_diff=208.2) ***
  output[2]: shape=(53, 1, 41, 1) dtype=torch.float32 max_diff=0.0001221
```

### Bug 3 — NCHWConv2d + int32/int64 (`max_diff=1`)
```
Bug 3: torch 2.11.0+cu128
  output[0]: shape=(1, 62) dtype=torch.float32 max_diff=0
  output[1]: shape=(1, 48, 36, 31) dtype=torch.int32 max_diff=1
  *** BUG: eager != JIT  (max_diff=1) ***
  output[2]: shape=(1, 36, 31) dtype=torch.int64 max_diff=1
  *** BUG: eager != JIT  (max_diff=1) ***
  output[3]: shape=(1, 48, 36, 31) dtype=torch.int32 max_diff=1
  *** BUG: eager != JIT  (max_diff=1) ***
```

### Bug 4 — complex128 + FFT (`max_diff=2829`)
```
Bug 4: torch 2.11.0+cu128
  output[0]: shape=(60, 39, 28) dtype=torch.complex128 max_diff=0
  output[1]: shape=(60, 24, 2) dtype=torch.float32 max_diff=2829
  *** BUG: eager != JIT  (max_diff=2829) ***
  output[2]: shape=(60, 24, 2) dtype=torch.int32 max_diff=0
  output[3]: shape=(60, 24, 28) dtype=torch.float32 max_diff=0
```

### Bug 5 — Concat (`max_diff=94`)
```
Bug 5: torch 2.11.0+cu128
  output[0]: shape=(1,) dtype=torch.float32 max_diff=0
  output[1]: shape=(7, 50, 2) dtype=torch.int64 max_diff=0
  output[2]: shape=(7, 1, 50, 2) dtype=torch.float32 max_diff=0
  output[3]: shape=(50, 1, 7, 8) dtype=torch.float32 max_diff=1.526e-05
  output[4]: shape=(50, 1, 7, 8) dtype=torch.float32 max_diff=94
  *** BUG: eager != JIT  (max_diff=94) ***
```

### Bug 6 — GELU + Flatten (`max_diff=0.01908`)
```
Bug 6: torch 2.11.0+cu128
  output[0]: shape=(26235,) dtype=torch.float32 max_diff=0
  output[1]: shape=() dtype=torch.int64 max_diff=0
  output[2]: shape=(1,) dtype=torch.float32 max_diff=0.01908
  *** BUG: eager != JIT  (max_diff=0.01908) ***
```

### Bug 7 — Concat + ExpandLast2 + int32 (`max_diff=1`)
```
Bug 7: torch 2.11.0+cu128
  output[0]: shape=(25, 40, 2, 10) dtype=torch.float32 max_diff=0
  output[1]: shape=(100, 100) dtype=torch.float32 max_diff=0
  output[2]: shape=(40, 25, 1, 54) dtype=torch.int32 max_diff=1
  *** BUG: eager != JIT  (max_diff=1) ***
  output[3]: shape=(40, 1, 54) dtype=torch.float32 max_diff=3.815e-06
```

### Bug 8 — Neg + Slice (`max_diff=0.1341`)
```
Bug 8: torch 2.11.0+cu128
  output[0]: shape=(1, 14, 59, 1) dtype=torch.float32 max_diff=0
  output[1]: shape=(59, 14, 27) dtype=torch.float32 max_diff=0.0006104
  output[2]: shape=(1, 59, 1) dtype=torch.float32 max_diff=0.01562
  *** BUG: eager != JIT  (max_diff=0.01562) ***
  output[3]: shape=(14, 59, 27) dtype=torch.float32 max_diff=0.1341
  *** BUG: eager != JIT  (max_diff=0.1341) ***
```

### Bug 9 — PTMatMul + float64 (`max_diff=6.6`)
```
Bug 9: torch 2.11.0+cu128
  output[0]: shape=(55,) dtype=torch.float32 max_diff=0
  output[1]: shape=(55,) dtype=torch.float32 max_diff=0
  output[2]: shape=(1,) dtype=torch.float32 max_diff=0
  output[3]: shape=(1,) dtype=torch.float64 max_diff=6.6
  *** BUG: eager != JIT  (max_diff=6.6) ***
```

### Bug 10 — Linear + BatchNorm2d + int64 (`max_diff=41`)
```
Bug 10: torch 2.11.0+cu128
  output[0]: shape=(22, 42, 22) dtype=torch.float32 max_diff=8.941e-08
  output[1]: shape=(1,) dtype=torch.float32 max_diff=0
  output[2]: shape=(22, 22) dtype=torch.int64 max_diff=41
  *** BUG: eager != JIT  (max_diff=41) ***
```

### Bug 11 — Sub (`max_diff=0.04404`)
```
Bug 11: torch 2.11.0+cu128
  output[0]: shape=() dtype=torch.int64 max_diff=0
  output[1]: shape=(220,) dtype=torch.float32 max_diff=0
  output[2]: shape=(1,) dtype=torch.float32 max_diff=0.04404
  *** BUG: eager != JIT  (max_diff=0.04404) ***
```

### Bug 12 — Clip (`max_diff=11.61`)
```
Bug 12: torch 2.11.0+cu128
  output[0]: shape=(1,) dtype=torch.float32 max_diff=0.0004883
  output[1]: shape=(1,) dtype=torch.float32 max_diff=11.61
  *** BUG: eager != JIT  (max_diff=11.61) ***
  output[2]: shape=(37,) dtype=torch.bool max_diff=0
```

### Bug 13 — Linear + ConstPad + NaN (`max_diff=4952 + NaN`)
```
Bug 13: torch 2.11.0+cu128
  output[0]: shape=(1, 1, 11, 54, 39) dtype=torch.float32 max_diff=0.0003662
  output[1]: shape=(1, 1, 11, 54, 39) dtype=torch.float32 max_diff=4952
  *** BUG: eager != JIT  (max_diff=4952) ***
  output[2]: shape=(1, 1, 11, 54, 39) dtype=torch.float32 max_diff=nan
  output[3]: shape=(1, 1, 11, 54, 39) dtype=torch.float32 max_diff=0.9922
  *** BUG: eager != JIT  (max_diff=0.9922) ***
  output[4]: shape=(1, 1, 11, 54, 39) dtype=torch.float32 max_diff=1.903
  *** BUG: eager != JIT  (max_diff=1.903) ***
```

### Bug 14 — Linear + Max (`max_diff=9.307`)
```
Bug 14: torch 2.11.0+cu128
  output[0]: shape=(62,) dtype=torch.float32 max_diff=0.0004883
  output[1]: shape=(62,) dtype=torch.float32 max_diff=9.307
  *** BUG: eager != JIT  (max_diff=9.307) ***
  output[2]: shape=(62,) dtype=torch.bool max_diff=0
  output[3]: shape=(1,) dtype=torch.float32 max_diff=0.0001221
```

### Bug 15 — Concat (`max_diff=288.5`)
```
Bug 15: torch 2.11.0+cu128
  output[0]: shape=(20, 3, 1, 22, 49) dtype=torch.float32 max_diff=0.0004883
  output[1]: shape=(20, 3, 1, 22, 49) dtype=torch.float32 max_diff=288.5
  *** BUG: eager != JIT  (max_diff=288.5) ***
  output[2]: shape=(20, 3, 1, 22, 49) dtype=torch.float32 max_diff=1.896
  *** BUG: eager != JIT  (max_diff=1.896) ***
  output[3]: shape=(20, 3, 1, 22, 49) dtype=torch.float32 max_diff=0.0009775
```

### Bug 16 — Slice + ReduceMean (`max_diff=31.14`)
```
Bug 16: torch 2.11.0+cu128
  output[0]: shape=(35, 35) dtype=torch.float32 max_diff=0
  output[1]: shape=(35, 35) dtype=torch.float32 max_diff=0
  output[2]: shape=(35, 35) dtype=torch.bool max_diff=0
  output[3]: shape=(35, 38) dtype=torch.float32 max_diff=31.14
  *** BUG: eager != JIT  (max_diff=31.14) ***
```

### Bug 17 — Atan (`max_diff=675.1`)
```
Bug 17: torch 2.11.0+cu128
  output[0]: shape=(59, 1, 1, 1, 38) dtype=torch.float32 max_diff=0
  output[1]: shape=(59, 19, 1, 1, 38) dtype=torch.float32 max_diff=675.1
  *** BUG: eager != JIT  (max_diff=675.1) ***
  output[2]: shape=(59, 19, 1, 1, 38) dtype=torch.float32 max_diff=1.973e-05
  output[3]: shape=(59, 19, 1, 1, 38) dtype=torch.float32 max_diff=4.53e-06
```

### Bug 18 — Transpose + complex64 (`max_diff=1933`)
```
Bug 18: torch 2.11.0+cu128
  output[0]: shape=(59, 59, 1, 6) dtype=torch.complex64 max_diff=0
  output[1]: shape=(59, 1, 6, 59) dtype=torch.float64 max_diff=0
  output[2]: shape=(59, 1, 6, 59) dtype=torch.int32 max_diff=0
  output[3]: shape=(59, 1, 6, 1) dtype=torch.float32 max_diff=1933
  *** BUG: eager != JIT  (max_diff=1933) ***
  output[4]: shape=(59, 1, 6, 59) dtype=torch.float64 max_diff=0
```

### Bug 19 — Neg + ReduceMin (`max_diff=49.1`)
```
Bug 19: torch 2.11.0+cu128
  output[0]: shape=(23, 1, 3, 1) dtype=torch.float32 max_diff=0
  output[1]: shape=(23, 1, 2, 1) dtype=torch.float32 max_diff=1.808
  *** BUG: eager != JIT  (max_diff=1.808) ***
  output[2]: shape=(23, 1, 2, 1) dtype=torch.float32 max_diff=1.526e-05
  output[3]: shape=(23, 1, 20, 11) dtype=torch.float32 max_diff=49.1
  *** BUG: eager != JIT  (max_diff=49.1) ***
```

### Bug 20 — TorchReduceSum + Linear (`max_diff=67.01`)
```
Bug 20: torch 2.11.0+cu128
  output[0]: shape=(38160,) dtype=torch.float32 max_diff=57.92
  *** BUG: eager != JIT  (max_diff=57.92) ***
  output[1]: shape=(45, 1, 2, 1) dtype=torch.float32 max_diff=0
  output[2]: shape=(45, 1, 16, 53) dtype=torch.float32 max_diff=67.01
  *** BUG: eager != JIT  (max_diff=67.01) ***
```

### Bug 21 — Mul + int64 (`max_diff=3`)
```
Bug 21: torch 2.11.0+cu128
  output[0]: shape=(1, 28, 1, 43) dtype=torch.int64 max_diff=3
  *** BUG: eager != JIT  (max_diff=3) ***
  output[1]: shape=(1, 28, 20, 2, 43) dtype=torch.bool max_diff=0
  output[2]: shape=(1, 28, 20, 1, 43) dtype=torch.bool max_diff=0
```

### Bug 22 — Neg + TorchReduceSum (`max_diff=0.02516`)
```
Bug 22: torch 2.11.0+cu128
  output[0]: shape=(1, 43, 25) dtype=torch.int64 max_diff=0
  output[1]: shape=(1, 2, 1, 2) dtype=torch.float32 max_diff=0.005859
  *** BUG: eager != JIT  (max_diff=0.005859) ***
  output[2]: shape=(1, 2, 43, 6) dtype=torch.float32 max_diff=0
  output[3]: shape=(1, 2, 6, 43) dtype=torch.float32 max_diff=0.02516
  *** BUG: eager != JIT  (max_diff=0.02516) ***
  output[4]: shape=(1, 43, 25, 6) dtype=torch.int16 max_diff=0
```

### Bug 23 — Slice (`max_diff=31.41`)
```
Bug 23: torch 2.11.0+cu128
  output[0]: shape=(24, 1, 37, 1, 41) dtype=torch.float32 max_diff=31.41
  *** BUG: eager != JIT  (max_diff=31.41) ***
  output[1]: shape=(24, 37, 1, 41) dtype=torch.float32 max_diff=1.526e-05
  output[2]: shape=(24, 1, 37, 1, 1) dtype=torch.bool max_diff=0
```

### Bug 24 — Sub (`max_diff=33.18`)
```
Bug 24: torch 2.11.0+cu128
  output[0]: shape=(63, 1, 5, 1, 2) dtype=torch.float32 max_diff=0
  output[1]: shape=(63, 1, 1, 2) dtype=torch.float32 max_diff=0
  output[2]: shape=(63, 1, 1, 1) dtype=torch.float32 max_diff=33.18
  *** BUG: eager != JIT  (max_diff=33.18) ***
  output[3]: shape=(63, 1, 1, 46) dtype=torch.float32 max_diff=7.629e-06
```

### Bug 25 — Slice + Sigmoid (`max_diff=2.07`)
```
Bug 25: torch 2.11.0+cu128
  output[0]: shape=(1, 1, 28, 28) dtype=torch.int64 max_diff=0
  output[1]: shape=(28, 1) dtype=torch.float32 max_diff=0.007812
  *** BUG: eager != JIT  (max_diff=0.007812) ***
  output[2]: shape=(28, 24) dtype=torch.float32 max_diff=2.07
  *** BUG: eager != JIT  (max_diff=2.07) ***
```

### Bug 26 — Linear + BatchNorm2d + int64 (`max_diff=41`, duplicate root cause of Bug 10)
```
Bug 26: torch 2.11.0+cu128
  output[0]: shape=(22, 42, 22) dtype=torch.float32 max_diff=8.941e-08
  output[1]: shape=(1,) dtype=torch.float32 max_diff=0
  output[2]: shape=(22, 22) dtype=torch.int64 max_diff=41
  *** BUG: eager != JIT  (max_diff=41) ***
```

---

## Summary Table

| Bug | Max Diff | Key Ops |
|-----|----------|---------|
| 0 | 0.01675 | ReduceMax, ReLU |
| 1 | 17.68 | NCHWConv2d |
| 2 | 208.2 | Sin |
| 3 | 1.0 | NCHWConv2d, int32/int64 |
| 4 | 2829 | complex128, FFT |
| 5 | 94 | Concat |
| 6 | 0.01908 | GELU, Flatten |
| 7 | 1.0 | Concat, ExpandLast2, int32 |
| 8 | 0.1341 | Neg, Slice |
| 9 | 6.6 | PTMatMul, float64 |
| 10 | 41 | Linear, BatchNorm2d, int64 |
| 11 | 0.04404 | Sub |
| 12 | 11.61 | Clip |
| 13 | 4952 + NaN | Linear, ConstPad |
| 14 | 9.307 | Linear, Max |
| 15 | 288.5 | Concat |
| 16 | 31.14 | Slice, ReduceMean |
| 17 | 675.1 | Atan |
| 18 | 1933 | Transpose, complex64 |
| 19 | 49.1 | Neg, ReduceMin |
| 20 | 67.01 | TorchReduceSum, Linear |
| 21 | 3.0 | Mul, int64 |
| 22 | 0.02516 | Neg, TorchReduceSum |
| 23 | 31.41 | Slice |
| 24 | 33.18 | Sub |
| 25 | 2.07 | Slice, Sigmoid |
| 26 | 41 | Linear, BatchNorm2d (same root as Bug 10) |

Discovered by [NNSmith](https://github.com/ise-uiuc/nnsmith) + [trion](https://github.com/beanduan22/trion) differential fuzzing campaign.
