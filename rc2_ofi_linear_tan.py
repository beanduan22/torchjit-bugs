"""
Root Cause 2: torch.jit.optimize_for_inference decomposes aten::linear into
aten::matmul + aten::add.  For large matrices, the two kernels accumulate
floating-point rounding errors differently.  When the linear output feeds into
a singularity-prone function such as torch.tan(), even a tiny difference in the
input (≈1e-6) causes a huge difference in the output because tan(x) diverges
as x → π/2 + k·π.

Affected: torch 2.11.0+cu128
Pipeline: eager vs torch.jit.trace -> torch.jit.freeze -> torch.jit.optimize_for_inference

Root cause
----------
torch.jit.freeze converts:
    aten::linear(x, W, b)
into:
    aten::matmul(x, W_transposed_constant) + b

OFI then applies additional BLAS/kernel optimizations (oneDNN, FBGEMM, etc.)
on top, using a different FP accumulation path.  For inputs with values in [3, 7]
fed through a 39×39 Linear, the accumulated rounding error is ≈1e-6.  When the
linear output is near π/2 + k·π, applying tan() amplifies this ≈1e-6 difference
into a difference of tens to hundreds.

Affected bug instances: 13, 15, 17, 18, 25
(largest observed diff: 1282; typical diff: 1–100)
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Minimal: Linear(39->39) followed by tan."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(39, 39)

    def forward(self, x):
        return torch.tan(self.linear(x))


def reproduce():
    # seed=5 produces a weight matrix whose outputs, for inputs in [3,7],
    # include elements within 1e-3 of π/2 + k·π — close enough that the
    # ≈1e-6 BLAS rounding introduced by OFI pushes some across the singularity.
    torch.manual_seed(5)
    net = Model().eval()

    # 5D input with values in [3, 7] (same range used by nnsmith fuzzer)
    x = torch.rand(1, 1, 5, 10, 39) * 4 + 3

    # Eager reference
    with torch.no_grad():
        eager = net(x)

    # JIT pipeline: trace -> freeze -> optimize_for_inference
    traced = torch.jit.trace(net, [x])
    frozen = torch.jit.freeze(traced)
    opt    = torch.jit.optimize_for_inference(frozen)

    with torch.no_grad():
        jit = opt(x)

    # Exclude NaN/Inf (natural singularities of tan)
    valid = ~(torch.isnan(eager) | torch.isinf(eager) |
              torch.isnan(jit)   | torch.isinf(jit))
    max_diff = (eager[valid] - jit[valid]).abs().max().item() if valid.any() else 0.0
    nan_change = ((torch.isnan(eager) | torch.isinf(eager)) !=
                  (torch.isnan(jit)   | torch.isinf(jit))).sum().item()

    print(f"max |eager - jit| on finite elements : {max_diff:.4f}")
    print(f"elements that change NaN/Inf status  : {nan_change}")
    print(f"valid elements compared              : {valid.sum().item()} / {eager.numel()}")

    assert max_diff > 0.1, f"Bug not triggered (diff={max_diff})"
    print(f"\nBUG CONFIRMED: optimize_for_inference gives wrong tan output (diff={max_diff:.4f})")
    print("Root cause: linear->matmul+add decomposition + tan singularity amplification")


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    reproduce()
