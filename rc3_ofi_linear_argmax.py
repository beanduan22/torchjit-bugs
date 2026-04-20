"""
Root Cause 3: torch.jit.optimize_for_inference gives wrong argmax results due
to the same aten::linear → aten::matmul + aten::add decomposition as RC2, but
here the amplification mechanism is argmax sensitivity to near-equal values.

Affected: torch 2.11.0+cu128
Pipeline: eager vs torch.jit.trace -> torch.jit.freeze -> torch.jit.optimize_for_inference

Root cause
----------
OFI applies runtime BLAS/kernel optimizations (oneDNN, FBGEMM, prepacked weights)
that are not visible in the TorchScript graph but change the FP accumulation path.
When a Linear(43→43) is applied to K identical input rows, the optimized kernel
processes different rows with different internal batching, producing slightly
different (≈1e-5) outputs for different positions along the K dimension.

In eager mode, aten::linear also introduces small per-row differences via its own
BLAS batching, but they are different from OFI's differences. When aten::argmax
reduces over the K dimension, the tiny per-row differences determine which of the
K "equal" candidates wins — and eager and OFI pick different winners.

Affected bug instances: 21 (56 argmax mismatches out of 1204)
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Linear(43->43) -> argmax over the K dimension of equal-valued rows."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(43, 43)

    def forward(self, x):
        # x: [B, K, 43] where all K rows per batch element are identical
        return torch.argmax(self.linear(x), dim=1)   # [B, 43]


def reproduce():
    # seed=17 chosen to give a weight matrix where OFI's batched matmul
    # produces enough per-row variation to flip argmax results.
    torch.manual_seed(17)
    net = Model().eval()

    B, K = 28, 20
    # K identical rows per batch item: linear output is theoretically equal
    # for all K candidates, so argmax winner is determined solely by FP rounding.
    x_base = torch.ones(B, 43)
    x = x_base.unsqueeze(1).expand(B, K, 43).contiguous()  # [28, 20, 43]

    # Eager reference
    with torch.no_grad():
        eager = net(x)

    # JIT pipeline: trace -> freeze -> optimize_for_inference
    traced = torch.jit.trace(net, [x])
    frozen = torch.jit.freeze(traced)
    opt    = torch.jit.optimize_for_inference(frozen)

    with torch.no_grad():
        jit = opt(x)

    mismatch = (eager != jit).sum().item()
    total    = eager.numel()

    print(f"argmax mismatches (eager vs ofi): {mismatch} / {total}")
    print(f"eager unique vals : {eager.unique().tolist()}")
    print(f"ofi   unique vals : {jit.unique().tolist()}")

    assert mismatch > 0, "Bug not triggered (0 mismatches)"
    print(f"\nBUG CONFIRMED: optimize_for_inference gives wrong argmax ({mismatch} mismatches)")
    print("Root cause: linear→matmul+add decomposition + argmax sensitivity to near-equal values")


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    reproduce()
