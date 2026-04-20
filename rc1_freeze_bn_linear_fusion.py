"""
Root Cause 1: torch.jit.freeze incorrectly folds BatchNorm2d into a preceding
Linear layer when the tensor layout is 4D and the BN channel dimension does not
correspond to the Linear output feature dimension.

Affected: torch 2.11.0+cu128 (and earlier versions)
Pipeline: eager vs torch.jit.trace -> torch.jit.freeze

Background
----------
torch.jit.freeze performs "constant folding" that fuses a BatchNorm2d layer
immediately following a Linear (or Conv2d) into a single fused linear call
by absorbing the BN scale/shift into the weight and bias.  The fusion is
correct only when the BN channel dimension matches the axis that Linear
broadcasts its output features along.

In the pattern below, the input is 4D with shape [N, C, H, 1].
  * Linear(1 -> C) acts on the LAST dimension -> output [N, C, H, C]
  * BatchNorm2d(C) normalises along DIM 1 (the channel axis)

After Linear, dim-1 has size C, and the LAST dim also has size C, but they
represent different axes.  freeze incorrectly applies the BN
scale/shift to the Linear weight as if they were the same axis, producing
wrong normalisation statistics.

Reproducer
----------
"""
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 22)      # 1 input feature -> 22 output features
        self.bn     = nn.BatchNorm2d(22)    # expects (N, 22, H, W); channel dim = 1

    def forward(self, x):
        # x: [N, 22, H, 1]
        t = self.linear(x)   # [N, 22, H, 22]  — last dim is the linear output (22)
        # bn sees (N=N, C=22, H=H, W=22), normalising over C=dim-1 (correct channels)
        # BUT freeze will fuse BN into linear using the WRONG dimension mapping
        return self.bn(t)

def reproduce():
    torch.manual_seed(0)
    net = Model()

    # Warm up BatchNorm running statistics
    net.train()
    for _ in range(10):
        net(torch.randn(2, 22, 6, 1))
    net.eval()

    eval_inp = torch.randn(3, 22, 6, 1)

    # Eager (correct reference)
    with torch.no_grad():
        eager = net(eval_inp)

    # JIT: trace -> freeze
    traced = torch.jit.trace(net, [torch.randn(2, 22, 6, 1)])
    frozen = torch.jit.freeze(traced)

    with torch.no_grad():
        jit = frozen(eval_inp)

    diff = (eager - jit).abs().max().item()
    print(f"max absolute difference (eager vs freeze): {diff:.6f}")
    assert diff > 0.01, "Bug not triggered (expected diff > 0.01)"
    print(f"BUG CONFIRMED: torch.jit.freeze gives wrong output (diff={diff:.4f})")
    print(f"  eager mean : {eager.mean().item():.6f}")
    print(f"  frozen mean: {jit.mean().item():.6f}")

if __name__ == "__main__":
    import torch
    print(f"PyTorch version: {torch.__version__}")
    reproduce()
