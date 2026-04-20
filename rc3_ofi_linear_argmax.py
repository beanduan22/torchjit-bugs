import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(43, 43)

    def forward(self, x):
        return torch.argmax(self.linear(x), dim=1)


torch.manual_seed(17)
net = Model().eval()

x = torch.ones(28, 20, 43)

with torch.no_grad():
    eager = net(x)

opt = torch.jit.optimize_for_inference(torch.jit.freeze(torch.jit.trace(net, [x])))

with torch.no_grad():
    jit = opt(x)

mask = eager != jit
rows, cols = mask.nonzero(as_tuple=True)
print("mismatched positions (batch, feature) -> eager / jit:")
for r, c in zip(rows[:5], cols[:5]):
    print(f"  [{r.item():2d}, {c.item():2d}]  eager={eager[r,c].item()}  jit={jit[r,c].item()}")
print(f"total: {mask.sum().item()} / {eager.numel()}")
