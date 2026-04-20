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

print(f"eager: {eager.flatten()[:10].tolist()}")
print(f"jit:   {jit.flatten()[:10].tolist()}")
print(f"mismatches: {(eager != jit).sum().item()} / {eager.numel()}")
