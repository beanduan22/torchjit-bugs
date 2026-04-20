"""
Bug 24: TorchJIT numerical inconsistency — torch 2.11.0+cu128
Max diff: 33.18
Ops: ['Constant', 'Input', 'Sub', 'Atan', 'Ceil', 'Slice', 'Concat5', 'ReduceMin', 'Slice', 'Linear', 'NearestInterp', 'BatchNorm2d']

Root cause: torch.jit.optimize_for_inference rewrites the graph
in a way that changes numerical results when inputs differ from
the tracing inputs.

Requirements: pip install nnsmith torch==2.11.0
"""
import torch, warnings, pickle
warnings.filterwarnings("ignore")
from pathlib import Path
from nnsmith.materialize.torch import TorchModelCPU

# ── Load model from bug dir ────────────────────────────────────────────────
BUG_DIR = Path("/home/binduan/myspace/compilerTesting/nnsmith_results/torchjit_2.11/bug-Symptom.INCONSISTENCY-Stage.VERIFICATION-24")
with open(BUG_DIR / "gir.pkl", "rb") as f:
    gir = pickle.load(f)
model = TorchModelCPU.from_gir(gir)
model.torch_model.load_state_dict(
    torch.load(BUG_DIR / "model.pth", weights_only=False)
)
model.torch_model.eval()
net = model.torch_model

# ── Inputs ─────────────────────────────────────────────────────────────────
# trace_inputs: used to trace the JIT graph (random, as the backend does)
trace_inputs = [
    torch.tensor([[3.895775556564331, 5.396644592285156]], dtype=torch.float32),
]
# eval_inputs: the oracle inputs that expose the divergence
eval_inputs = [
    torch.tensor([[3.8583176136016846, 6.461281776428223]], dtype=torch.float32),
]

# ── Eager forward ──────────────────────────────────────────────────────────
with torch.no_grad():
    eager_out = net(*eval_inputs)

# ── JIT: trace → freeze → optimize_for_inference ───────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    jit_model = torch.jit.trace(net, trace_inputs)
    jit_model = torch.jit.freeze(jit_model)
    jit_model = torch.jit.optimize_for_inference(jit_model)

with torch.no_grad():
    jit_out = jit_model(*eval_inputs)

# ── Compare ────────────────────────────────────────────────────────────────
def flat(x):
    if isinstance(x, dict): return list(x.values())
    if isinstance(x, (list, tuple)): return list(x)
    return [x]

print(f"Bug 24: torch {torch.__version__}")
for i, (e, j) in enumerate(zip(flat(eager_out), flat(jit_out))):
    diff = (e.float() - j.float()).abs().max().item()
    print(f"  output[{i}]: shape={tuple(e.shape)} dtype={e.dtype} max_diff={diff:.4g}")
    if diff > 1e-3:
        print(f"  *** BUG: eager != JIT  (max_diff={diff:.4g}) ***")
