"""
Bug 14: TorchJIT numerical inconsistency — torch 2.11.0+cu128
Max diff: 9.307
Ops: ['Input', 'Linear', 'Max', 'Slice', 'PReLU', 'Tan', 'Greater', 'Xor', 'Slice', 'Slice', 'Max']

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
BUG_DIR = Path("/home/binduan/myspace/compilerTesting/nnsmith_results/torchjit_2.11/bug-Symptom.INCONSISTENCY-Stage.VERIFICATION-14")
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
    torch.tensor([3.5752310752868652, 5.897207260131836, 6.956864356994629, 5.2224860191345215, 4.5521087646484375, 6.220518112182617, 5.275952339172363, 3.034585952758789, 6.063022613525391, 3.5950028896331787, 6.593664169311523, 6.343135833740234, 5.593897342681885, 5.810580253601074, 4.854758262634277, 5.327016830444336, 5.099135875701904, 4.500162124633789, 5.008477210998535, 3.3359930515289307, 5.213382720947266, 6.034413814544678, 4.751275539398193, 3.6582274436950684, 3.235354423522949, 6.132234573364258, 4.2089033126831055, 3.650923013687134, 4.128320693969727, 4.377978324890137, 4.5573930740356445, 3.8627424240112305, 6.799679756164551, 3.2947981357574463, 4.990158557891846, 5.691540718078613, 3.4891488552093506, 3.872191905975342, 3.0369253158569336, 6.383854866027832, 3.9606220722198486, 6.783394813537598, 6.238842010498047, 4.630185604095459, 5.422342300415039, 6.080606460571289, 4.007748126983643, 6.862565040588379, 4.439046859741211, 6.01435661315918, 3.0552568435668945, 6.0815324783325195, 4.359716892242432, 4.491901397705078, 6.122427940368652, 5.993467807769775, 4.155457496643066, 6.181275367736816, 4.453871250152588, 3.621965169906616, 3.3267595767974854, 3.6330204010009766], dtype=torch.float32),
]
# eval_inputs: the oracle inputs that expose the divergence
eval_inputs = [
    torch.tensor([6.7209062576293945, 5.489828109741211, 3.158006429672241, 4.2689924240112305, 4.582344055175781, 6.9720563888549805, 5.6774492263793945, 3.7001354694366455, 4.062746047973633, 3.901798725128174, 3.0040061473846436, 5.482828140258789, 4.919044494628906, 4.0414605140686035, 3.2466115951538086, 3.959521770477295, 3.072683095932007, 3.5453622341156006, 3.690312623977661, 5.8282318115234375, 6.81260871887207, 6.593343734741211, 5.242774963378906, 3.4340922832489014, 4.061765670776367, 3.5666275024414062, 6.407768726348877, 5.045027732849121, 3.995128631591797, 6.740717887878418, 4.189964294433594, 6.527456283569336, 3.1232688426971436, 4.436756134033203, 4.450937271118164, 3.253041982650757, 3.6746602058410645, 3.8088722229003906, 6.595870018005371, 3.9375126361846924, 3.062286853790283, 3.2937161922454834, 6.5777201652526855, 3.3309993743896484, 5.2685933113098145, 4.560458660125732, 4.395634174346924, 5.333614349365234, 5.626360893249512, 5.132355213165283, 4.373936176300049, 4.528861045837402, 4.521841049194336, 4.492467403411865, 4.182124137878418, 5.107997894287109, 4.442131996154785, 6.69785213470459, 3.1569905281066895, 4.365805625915527, 3.5236449241638184, 4.499330520629883], dtype=torch.float32),
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

print(f"Bug 14: torch {torch.__version__}")
for i, (e, j) in enumerate(zip(flat(eager_out), flat(jit_out))):
    diff = (e.float() - j.float()).abs().max().item()
    print(f"  output[{i}]: shape={tuple(e.shape)} dtype={e.dtype} max_diff={diff:.4g}")
    if diff > 1e-3:
        print(f"  *** BUG: eager != JIT  (max_diff={diff:.4g}) ***")
