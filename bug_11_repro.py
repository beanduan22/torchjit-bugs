"""
Bug 11: TorchJIT numerical inconsistency — torch 2.11.0+cu128
Max diff: 0.04404
Ops: ['Constant', 'Input', 'Sub', 'Input', 'Constant', 'Mul', 'Add', 'Input', 'Mul', 'Flatten', 'ArgMax', 'Concat4', 'GELU', 'Linear', 'Cos']

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
BUG_DIR = Path("/home/binduan/myspace/compilerTesting/nnsmith_results/torchjit_2.11/bug-Symptom.INCONSISTENCY-Stage.VERIFICATION-11")
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
    torch.tensor(4.107760429382324, dtype=torch.float32),
    torch.tensor(3.4803454875946045, dtype=torch.float32),
    torch.tensor([[6.099553108215332], [3.0584356784820557], [6.102806091308594], [3.03694486618042], [3.4803035259246826], [4.718481540679932], [3.7851674556732178], [4.730016708374023], [6.246649265289307], [6.799515724182129], [5.716449737548828], [5.338762283325195], [3.526850700378418], [6.888127326965332], [3.3615617752075195], [3.5724222660064697], [5.784472465515137], [6.8607378005981445], [5.781941890716553], [4.701910972595215], [4.145702362060547], [6.121409893035889], [6.518064975738525], [3.525031805038452], [6.635618209838867], [5.550836086273193], [3.590496778488159], [5.549510955810547], [3.5801074504852295], [5.110598564147949], [5.039118766784668], [3.130086898803711], [3.9718496799468994], [6.301724433898926], [3.0243351459503174], [3.086588144302368], [5.428722381591797], [3.9977595806121826], [4.6824421882629395], [5.790262222290039], [5.21651554107666], [3.102449655532837], [6.502443313598633], [6.679350852966309], [6.799726486206055], [3.9627766609191895], [6.604555130004883], [6.160857200622559], [6.416184425354004], [6.069120407104492], [6.840721607208252], [3.939312219619751], [5.7793426513671875], [5.641592025756836], [3.144731283187866]], dtype=torch.float32),
]
# eval_inputs: the oracle inputs that expose the divergence
eval_inputs = [
    torch.tensor(4.241179466247559, dtype=torch.float32),
    torch.tensor(6.3883771896362305, dtype=torch.float32),
    torch.tensor([[4.8968963623046875], [6.533715724945068], [3.8574459552764893], [5.026226997375488], [3.8147964477539062], [3.350688934326172], [6.174614906311035], [5.279245376586914], [6.912731170654297], [5.6704301834106445], [4.0318450927734375], [6.554147720336914], [6.571943759918213], [4.469766616821289], [4.5201945304870605], [5.106853008270264], [5.623481750488281], [5.106599807739258], [4.498797416687012], [6.805603981018066], [6.61827278137207], [6.24208927154541], [4.859267234802246], [4.894802093505859], [4.39981746673584], [5.654217720031738], [5.457810401916504], [4.269301414489746], [4.692037105560303], [3.2127249240875244], [4.555237293243408], [5.900867462158203], [4.464590072631836], [4.7258501052856445], [6.5186052322387695], [5.1351823806762695], [3.280407428741455], [6.777009963989258], [4.901080131530762], [3.158543586730957], [6.260950088500977], [6.053189277648926], [3.7449958324432373], [3.644169569015503], [5.558067798614502], [4.563841819763184], [5.325362682342529], [3.271929979324341], [5.916488170623779], [3.215620279312134], [3.5148203372955322], [6.320707321166992], [4.466443061828613], [3.5520174503326416], [3.1970841884613037]], dtype=torch.float32),
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

print(f"Bug 11: torch {torch.__version__}")
for i, (e, j) in enumerate(zip(flat(eager_out), flat(jit_out))):
    diff = (e.float() - j.float()).abs().max().item()
    print(f"  output[{i}]: shape={tuple(e.shape)} dtype={e.dtype} max_diff={diff:.4g}")
    if diff > 1e-3:
        print(f"  *** BUG: eager != JIT  (max_diff={diff:.4g}) ***")
