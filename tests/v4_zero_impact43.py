"""V4.3 zero-impact test (V4_PLAN §12.8 point 1), on GPU, inside the smoke43 scratch nnU-Net home.

With MRI and mask all zero and no dropout, nnUNetTrainerV43_MM must compute what the stock nnUNetTrainer_300epochs
computes: same network outputs when every weight is shared (first conv: the CT column copied, the MRI and mask
columns arbitrary non-zero), and the same loss. fp32, TF32 off. Differences that come from the extra input
channels (first-conv init fan-in, CT-column gradients under real MRI, nonzero crop) are not tested here; they
are listed in §12.8.

  CNV_NNUNET_HOME=<smoke home> python tests/v4_zero_impact43.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("nnUNet_compile", "f")
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import v4_mm43 as M  # noqa: E402
import v4_nnunet as N  # noqa: E402

os.environ.update(N.env())        # nnunetv2.paths reads nnUNet_* at import: set them before any nnunetv2 import

torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
dev = torch.device("cuda")
pre = N.NNUNET_HOME / "nnUNet_preprocessed"


def trainer(cls_name, dname):
    plans = json.loads((pre / dname / f"{N.PLANS}.json").read_text())
    dj = json.loads((pre / dname / "dataset.json").read_text())
    t = N.trainer_class(cls_name)(plans, N.CONFIG, N.v42_split_index("binY", 0, "dev"), dj, device=dev)
    t.initialize()
    return t


t0 = trainer(N.R0_TRAINER, N.V42["binY"][1])
t43 = trainer("nnUNetTrainerV43_MM", M.V43["mm"][1])
sd0, sd43 = t0.network.state_dict(), t43.network.state_dict()
assert set(sd0) == set(sd43), "the two networks have different layers"
g = torch.Generator(device="cpu").manual_seed(0)
diff_keys = []
for k in sd0:
    if sd0[k].shape == sd43[k].shape:
        sd43[k] = sd0[k].clone()
    else:
        diff_keys.append(k)
        assert sd43[k].shape[1] == 3 and sd0[k].shape[1] == 1 and sd43[k].shape[2:] == sd0[k].shape[2:], k
        w = torch.randn(sd43[k].shape, generator=g).to(sd43[k])
        w[:, :1] = sd0[k]
        sd43[k] = w
assert len(diff_keys) == 1, f"only the first conv may differ, got {diff_keys}"
t43.network.load_state_dict(sd43)
t0.network.train()
t43.network.train()

patch = [int(p) for p in t0.configuration_manager.patch_size]
x1 = torch.randn([2, 1] + patch, generator=g).to(dev)
x3 = torch.cat([x1, torch.zeros_like(x1), torch.zeros_like(x1)], 1)
assert not t43._dropping and not t43._no_mri
with torch.no_grad():
    o0 = t0.network(x1)
    o43 = t43.network(t43._mask(x3))
    seg = (torch.rand([2, 1] + patch, generator=g) > 0.9).float().to(dev)
    tg = [torch.nn.functional.interpolate(seg, size=o.shape[2:], mode="nearest") for o in o0]
    l0 = float(t0.loss(o0, tg))
    l43 = float(t43.loss(o43, tg))
    l43_same = float(t43.loss(o0, tg))
    scale = max(1.0, max(float(o.abs().max()) for o in o0))
    dmax = max(float((a - b).abs().max()) for a, b in zip(o0, o43))
    x3r = x3.clone()
    x3r[:, 1] = torch.rand_like(x1)
    x3r[:, 2] = 1.0
    dmri = max(float((a - b).abs().max()) for a, b in zip(o0, t43.network(t43._mask(x3r))))
print(f"first conv differs only: {diff_keys[0]}")
print(f"outputs (no MRI): max |V43 - stock| = {dmax:.3e} (output scale {scale:.2f})")
print(f"loss: stock {l0:.8f}  V43 {l43:.8f}  V43 loss on the stock outputs {l43_same:.8f}  (diff {abs(l43_same - l0):.3e})")
print(f"with MRI and mask set: max |V43 - stock| = {dmri:.3e}")
assert dmax <= 1e-4 * scale, "with no MRI the V4.3 network does not reproduce the stock network"
assert abs(l43_same - l0) <= 1e-6 * abs(l0), "the V4.3 loss path differs from the stock loss on identical outputs"
assert abs(l43 - l0) <= 1e-5 * abs(l0), "the V4.3 loss differs from the stock loss"
assert dmri > 1e-3, "the MRI channels have no effect on the output"
print("ZERO_IMPACT_OK")
