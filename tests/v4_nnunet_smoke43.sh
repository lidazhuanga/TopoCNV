#!/bin/bash
# V4.3 smoke (V4_PLAN §12): tiny Dataset540/557/558 copies in a scratch nnU-Net home, then on one GPU,
# 2 epochs x 5 iterations with strong dropout (nnUNetTrainerV43_SMOKE_MMd):
#   1. prep43 mm / shuf: channels as built (MRI in [0, 1], mask 0/1, CT = Dataset540's, reg_ok = 0 case all zero,
#      shuffled channels = shuffled_mri of the derangement donor)
#   2. train43 -> every dropout branch fired; 3-input network; validation_v4 and validation_v4_nomri
#   3. an independent sliding-window prediction from checkpoint_final reproduces both conditions, and the MRI
#      channels change the network output
#   4. zero-impact test (tests/v4_zero_impact43.py): no MRI, no dropout -> same outputs and loss as the stock trainer
# No production dataset or marker is touched; on success writes log/v4/smoke43.done, the gate of the mm43 runs.
#
#  bash tests/v4_nnunet_smoke43.sh <gpu_id>      # or 'queue': use the CUDA_VISIBLE_DEVICES the queue set
set -euo pipefail
# every long step has a 30-min timeout: a hung nnU-Net preprocessing pool (seen 2026-10-28) must not hold a GPU slot
cd "$(dirname "$0")/.."
PY=$HOME/miniconda3/envs/topocnv/bin/python
NNPY=$HOME/cnv/envs/nnunetv2_cnv/bin/python
GPU=${1:?usage: bash tests/v4_nnunet_smoke43.sh <gpu_id>|queue}
[ "$GPU" = queue ] && GPU=${CUDA_VISIBLE_DEVICES:?queue mode needs CUDA_VISIBLE_DEVICES}
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
if [ "$FREE" -lt 40960 ]; then echo "GPU $GPU has $FREE MiB free (< 40 GB): not starting"; exit 1; fi
export CUDA_VISIBLE_DEVICES=$GPU
SCRATCH=$PWD/results/v4_smoke_nn43
REAL_HOME=$HOME/cnv/nnunet_cnv
export CNV_NNUNET_HOME=$SCRATCH/nnunet
rm -rf "$SCRATCH" results/v4_dev_preds/smoke43 results/v4_dev_preds/smoke43_nomri
rm -f log/v4/smoke43.done                   # a failed rerun must not leave an earlier pass in place
mkdir -p "$SCRATCH/nnunet/nnUNet_preprocessed/Dataset501_CNVbinaryY" "$SCRATCH/nnunet/nnUNet_raw" \
         "$SCRATCH/nnunet/nnUNet_results"
cp "$REAL_HOME/nnUNet_preprocessed/Dataset501_CNVbinaryY/nnUNetPlans.json" \
   "$SCRATCH/nnunet/nnUNet_preprocessed/Dataset501_CNVbinaryY/"
# production paths only the V4.3 chain writes; its runs wait for this smoke
marks() { for p in log/v4/nnprep_557.done log/v4/nnprep_558.done log/v4/nn43_v43_mmd1_dev_f0.done \
                   results/v4_dev_preds/v43_mmd1 results/v4_select43 configs/v4_frozen43; do
            [ -e "$p" ] && echo "$p"; done; true; }
MARKS_BEFORE=$(marks)
$NNPY -c "import nnunetv2.training.nnUNetTrainer.variants.zz_cnv_v4.nnUNetTrainerV43" \
  || { echo "V4.3 trainer not installed: run tools/v4_mm43.py install43"; exit 1; }

echo "== [smoke43 1/5] cases, Dataset540 / 557 / 558 subsets =="
read -r CASES VCASE ZCASE < <($PY -c "
import sys; sys.path.insert(0, '.')
from trainer.v4.splits import internal_split
tr, v = internal_split(0)
z = '2500546042'                       # reg_ok = 0 (no MR): MRI and mask all zero
assert z in tr or z in v, z
cases = [c for c in tr if c != z][:2] + [v[0]] + ([z] if z != v[0] else [])
print(','.join(cases), v[0], z)")
echo "cases $CASES (V_0 case $VCASE, reg_ok = 0 case $ZCASE)"
timeout 1800 $NNPY tools/v4_nnunet.py prep42 --what binY --cases "$CASES" --np 4 | tail -1
timeout 1800 $NNPY tools/v4_mm43.py prep43 --what mm --cases "$CASES" --np 4 --skip_verify | tail -1
timeout 1800 $NNPY tools/v4_mm43.py prep43 --what shuf --cases "$CASES" --np 4 --skip_verify | tail -1
$NNPY - "$CASES" "$ZCASE" <<'EOF'
import json
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, "tools")
import v4_mm43 as M
import v4_nnunet as N

cases, z = sys.argv[1].split(","), sys.argv[2]
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
pre = {w: N.NNUNET_HOME / "nnUNet_preprocessed" / n / f"{N.PLANS}_{N.CONFIG}"
       for w, n in (("540", N.V42["binY"][1]), ("mm", M.V43["mm"][1]), ("shuf", M.V43["shuf"][1]))}
ds = {w: nnUNetDataset(str(p), case_identifiers=cases) for w, p in pre.items()}
meta_shuf = json.loads((N.NNUNET_HOME / "nnUNet_raw" / M.V43["shuf"][1] / "v42_meta.json").read_text())
for c in cases:
    d540, _, p540 = ds["540"].load_case(c)
    for w in ("mm", "shuf"):
        d, _, p = ds[w].load_case(c)
        assert d.shape[0] == 3, (c, w, d.shape)
        full = []
        for arr, pp in ((d540[0], p540), (d[0], p)):     # the MRI may widen nnU-Net's nonzero crop: compare uncropped
            f = np.zeros(tuple(pp["shape_before_cropping"]), np.float32)
            f[tuple(slice(a, b) for a, b in pp["bbox_used_for_cropping"])] = np.asarray(arr)
            full.append(f)
        assert np.array_equal(full[0], full[1]), (c, w, "CT channel != Dataset540")
        m = np.asarray(d[2])
        assert set(np.unique(m)) <= {0.0, 1.0}, (c, w, np.unique(m)[:5])
        assert np.asarray(d[1]).min() >= 0 and np.asarray(d[1]).max() <= 1, (c, w)
        if c == z:
            assert not np.asarray(d[1]).any() and not m.any(), (c, w, "reg_ok = 0 case is not all zero")
    if c != z:
        donor = meta_shuf["cases"][c]["donor"]
        assert donor != c, c
        mri, valid, shift = M.shuffled_mri(c, donor)
        raw = N.NNUNET_HOME / "nnUNet_raw" / M.V43["shuf"][1] / "imagesTr"
        back = np.asarray(nib.load(str(raw / f"{c}_0001.nii.gz")).dataobj).transpose(2, 1, 0)
        assert np.array_equal(back, mri) and shift == meta_shuf["cases"][c]["shift"], (c, "558 MRI != shuffled_mri")
print(f"557/558 channels OK for {cases}")
EOF

echo "== [smoke43 2/5] train43 (smoke trainer, strong dropout) =="
timeout 1800 $NNPY tools/v4_mm43.py train43 --what mm --fold 0 --variant dev --trainer nnUNetTrainerV43_SMOKE_MMd \
    --name smoke43 --cases "$VCASE" | tail -2

echo "== [smoke43 3/5] dropout branches, both conditions, independent prediction =="
$NNPY - "$VCASE" <<'EOF'
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "tools")
import v4_mm43 as M
import v4_nnunet as N

v = sys.argv[1]
T = "nnUNetTrainerV43_SMOKE_MMd"
did, name = M.V43["mm"]
split = N.v42_split_index("binY", 0, "dev")
fd = N.results_dir(did, name, T, split)
log = "\n".join(p.read_text() for p in sorted(fd.glob("training_log_*.txt")))
counts = [eval(m) for m in re.findall(r"V4.3 dropout so far: (\{.*\})", log)]
assert counts, "no dropout counts in the training log"
last = counts[-1]
assert last["samples"] > 0 and last["case"] > 0 and last["trunc"] > 0 and last["slab"] > 0, last
print(f"dropout branches fired: {last}")
ck = torch.load(fd / "checkpoint_final.pth", map_location="cpu", weights_only=False)
w = next(t for k, t in ck["network_weights"].items() if t.ndim == 5)
assert w.shape[1] == 3, f"first conv sees {w.shape[1]} channels"
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, perform_everything_on_device=True,
                       device=torch.device("cuda"), verbose=False, verbose_preprocessing=False, allow_tqdm=False)
pred.initialize_from_trained_model_folder(str(fd.parent), use_folds=(split,), checkpoint_name="checkpoint_final.pth")
ds = nnUNetDataset(str(N.NNUNET_HOME / "nnUNet_preprocessed" / name / f"{N.PLANS}_{N.CONFIG}"), case_identifiers=[v])
data, _, props = ds.load_case(v)
x = torch.from_numpy(np.asarray(data, dtype=np.float32).copy())
x[2] = (x[2] > 0.5).float()
x[1] = x[1] * x[2]
shape = tuple(props["shape_before_cropping"])
sl = tuple(slice(a, b) for a, b in props["bbox_used_for_cropping"])
logits = {}
for cond, sub in (("with_mri", "validation_v4"), ("no_mri", "validation_v4_nomri")):
    xi = x.clone()
    if cond == "no_mri":
        xi[1:] = 0
    with torch.no_grad():
        lg = pred.predict_sliding_window_return_logits(xi).float().cpu()
    logits[cond] = lg
    y = np.zeros(shape, np.uint8)
    y[sl] = (torch.softmax(lg[:2], 0)[1] > 0.5).numpy()
    ref = np.load(fd / sub / f"{v}.npz")["Y"]
    agree = float((y == ref).mean())
    assert agree > 0.9999, f"{cond}: independent prediction vs {sub} agree on {agree}"
    out = N.RESULTS / "v4_dev_preds" / ("smoke43" if cond == "with_mri" else "smoke43_nomri") / "fold0"
    assert (out / "EXPORTED_PARTIAL").exists() and not (out / "EXPORTED").exists(), out
    assert f"condition {cond}" in (out / "EXPORTED_PARTIAL").read_text(), out
    print(f"{cond}: independent prediction vs {sub} agree on {agree:.6f}")
dl = float((logits["with_mri"] - logits["no_mri"]).abs().max())
has_mri = bool(x[2].any())
assert dl > 0 or not has_mri, "zeroing the MRI channels does not change the output"
print(f"max |logit(with MRI) - logit(no MRI)| = {dl:.3e} (case has MRI: {has_mri})")
EOF

echo "== [smoke43 4/5] zero-impact test (V4_PLAN §12.8): no MRI + no dropout == stock trainer =="
timeout 1800 $NNPY tests/v4_zero_impact43.py 2>&1 | grep -v -i warn | tail -6
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "SMOKE43 FAIL: zero-impact test"; exit 1; }

echo "== [smoke43 5/5] production markers untouched =="
[ -e log/v4/nn43_smoke43_dev_f0.done ] && { echo "SMOKE43 FAIL: a smoke run wrote a queue marker"; exit 1; }
if [ "$(marks)" != "$MARKS_BEFORE" ]; then echo "SMOKE43 FAIL: a production marker changed"; exit 1; fi
echo "$(date -Iseconds) smoke43 OK on GPU $GPU, HEAD $(git rev-parse --short HEAD), script $(sha256sum tests/v4_nnunet_smoke43.sh | cut -c1-12), trainer $(sha256sum trainer/v4/nnunet/nnUNetTrainerV43.py | cut -c1-12), tool $(sha256sum tools/v4_mm43.py | cut -c1-12)" > log/v4/smoke43.done
echo "SMOKE43_OK (values meaningless; plumbing validated)"
