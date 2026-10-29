#!/usr/bin/env python3
"""Prove harm_v1's rule/build code is unchanged after the V5 refactor, and record where it lives now.

harm_v1.json is byte-frozen: its sha256 sits in the cache manifest, Dataset540/541/557's v42_meta.json, the
done markers and every run record, so it cannot follow the code. The refactor moved the rule module
(trainer/v4/harmonize.py -> topocnv/data/harmonize.py) and rewrote the build tool's imports, so the frozen
code_sha256 entries can no longer match the working tree. This tool re-establishes that check with evidence
instead of a byte match:

  * rule module: the calibrated source (git show <commit>:<path>) is exec'd next to the current module and
    both are run on real labels at the production t plus the grid ends; every array must be identical.
  * build tool: after dropping import statements the two texts must differ only in the code_sha256 entry that
    names the rule module; the remaining diff is recorded verbatim.

Writes configs/v4_harm/harm_v1_code_moves.json (write-once unless --force); harm_cache() reads it in place of
a byte match, and still checks the current file's sha256 on every run. Run in the nnU-Net venv:

  python tools/v4_harm_code_moves.py            # check only
  python tools/v4_harm_code_moves.py --write    # check and write the record
"""
import argparse
import difflib
import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from topocnv.data.utils import SPACING, load_label  # noqa: E402
from topocnv.data.v4_splits import all_cases  # noqa: E402
from topocnv.utils.box_ops import mask_Y  # noqa: E402

CFG = REPO / "configs" / "v4_harm" / "harm_v1.json"
OUT = REPO / "configs" / "v4_harm" / "harm_v1_code_moves.json"
RULE = "trainer/v4/harmonize.py"          # calibrated path -> the module that now holds the rule
RULE_NOW = "topocnv/data/harmonize.py"
TOOL = "tools/v4_harmonize_labels.py"     # same path, imports rewritten
TS = (0.10, 0.19, 0.30)                   # grid ends and the calibrated production t
N_THIN, N_OTHER = 8, 4                    # real labels used for the behavioural comparison


def sha(b):
    return hashlib.sha256(b).hexdigest()


def git_show(commit, path):
    return subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"],
                          capture_output=True, check=True).stdout


def calibrated_harmonise(commit):
    """Exec the calibrated rule module with its relative import replaced by the helpers it used."""
    src = git_show(commit, RULE).decode().replace("from .common import SPACING, mask_Y",
                                                  "SPACING, mask_Y = _SPACING, _MASK_Y")
    ns = {"__name__": "calibrated_harmonize", "_SPACING": SPACING, "_MASK_Y": mask_Y}
    exec(compile(src, f"<{commit}:{RULE}>", "exec"), ns)
    return ns


def sample_cases(cfg, limit_thin=N_THIN, limit_other=N_OTHER):
    thin = set(cfg["thin_cases"])
    cases = [c for c in all_cases() if c in thin][:limit_thin]
    cases += [c for c in all_cases() if c not in thin][:limit_other]
    return cases


def check_rule(commit, cfg):
    """Behavioural equivalence of the calibrated and the current rule module on real labels."""
    old = calibrated_harmonise(commit)
    from topocnv.data.harmonize import harmonise as new
    assert tuple(old["SPACING"]) == tuple(SPACING), "SPACING changed since calibration"
    cases = sample_cases(cfg)
    bad, n = [], 0
    for c in cases:
        lab = np.asarray(load_label(c, mmap=False))
        assert np.array_equal(old["mask_Y"](lab), mask_Y(lab)), f"{c}: mask_Y changed since calibration"
        for t in TS:
            a, b = old["harmonise"](lab, t), new(lab, t)
            n += 1
            if a.dtype != b.dtype or not np.array_equal(a, b):
                bad.append((c, t))
    assert not bad, f"the rule module behaves differently from the calibration: {bad[:5]}"
    return cases, n


def strip_imports(text):
    """Drop import statements, including their parenthesized continuation lines."""
    out, depth = [], 0
    for line in text.splitlines():
        if depth or line.startswith(("import ", "from ")):
            depth = max(0, depth + line.count("(") - line.count(")"))
            continue
        out.append(line)
    return out


def check_tool(commit):
    """The build tool may only differ in imports and in the code_sha256 entry that names the rule module."""
    d = list(difflib.unified_diff(strip_imports(git_show(commit, TOOL).decode()),
                                  strip_imports((REPO / TOOL).read_text()),
                                  f"calibrated/{TOOL}", f"current/{TOOL}", lineterm="", n=1))
    changed = [l for l in d if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    bad = [l for l in changed if "code_sha256" not in l and "harmonize.py" not in l]
    assert not bad, f"{TOOL} changed beyond imports and the code_sha256 entry: {bad[:5]}"
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="85a6f2d5ee7420f33db65fa39d6ee47233d25ba8",
                    help="the commit whose copies of harm_v1's code hashes are recorded in harm_v1.json")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite an existing record")
    a = ap.parse_args()

    cfg_bytes = CFG.read_bytes()
    cfg = json.loads(cfg_bytes)
    print(f"harm_v1.json sha256 {sha(cfg_bytes)[:12]}")
    want = {rel: cfg["code_sha256"][rel] for rel in (RULE, TOOL)}
    for rel, w in want.items():
        assert sha(git_show(a.commit, rel)) == w, f"{a.commit}:{rel} is not the calibrated copy"
        print(f"calibrated {rel} {w[:12]} = git {a.commit}")

    cases, n_arrays = check_rule(a.commit, cfg)
    print(f"rule module: {n_arrays} arrays identical on {len(cases)} labels at t = {TS}")
    diff = check_tool(a.commit)
    print(f"build tool: {len([l for l in diff if l[:1] in '+-'])} changed lines, all imports / code_sha256")

    rec = {
        "status": "verified",
        "date": date.today().isoformat(),
        "calibrated_commit": a.commit,
        "why": ("harm_v1.json stays byte-frozen (its sha256 is in the cache manifest, the v42_meta of "
                "540/541/557, the done markers and the run records); the V5 move relocated the rule module and "
                "rewrote the build tool's imports, so harm_cache() accepts the two entries below instead of a "
                "byte match and still pins the current files by sha256."),
        "moves": {
            RULE: {"calibrated_sha256": want[RULE], "current_path": RULE_NOW,
                   "current_sha256": sha((REPO / RULE_NOW).read_bytes()), "check": "behaviour",
                   "evidence": {"cases": cases, "t": list(TS), "arrays_compared": n_arrays, "mismatches": 0}},
            TOOL: {"calibrated_sha256": want[TOOL], "current_path": TOOL,
                   "current_sha256": sha((REPO / TOOL).read_bytes()), "check": "imports-only",
                   "evidence": {"diff": diff}},
        },
    }
    if OUT.exists() and not a.force:
        raise SystemExit(f"{OUT} already exists (write-once); pass --force to rewrite")
    if a.write:
        OUT.write_text(json.dumps(rec, indent=1) + "\n")
        print(f"wrote {OUT.relative_to(REPO)}")
    else:
        print("check only; pass --write to record")


if __name__ == "__main__":
    main()
