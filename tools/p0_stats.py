#!/usr/bin/env python3
"""
P0 final statistics (protocol v0.3, binary Y, metrics v2). CPU only: reads the per-case CSVs that
tools/p0_summary.py writes to results/p0_binary_y_eval_v2/ and never re-scores predictions.

Pre-specified test families (Holm correction within each family only):
  F1  Full vs A_mem, A_gate          on Dice_Y and Dice_L   (4 tests)
  F2  C1 vs B0, A2                   on Dice_Y              (2 tests)
  F3  Full vs external networks      on Dice_Y              (one per network; --external)
HD95_Y, Recall_L and Precision_L are reported with paired CIs only, no p-value.
Dice_123 and Dice_c4 are not analysed until the Secondary definition is decided.

Descriptives: mean ± SD, median [IQR], case-level bootstrap 95% CI of the mean (percentile).
Paired differences: arm - reference, case-level bootstrap 95% CI; two-sided Wilcoxon signed-rank.
Per fold: fold membership is read from the run logs ("Saved: <case> stage2 label" lines).

  python tools/p0_stats.py                           # writes results/p0_stats/
  python tools/p0_stats.py --require_complete        # exit 1 unless 6 arms x 154 cases x 5 folds
  python tools/p0_stats.py --external nnUNet,SOTA   # adds family F3 when those CSVs exist
"""
import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MIA" / "scripts"))
import ablation_configs as ac  # noqa: E402

RUN_TAG = "p0_binary_y_20261020"
LOG_DIR = ROOT / "log" / RUN_TAG
EVAL_DIR = Path(ac.RESULTS_ROOT) / "p0_binary_y_eval_v2"
OUT_DIR = Path(ac.RESULTS_ROOT) / "p0_stats"
ARMS = ["Full", "A_mem", "A_gate", "C1", "B0", "A2"]
N_CASES = 154
FAR_HD95_MM = 50.0

FAMILIES = [
    {"name": "F1", "ref": "Full", "arms": ["A_mem", "A_gate"], "tests": ["Dice_Y", "Dice_L"]},
    {"name": "F2", "ref": "C1", "arms": ["B0", "A2"], "tests": ["Dice_Y"]},
    {"name": "F3", "ref": "Full", "arms": [], "tests": ["Dice_Y"]},  # external networks
]
CI_ONLY = ["HD95_Y", "Recall_L", "Precision_L"]
# metric -> direction (+1 higher is better, -1 lower is better, 0 descriptive)
METRICS = {
    "Dice_Y": 1, "HD95_Y": -1, "ASSD_Y": -1, "Dice_L": 1, "Precision_L": 1, "Recall_L": 1,
    "Dice_H": 1, "Dice_L_legacy": 1, "L_over_Y": 0,
}
MEDIAN_IN_TABLE = {"HD95_Y", "ASSD_Y"}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def read_arm(arm):
    p = EVAL_DIR / f"{arm}.csv"
    if not p.exists():
        return None, []
    with open(p) as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = {r["case_id"]: r for r in reader if r["case_id"] != "MEAN"}
    return rows, cols


def fold_map():
    """case_id -> fold, from 'Saved: <case> stage2 label' lines in the run logs."""
    pat = re.compile(r"Saved: (\S+) stage2 label")
    cmap, conflicts = {}, set()
    for log in sorted(LOG_DIR.glob("*_fold*.log")):
        m = re.match(r"(.+)_fold(\d+)\.log$", log.name)
        if not m:
            continue
        fold = int(m.group(2))
        for case in pat.findall(log.read_text(errors="replace")):
            if cmap.setdefault(case, fold) != fold:
                conflicts.add(case)
    return cmap, conflicts


def folds_done(arm):
    return [f for f in range(ac.NUM_FOLDS)
            if (Path(ac.RESULTS_ROOT) / ac.stage2_exp(arm) / f"fold{f}" / "COMPLETED").exists()]


def boot_ci(x, rng, n_boot):
    if len(x) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return float(lo), float(hi)


def holm(pvals):
    """Holm step-down adjusted p-values; NaN entries are left out of the family."""
    out = [float("nan")] * len(pvals)
    valid = sorted((p, i) for i, p in enumerate(pvals) if not math.isnan(p))
    m, running = len(valid), 0.0
    for rank, (p, i) in enumerate(valid):
        running = max(running, min(1.0, (m - rank) * p))
        out[i] = running
    return out


def describe(x, rng, n_boot):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(n=0)
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    lo, hi = boot_ci(x, rng, n_boot)
    return dict(n=len(x), mean=float(x.mean()), sd=float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
                median=float(med), q1=float(q1), q3=float(q3), ci_lo=lo, ci_hi=hi)


def paired(ref_rows, arm_rows, metric, direction, rng, n_boot, cases=None):
    common = sorted(set(ref_rows) & set(arm_rows))
    if cases is not None:
        common = [c for c in common if c in cases]
    a = np.array([fnum(arm_rows[c].get(metric)) for c in common])
    r = np.array([fnum(ref_rows[c].get(metric)) for c in common])
    ok = np.isfinite(a) & np.isfinite(r)
    d = a[ok] - r[ok]
    res = dict(n=int(ok.sum()), p=float("nan"))
    if len(d) == 0:
        return res
    lo, hi = boot_ci(d, rng, n_boot)
    res.update(mean_diff=float(d.mean()), median_diff=float(np.median(d)), ci_lo=lo, ci_hi=hi)
    if direction != 0:
        res.update(arm_better=int((d * direction > 0).sum()), ref_better=int((d * direction < 0).sum()),
                   ties=int((d == 0).sum()))
    if len(d) >= 6 and np.any(d != 0):
        res["p"] = float(wilcoxon(d).pvalue)
    return res


def fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "—"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--external", default="", help="comma-separated external network names for family F3")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--require_complete", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")
    external = [e for e in args.external.split(",") if e]

    families = [dict(f) for f in FAMILIES]
    for f in families:
        if f["name"] == "F3":
            f["arms"] = external

    data, cols = {}, set()
    for arm in arms + external:
        rows, c = read_arm(arm)
        if rows:
            data[arm] = rows
            cols.update(c)
    metrics = [m for m in METRICS if m in cols]
    cmap, conflicts = fold_map()

    status = {}
    for arm in arms:
        status[arm] = dict(folds_completed=len(folds_done(arm)), n_cases=len(data.get(arm, {})),
                           cases_without_fold=sum(c not in cmap for c in data.get(arm, {})))
    complete = all(s["folds_completed"] == ac.NUM_FOLDS and s["n_cases"] == N_CASES for s in status.values())
    if args.require_complete and not complete:
        print(json.dumps(status, indent=1))
        sys.exit("not complete: each arm needs 5/5 folds and 154 cases")

    rng = np.random.default_rng(args.seed)

    # ---- descriptives --------------------------------------------------------------------
    desc, desc_rows = {}, []
    for arm in arms + external:
        if arm not in data:
            continue
        for m in metrics:
            x = np.array([fnum(r.get(m)) for r in data[arm].values()])
            d = describe(x, rng, args.n_boot)
            if m == "HD95_Y" and d["n"]:
                d["n_far"] = int((x[np.isfinite(x)] > FAR_HD95_MM).sum())
            desc[(arm, m)] = d
            desc_rows.append(dict(arm=arm, metric=m, **d))
    write_csv(out / "descriptives.csv", desc_rows)

    # ---- pre-specified families -----------------------------------------------------------
    fam_rows, pair = [], {}
    for f in families:
        ref = f["ref"]
        if ref not in data:
            continue
        rows = []
        for arm in f["arms"]:
            if arm not in data:
                continue
            for m in f["tests"] + CI_ONLY:
                if m not in metrics:
                    continue
                r = paired(data[ref], data[arm], m, METRICS[m], rng, args.n_boot)
                is_test = m in f["tests"]
                if not is_test:
                    r["p"] = float("nan")
                rows.append(dict(family=f["name"], ref=ref, arm=arm, metric=m, is_test=is_test, **r))
        tested = [i for i, r in enumerate(rows) if r["is_test"]]
        adj = holm([rows[i]["p"] for i in tested])
        for i, pa in zip(tested, adj):
            rows[i]["p_holm"] = pa
        for r in rows:
            r.setdefault("p_holm", float("nan"))
            pair[(r["arm"], r["metric"])] = r
        fam_rows += rows
    write_csv(out / "paired_families.csv", fam_rows)

    # ---- per fold -------------------------------------------------------------------------
    cmap_cases = {}
    fold_rows = []
    for arm in arms + external:
        if arm not in data:
            continue
        ref = next((f["ref"] for f in families if arm in f["arms"]), None)
        for fold in range(ac.NUM_FOLDS):
            fcases = {c for c in data[arm] if cmap.get(c) == fold}
            for m in ("Dice_Y", "HD95_Y", "Dice_L"):
                if m not in metrics or not fcases:
                    continue
                x = np.array([fnum(data[arm][c].get(m)) for c in fcases])
                x = x[np.isfinite(x)]
                row = dict(arm=arm, fold=fold, metric=m, n=len(x),
                           mean=float(x.mean()) if len(x) else float("nan"),
                           median=float(np.median(x)) if len(x) else float("nan"))
                if ref in data and arm != ref:
                    r = paired(data[ref], data[arm], m, METRICS[m], rng, args.n_boot, cases=fcases)
                    row.update(ref=ref, n_paired=r["n"], mean_diff=r.get("mean_diff", float("nan")))
                fold_rows.append(row)
    write_csv(out / "per_fold.csv", fold_rows)

    # ---- report ---------------------------------------------------------------------------
    stamp = f"{datetime.now():%Y-%m-%d %H:%M}"
    L = [f"# P0 statistics (binary Y, protocol v0.3, metrics v2) — {stamp}", ""]
    L.append("Status: " + ("**complete** (6 arms × 154 cases × 5 folds)" if complete
                           else "**INCOMPLETE — interim, not for the paper**"))
    L += ["", "| Arm | folds completed | cases | cases without fold |", "|---|---|---|---|"]
    for a in arms:
        s = status[a]
        L.append(f"| {a} | {s['folds_completed']}/{ac.NUM_FOLDS} | {s['n_cases']} | {s['cases_without_fold']} |")
    if conflicts:
        L.append(f"\nWARNING: {len(conflicts)} cases appear in more than one fold: "
                 + ", ".join(sorted(conflicts)[:10]))
    L += ["", f"Bootstrap: {args.n_boot} case-level resamples, percentile 95% CI, seed {args.seed}. "
          "Wilcoxon signed-rank, two-sided. Δ = arm − reference. Holm within each family.",
          "Dice_123 and Dice_c4 are not reported here (Secondary definition pending).", ""]

    L += ["## Descriptives", "", "| Metric | Arm | n | mean ± SD | 95% CI (mean) | median [IQR] |",
          "|---|---|---|---|---|---|"]
    for m in metrics:
        for a in arms + external:
            d = desc.get((a, m))
            if d is None:
                continue
            if not d["n"]:
                L.append(f"| {m} | {a} | 0 | N/A | | |")
                continue
            extra = f" ({d['n_far']} > {FAR_HD95_MM:.0f} mm)" if "n_far" in d else ""
            L.append(f"| {m} | {a} | {d['n']} | {fmt(d['mean'])} ± {fmt(d['sd'])} | "
                     f"{fmt(d['ci_lo'])}–{fmt(d['ci_hi'])} | {fmt(d['median'])} "
                     f"[{fmt(d['q1'])}–{fmt(d['q3'])}]{extra} |")

    L += ["", "## Paired comparisons by family", "",
          "| Family | Ref | Arm | Metric | Test | n pairs | mean Δ [95% CI] | median Δ | arm better / ref better | p (Wilcoxon) | p (Holm, family) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in fam_rows:
        if not r["n"]:
            continue
        wins = f"{r['arm_better']} / {r['ref_better']}" if "arm_better" in r else "—"
        p_raw = fmt_p(r["p"]) if r["is_test"] else "—"
        p_h = fmt_p(r["p_holm"]) if r["is_test"] else "—"
        L.append(f"| {r['family']} | {r['ref']} | {r['arm']} | {r['metric']} | "
                 f"{'yes' if r['is_test'] else 'CI only'} | {r['n']} | "
                 f"{fmt(r['mean_diff'], 4)} [{fmt(r['ci_lo'], 4)}, {fmt(r['ci_hi'], 4)}] | "
                 f"{fmt(r['median_diff'], 4)} | {wins} | {p_raw} | {p_h} |")
    if not external:
        L.append("\nF3 (external networks) has no CSVs yet; it is reported only when cnv-sota's results are added via --external.")

    L += ["", "## Per fold (mean Δ vs reference)", ""]
    for m in ("Dice_Y", "Dice_L", "HD95_Y"):
        L += [f"**{m}**", "", "| Arm | ref | " + " | ".join(f"fold {f}" for f in range(ac.NUM_FOLDS))
              + " | arm better |", "|---|---|" + "---|" * (ac.NUM_FOLDS + 1)]
        for arm in arms + external:
            ref = next((f["ref"] for f in families if arm in f["arms"]), None)
            if ref is None:
                continue
            cells, better, total = [], 0, 0
            for f in range(ac.NUM_FOLDS):
                row = next((r for r in fold_rows if r["arm"] == arm and r["fold"] == f and r["metric"] == m), None)
                v = row.get("mean_diff", float("nan")) if row else float("nan")
                cells.append(fmt(v, 4))
                if not math.isnan(v):
                    total += 1
                    better += (v * METRICS[m]) > 0
            L.append(f"| {arm} | {ref} | " + " | ".join(cells) + f" | {better}/{total} |")
        L.append("")
    (out / "P0_stats_report.md").write_text("\n".join(L) + "\n")

    # ---- paper table ----------------------------------------------------------------------
    tab_arms = [a for a in arms + external if a in data]
    tab_metrics = [m for m in ("Dice_Y", "HD95_Y", "Dice_L", "Recall_L", "Precision_L") if m in metrics]
    labels = {"Dice_Y": "Dice$_Y$", "HD95_Y": "HD95$_Y$ (mm)", "Dice_L": "Dice$_L$", "Recall_L": "Recall$_L$",
              "Precision_L": "Precision$_L$"}

    def cell(a, m, latex=False):
        d = desc.get((a, m), {"n": 0})
        if not d["n"]:
            return "N/A"
        if m in MEDIAN_IN_TABLE:
            return f"{d['median']:.2f} [{d['q1']:.2f}–{d['q3']:.2f}]"
        pm = "$\\pm$" if latex else "±"
        return f"{d['mean']:.3f} {pm} {d['sd']:.3f}"

    def dcell(a, latex=False):
        r = pair.get((a, "Dice_Y"))
        if r is None:
            return "Ref."
        if not r["n"]:
            return "—"
        return f"{r['mean_diff']:+.3f} [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]"

    def ref_of(a):
        return next((f["ref"] for f in families if a in f["arms"]), "—")

    def pcell(a, m):
        r = pair.get((a, m))
        return "—" if r is None or not r.get("is_test") else fmt_p(r["p_holm"])

    md = [f"Table. P0 ablation, out-of-fold, 5-fold CV, n = 154 cases per arm"
          f"{'' if complete else ' (INCOMPLETE)'}. Dice and Recall/Precision: mean ± SD; HD95: median [IQR]. "
          "Δ Dice$_Y$ = arm − reference (ref. column), mean with case-level bootstrap 95% CI. "
          "p: two-sided Wilcoxon, Holm within family (F1: Full vs A_mem/A_gate on Dice$_Y$ and Dice$_L$; "
          "F2: C1 vs B0/A2 on Dice$_Y$).", "",
          "| Arm | ref. | " + " | ".join(tab_metrics) + " | Δ Dice$_Y$ [95% CI] | p Dice$_Y$ | p Dice$_L$ |",
          "|---|---|" + "---|" * (len(tab_metrics) + 3)]
    tex = ["% generated by tools/p0_stats.py " + stamp, "\\begin{table}[t]", "\\centering\\small",
           "\\caption{P0 ablation, out-of-fold, 5-fold CV (n = 154). Dice, Recall and Precision: mean $\\pm$ SD; "
           "HD95: median [IQR]. $\\Delta$Dice$_Y$ = arm $-$ reference, mean with case-level bootstrap 95\\% CI. "
           "$p$: two-sided Wilcoxon signed-rank, Holm-corrected within family.}",
           "\\label{tab:p0}", "\\begin{tabular}{ll" + "c" * (len(tab_metrics) + 3) + "}", "\\toprule",
           "Arm & ref. & " + " & ".join(labels[m] for m in tab_metrics)
           + " & $\\Delta$Dice$_Y$ [95\\% CI] & $p$ Dice$_Y$ & $p$ Dice$_L$ \\\\", "\\midrule"]
    for a in tab_arms:
        ref = ref_of(a)
        md.append(f"| {a} | {ref} | " + " | ".join(cell(a, m) for m in tab_metrics)
                  + f" | {dcell(a)} | {pcell(a, 'Dice_Y')} | {pcell(a, 'Dice_L')} |")
        a_tex = a.replace("_", "\\_")
        p_y = pcell(a, "Dice_Y").replace("<", "$<$")
        p_l = pcell(a, "Dice_L").replace("<", "$<$")
        tex.append(f"{a_tex} & {ref} & " + " & ".join(cell(a, m, True) for m in tab_metrics)
                   + f" & {dcell(a, True)} & {p_y} & {p_l} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "P0_paper_table.md").write_text("\n".join(md) + "\n")
    (out / "P0_paper_table.tex").write_text("\n".join(tex) + "\n")
    (out / "status.json").write_text(json.dumps(dict(generated=stamp, complete=complete, arms=status,
                                                     metrics=metrics, external=external,
                                                     n_boot=args.n_boot, seed=args.seed), indent=1))
    print("\n".join(L))
    print("\n".join(md))


if __name__ == "__main__":
    main()
