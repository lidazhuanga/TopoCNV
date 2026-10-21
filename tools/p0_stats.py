#!/usr/bin/env python3
"""
P0 final statistics (protocol v0.3, binary Y). CPU only; reads the per-case CSVs that
tools/p0_summary.py writes and never re-scores predictions.

For every arm and metric: n, mean ± SD, median [IQR], case-level bootstrap 95% CI of the mean.
For every arm vs Full: paired Wilcoxon signed-rank on cases scored in both, Holm-corrected
across the arms within each metric, paired difference (arm − Full) with bootstrap 95% CI and
per-case win counts. Per fold: per-arm means and the paired difference vs Full, to check that
the direction holds fold by fold. Supplementary contrasts (default A2 vs C1) are reported with
raw p only, outside the Holm family.

Fold membership is read from the run logs (each run prints the validation cases it saved),
so it matches the trainer's split exactly.

Columns are taken from whatever the CSVs contain; metrics in METRICS that are missing are
skipped, so new columns from metrics_protocol.py (Dice_L, Precision_L, Recall_L,
Dice_L_legacy, ...) are picked up automatically.

  python tools/p0_stats.py                      # writes results/p0_stats/
  python tools/p0_stats.py --require_complete   # exit 1 unless every arm has 5/5 folds
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
EVAL_DIR = Path(ac.RESULTS_ROOT) / "p0_binary_y_eval"
OUT_DIR = Path(ac.RESULTS_ROOT) / "p0_stats"
ARMS = ["Full", "A_mem", "A_gate", "C1", "B0", "A2"]
REF = "Full"

# metric -> (+1 higher is better, -1 lower is better, 0 descriptive only), group
METRICS = {
    "Dice_Y": (+1, "Primary"),
    "HD95_Y": (-1, "Primary"),
    "ASSD_Y": (-1, "Primary (supporting)"),
    "Dice_L": (+1, "L1"),
    "Precision_L": (+1, "L1"),
    "Recall_L": (+1, "L1"),
    "Dice_L_legacy": (+1, "L1 (legacy)"),
    "L_over_Y": (0, "L1"),
    "Dice_123": (+1, "Secondary (Stage-1)"),
    "Dice_c4": (+1, "Secondary (Stage-1)"),
}
# metrics whose distribution is heavy-tailed: the paper table shows median [IQR]
MEDIAN_IN_TABLE = {"HD95_Y", "ASSD_Y"}
FAR_HD95_MM = 50.0  # HD95_Y above this counts as a distant false-positive/miss outlier


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


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
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def holm(pvals):
    """Holm step-down adjusted p-values; NaN entries are left out of the family."""
    out = [float("nan")] * len(pvals)
    valid = [(p, i) for i, p in enumerate(pvals) if not math.isnan(p)]
    valid.sort()
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
    res = dict(n=int(ok.sum()))
    if len(d) == 0:
        return res
    lo, hi = boot_ci(d, rng, n_boot)
    res.update(mean_diff=float(d.mean()), median_diff=float(np.median(d)), ci_lo=lo, ci_hi=hi)
    if direction != 0:
        better = d * direction > 0
        res.update(arm_better=int(better.sum()), ref_better=int((d * direction < 0).sum()),
                   ties=int((d == 0).sum()))
    if len(d) >= 6 and np.any(d != 0):
        res["p"] = float(wilcoxon(d).pvalue)
    else:
        res["p"] = float("nan")
    return res


def fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{nd}f}"


def fmt_p(p):
    if p is None or math.isnan(p):
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
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261021)
    ap.add_argument("--extra", default="A2:C1",
                    help="supplementary contrasts arm:ref, comma-separated (raw p, no Holm)")
    ap.add_argument("--require_complete", action="store_true",
                    help="exit 1 unless every arm has all folds COMPLETED and scored")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = args.arms.split(",")

    data, cols = {}, set()
    for arm in arms:
        rows, c = read_arm(arm)
        if rows:
            data[arm] = rows
            cols.update(c)
    metrics = [m for m in METRICS if m in cols]
    cmap, conflicts = fold_map()
    status = {arm: dict(folds_completed=len(folds_done(arm)), n_cases=len(data.get(arm, {})))
              for arm in arms}
    for arm in arms:
        missing = [c for c in data.get(arm, {}) if c not in cmap]
        status[arm]["cases_without_fold"] = len(missing)
    complete = all(s["folds_completed"] == ac.NUM_FOLDS for s in status.values()) and \
        len({s["n_cases"] for s in status.values()}) == 1
    if args.require_complete and not complete:
        print(json.dumps(status, indent=1))
        sys.exit("not complete: some arm lacks 5/5 folds or has a different case count")

    # ---- per-arm descriptives -------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    desc = {}
    desc_rows = []
    for arm in arms:
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

    # ---- paired vs Full, Holm within each metric ----------------------------------------
    pair = {}
    pair_rows = []
    if REF in data:
        for m in metrics:
            direction = METRICS[m][0]
            others = [a for a in arms if a != REF and a in data]
            res = [paired(data[REF], data[a], m, direction, rng, args.n_boot) for a in others]
            adj = holm([r.get("p", float("nan")) for r in res])
            for a, r, pa in zip(others, res, adj):
                r["p_holm"] = pa
                pair[(a, m)] = r
                pair_rows.append(dict(arm=a, ref=REF, metric=m, holm_family=f"{m} ({len(others)} arms vs {REF})", **r))
    extra_rows = []
    for spec in filter(None, args.extra.split(",")):
        a, ref = spec.split(":")
        if a in data and ref in data:
            for m in metrics:
                r = paired(data[ref], data[a], m, METRICS[m][0], rng, args.n_boot)
                extra_rows.append(dict(arm=a, ref=ref, metric=m, **r))
    write_csv(out / "paired_vs_full.csv", pair_rows)
    write_csv(out / "paired_supplementary.csv", extra_rows)

    # ---- per fold -------------------------------------------------------------------------
    fold_rows = []
    for arm in arms:
        if arm not in data:
            continue
        for f in range(ac.NUM_FOLDS):
            fcases = {c for c in data[arm] if cmap.get(c) == f}
            if not fcases:
                continue
            for m in metrics:
                x = np.array([fnum(data[arm][c].get(m)) for c in fcases])
                x = x[np.isfinite(x)]
                row = dict(arm=arm, fold=f, metric=m, n=len(x),
                           mean=float(x.mean()) if len(x) else float("nan"),
                           median=float(np.median(x)) if len(x) else float("nan"))
                if arm != REF and REF in data:
                    r = paired(data[REF], data[arm], m, METRICS[m][0], rng, args.n_boot, cases=fcases)
                    row.update(n_paired=r["n"], mean_diff_vs_full=r.get("mean_diff", float("nan")),
                               p_raw_vs_full=r.get("p", float("nan")))
                fold_rows.append(row)
    write_csv(out / "per_fold.csv", fold_rows)

    # ---- reports --------------------------------------------------------------------------
    stamp = f"{datetime.now():%Y-%m-%d %H:%M}"
    present = [a for a in arms if a in data]
    L = [f"# P0 statistics (binary Y, protocol v0.3) — {stamp}", ""]
    L.append("Status: " + ("**complete** (all arms 5/5 folds, same case set)" if complete
                           else "**INCOMPLETE — interim numbers, not for the paper**"))
    L.append("")
    L.append("| Arm | folds completed | cases scored | cases without fold |")
    L.append("|---|---|---|---|")
    for a in arms:
        s = status[a]
        L.append(f"| {a} | {s['folds_completed']}/{ac.NUM_FOLDS} | {s['n_cases']} | {s['cases_without_fold']} |")
    if conflicts:
        L.append(f"\nWARNING: {len(conflicts)} cases appear in more than one fold in the run logs: "
                 + ", ".join(sorted(conflicts)[:10]))
    L += ["", f"Bootstrap: {args.n_boot} case-level resamples, percentile 95% CI, seed {args.seed}. "
          "Wilcoxon signed-rank, two-sided, on cases scored in both arms; Holm across the "
          f"{len(present) - 1} arms within each metric. Δ = arm − {REF}.", ""]

    L += ["## Descriptives", "", "| Metric | Arm | n | mean ± SD | 95% CI (mean) | median [IQR] |",
          "|---|---|---|---|---|---|"]
    for m in metrics:
        for a in present:
            d = desc[(a, m)]
            if not d["n"]:
                L.append(f"| {m} | {a} | 0 | N/A | | |")
                continue
            extra = f" ({d['n_far']} > {FAR_HD95_MM:.0f} mm)" if "n_far" in d else ""
            L.append(f"| {m} | {a} | {d['n']} | {fmt(d['mean'])} ± {fmt(d['sd'])} | "
                     f"{fmt(d['ci_lo'])}–{fmt(d['ci_hi'])} | {fmt(d['median'])} [{fmt(d['q1'])}–{fmt(d['q3'])}]{extra} |")
    L += ["", f"## Paired vs {REF}", "",
          "| Metric | Arm | n pairs | mean Δ [95% CI] | median Δ | arm better / Full better | p (Wilcoxon) | p (Holm) |",
          "|---|---|---|---|---|---|---|---|"]
    for m in metrics:
        for a in present:
            if a == REF or (a, m) not in pair:
                continue
            r = pair[(a, m)]
            if not r["n"]:
                L.append(f"| {m} | {a} | 0 | N/A | | | | |")
                continue
            wins = f"{r['arm_better']} / {r['ref_better']}" if "arm_better" in r else "—"
            L.append(f"| {m} | {a} | {r['n']} | {fmt(r['mean_diff'], 4)} [{fmt(r['ci_lo'], 4)}, {fmt(r['ci_hi'], 4)}] | "
                     f"{fmt(r['median_diff'], 4)} | {wins} | {fmt_p(r['p'])} | {fmt_p(r['p_holm'])} |")
    if extra_rows:
        L += ["", "## Supplementary contrasts (raw p, not Holm-corrected)", "",
              "| Contrast | Metric | n pairs | mean Δ [95% CI] | p (Wilcoxon) |", "|---|---|---|---|---|"]
        for r in extra_rows:
            if r["n"]:
                L.append(f"| {r['arm']} − {r['ref']} | {r['metric']} | {r['n']} | {fmt(r['mean_diff'], 4)} "
                         f"[{fmt(r['ci_lo'], 4)}, {fmt(r['ci_hi'], 4)}] | {fmt_p(r['p'])} |")
    L += ["", f"## Per fold: mean Δ vs {REF} (Dice_Y, HD95_Y)", ""]
    for m in [x for x in ("Dice_Y", "HD95_Y") if x in metrics]:
        L += [f"**{m}** (mean Δ per fold; last column = folds where the arm is better than {REF})", "",
              "| Arm | " + " | ".join(f"fold {f}" for f in range(ac.NUM_FOLDS)) + " | arm better |",
              "|---|" + "---|" * (ac.NUM_FOLDS + 1)]
        for a in present:
            if a == REF:
                continue
            cells, better, total = [], 0, 0
            for f in range(ac.NUM_FOLDS):
                row = next((r for r in fold_rows if r["arm"] == a and r["fold"] == f and r["metric"] == m), None)
                v = row.get("mean_diff_vs_full", float("nan")) if row and row.get("n_paired") else float("nan")
                cells.append(fmt(v, 4))
                if not math.isnan(v):
                    total += 1
                    better += (v * METRICS[m][0]) > 0
            L.append(f"| {a} | " + " | ".join(cells) + f" | {better}/{total} |")
        L.append("")
    (out / "P0_stats_report.md").write_text("\n".join(L) + "\n")

    # ---- paper table (Markdown + LaTeX) ---------------------------------------------------
    tab_metrics = [m for m in ("Dice_Y", "HD95_Y", "Dice_L", "Recall_L") if m in metrics]

    def cell(a, m, latex=False):
        d = desc.get((a, m), {"n": 0})
        if not d["n"]:
            return "N/A"
        if m in MEDIAN_IN_TABLE:
            return f"{d['median']:.2f} [{d['q1']:.2f}–{d['q3']:.2f}]".replace("–", "--" if latex else "–")
        pm = "$\\pm$" if latex else "±"
        return f"{d['mean']:.3f} {pm} {d['sd']:.3f}"

    def dcell(a, latex=False):
        if a == REF or (a, "Dice_Y") not in pair or not pair[(a, "Dice_Y")]["n"]:
            return "Ref." if a == REF else "—"
        r = pair[(a, "Dice_Y")]
        sep = ", "
        return f"{r['mean_diff']:+.3f} [{r['ci_lo']:+.3f}{sep}{r['ci_hi']:+.3f}]"

    def pcell(a, m):
        if a == REF or (a, m) not in pair:
            return "—"
        return fmt_p(pair[(a, m)].get("p_holm", float("nan")))

    labels = {"Dice_Y": "Dice$_Y$", "HD95_Y": "HD95$_Y$ (mm)", "Dice_L": "Dice$_L$", "Recall_L": "Recall$_L$"}
    md = [f"Table. P0 ablation, out-of-fold (5-fold CV, n = {status[REF]['n_cases']} per arm"
          f"{'' if complete else ', INCOMPLETE'}). Dice: mean ± SD; HD95: median [IQR]. "
          f"ΔDice_Y = arm − {REF}, mean with case-level bootstrap 95% CI. "
          "p: paired Wilcoxon vs Full, Holm-corrected across arms.", "",
          "| Arm | n | " + " | ".join(m for m in tab_metrics) + " | ΔDice_Y vs Full [95% CI] | p Dice_Y | p HD95_Y |",
          "|---|---|" + "---|" * (len(tab_metrics) + 3)]
    tex = ["% generated by tools/p0_stats.py " + stamp, "\\begin{table}[t]", "\\centering\\small",
           "\\caption{P0 ablation on out-of-fold predictions (5-fold CV). Dice: mean $\\pm$ SD; "
           "HD95: median [IQR]. $\\Delta$Dice$_Y$ = arm $-$ Full, mean with case-level bootstrap 95\\% CI. "
           "$p$: paired Wilcoxon signed-rank vs Full, Holm-corrected across arms.}",
           "\\label{tab:p0}", "\\begin{tabular}{l" + "c" * (len(tab_metrics) + 4) + "}", "\\toprule",
           "Arm & $n$ & " + " & ".join(labels[m] for m in tab_metrics)
           + " & $\\Delta$Dice$_Y$ [95\\% CI] & $p$ (Dice$_Y$) & $p$ (HD95$_Y$) \\\\", "\\midrule"]
    for a in present:
        n = desc[(a, "Dice_Y")]["n"] if (a, "Dice_Y") in desc else 0
        md.append(f"| {a} | {n} | " + " | ".join(cell(a, m) for m in tab_metrics)
                  + f" | {dcell(a)} | {pcell(a, 'Dice_Y')} | {pcell(a, 'HD95_Y')} |")
        a_tex = a.replace("_", "\\_")
        tex.append(f"{a_tex} & {n} & "+ " & ".join(cell(a, m, True) for m in tab_metrics)
                   + f" & {dcell(a, True)} & {pcell(a, 'Dice_Y').replace('<', '$<$')} & "
                   f"{pcell(a, 'HD95_Y').replace('<', '$<$')} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "P0_paper_table.md").write_text("\n".join(md) + "\n")
    (out / "P0_paper_table.tex").write_text("\n".join(tex) + "\n")
    (out / "status.json").write_text(json.dumps(dict(generated=stamp, complete=complete, arms=status,
                                                     metrics=metrics, n_boot=args.n_boot, seed=args.seed), indent=1))
    print("\n".join(L))
    print("\n".join(md))


if __name__ == "__main__":
    main()
