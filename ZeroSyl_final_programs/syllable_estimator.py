"""Estimate the number of syllables per file from ZeroSyl output alone (no ground truth).

Three steps:

  fit       learn a correction on a set WITH ground truth (Babaloon dev)
            -> output/calibration/<name>.json
  predict   apply it to any ZeroSyl output folder, NO ground truth needed
            -> output/estimates/<name>_<folder>.csv  (estimated syllables per file and per second)
  evaluate  compare the estimates with ground truth on a held-out set (Babaloon test)
            -> output/results/SyllableEstimator_<name>_<folder>.txt / .csv
            -> output/plots/SyllableEstimator_<name>_<folder>.png

Syllables are counted exactly like syllable_count_plot.py:
  ground truth = non-empty intervals in the "syllables" tier of each TextGrid
  ZeroSyl      = segments in each .pt file, excluding silence segments (the largest id)
File duration is the end frame of the last segment / 50 Hz, so it also needs no ground truth.

Estimators (all learned on the fit set only; the test set is never used for fitting):
  raw        ZeroSyl count, uncorrected
  inverse    your dev line ZeroSyl = m*GT + c, flipped:  GT = (ZeroSyl - c) / m
  direct     least-squares line fitted the other way:   GT = a*ZeroSyl + b
  rate       per-second correction:  GT rate = a*(ZeroSyl rate) + b,  times duration
  duration   "stupid" estimator, ignores ZeroSyl:  GT = dev syllables-per-second * duration

Usage (from ~/ZeroSyl-master):

  python syllable_estimator.py fit \
      data/babaloonAlignments/dev/ output/segments/ZeroSylCollapsed/Babaloon/dev/ --name Babaloon

  python syllable_estimator.py predict \
      output/segments/ZeroSylCollapsed/Babaloon/test/ --name Babaloon

  python syllable_estimator.py evaluate \
      data/babaloonAlignments/test/ output/segments/ZeroSylCollapsed/Babaloon/test/ --name Babaloon
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

FRAME_RATE = 50.0
ESTIMATORS = ["raw", "inverse", "direct", "rate", "duration"]
DESCRIPTION = {
    "raw": "Raw ZeroSyl count (no correction)",
    "inverse": "Flipped dev line  GT = (ZS - c) / m",
    "direct": "Direct dev fit    GT = a*ZS + b",
    "rate": "Rate fit          GT/s = a*ZS/s + b",
    "duration": "Duration only     GT = rate * seconds",
}


# ------------------------------------------------------------------ data loading


def load_predictions(pred_dir: Path, sil_id=None, keep_silence=False):
    """{stem: (zerosyl_count, duration_s)} counted like syllable_count_plot.py."""
    files = sorted(pred_dir.rglob("*.pt"))
    if not files:
        raise SystemExit(f"No .pt files found in {pred_dir}")
    segs = {p.stem: torch.load(p, map_location="cpu") for p in files}

    if not keep_silence and sil_id is None:
        ids = torch.cat([s[:, 2] for s in segs.values() if s.numel()])
        if ids.unique().numel() > 1:
            sil_id = int(ids.max())
    if keep_silence:
        sil_id = None

    out = {}
    for stem, s in segs.items():
        if s.numel() == 0:
            out[stem] = (0, 0.0)
            continue
        n = s.shape[0] if sil_id is None else int((s[:, 2] != sil_id).sum())
        out[stem] = (n, int(s[-1, 1]) / FRAME_RATE)
    return out, sil_id


def load_ground_truth(gt_dir: Path, tier: str):
    import tgt

    files = sorted(gt_dir.rglob("*.TextGrid"))
    if not files:
        raise SystemExit(f"No .TextGrid files found in {gt_dir}")
    out = {}
    for p in files:
        tg = tgt.read_textgrid(p, include_empty_intervals=True)
        out[p.stem] = sum(1 for i in tg.get_tier_by_name(tier).intervals if i.text.strip())
    return out


def paired(gt_dir, pred_dir, tier, sil_id, keep_silence):
    gt = load_ground_truth(gt_dir, tier)
    pred, sil = load_predictions(pred_dir, sil_id, keep_silence)
    names = sorted(set(gt) & set(pred))
    missing = (set(gt) ^ set(pred))
    if missing:
        print(f"WARNING: {len(missing)} files without a partner are ignored, e.g. "
              f"{sorted(missing)[:3]}")
    if len(names) < 3:
        raise SystemExit("Need at least 3 matching files.")
    x = np.array([gt[n] for n in names], float)           # ground truth
    y = np.array([pred[n][0] for n in names], float)      # ZeroSyl count
    d = np.array([pred[n][1] for n in names], float)      # duration (s)
    return names, x, y, d, sil


# ------------------------------------------------------------------ estimators


def apply(cal, zs, dur):
    """All estimates for arrays of ZeroSyl counts and durations."""
    with np.errstate(divide="ignore", invalid="ignore"):
        zs_rate = np.where(dur > 0, zs / dur, 0.0)
    est = {
        "raw": zs.astype(float),
        "inverse": (zs - cal["inverse"]["c"]) / cal["inverse"]["m"],
        "direct": cal["direct"]["a"] * zs + cal["direct"]["b"],
        "rate": (cal["rate"]["a"] * zs_rate + cal["rate"]["b"]) * dur,
        "duration": cal["duration"]["syllables_per_s"] * dur,
    }
    return {k: np.clip(v, 0, None) for k, v in est.items()}  # no negative counts


def cmd_fit(args):
    names, x, y, d, sil = paired(args.gt_dir, args.pred_dir, args.tier,
                                 args.sil_id, args.keep_silence)
    m, c = np.polyfit(x, y, 1)              # ZS = m*GT + c (what syllable_count_plot draws)
    a, b = np.polyfit(y, x, 1)              # GT = a*ZS + b
    ra, rb = np.polyfit(y / d, x / d, 1)    # GT/s = ra*(ZS/s) + rb
    r = float(np.corrcoef(x, y)[0, 1])

    cal = {
        "fitted_on": {"ground_truth": str(args.gt_dir), "predictions": str(args.pred_dir),
                      "files": len(names), "pearson_r": r},
        "counting": {"tier": args.tier, "silence_id": sil, "keep_silence": args.keep_silence},
        "inverse": {"m": m, "c": c},
        "direct": {"a": a, "b": b},
        "rate": {"a": ra, "b": rb},
        "duration": {"syllables_per_s": float(x.sum() / d.sum())},
        "zerosyl_syllables_per_s": float(y.sum() / d.sum()),
    }
    out = Path("output/calibration") / f"{args.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cal, indent=2))

    print(f"Fitted on {len(names)} files ({d.sum() / 60:.1f} min of audio), Pearson r = {r:.3f}")
    print(f"  dev line (as in syllable_count_plot): ZS = {m:.2f}*GT {c:+.1f}")
    print(f"  inverse  : GT = (ZS {-c:+.1f}) / {m:.2f}")
    print(f"  direct   : GT = {a:.3f}*ZS {b:+.1f}")
    print(f"  rate     : GT/s = {ra:.3f}*ZS/s {rb:+.3f}")
    print(f"  duration : GT = {cal['duration']['syllables_per_s']:.3f} syll/s * seconds "
          f"(ZeroSyl gives {cal['zerosyl_syllables_per_s']:.3f} syll/s)")
    print(f"Saved calibration: {out}")


def load_cal(name):
    p = Path("output/calibration") / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"No calibration {p}. Run `fit` first.")
    return json.loads(p.read_text())


def tag(pred_dir: Path, name: str = ""):
    """Short label for output files, e.g. 'test' for .../ZeroSylCollapsed/Babaloon/test."""
    parts = list(pred_dir.resolve().parts[-2:])
    if parts[0] == name:
        parts = parts[1:]
    return "_".join(parts)


def cmd_predict(args):
    cal = load_cal(args.name)
    cnt = cal["counting"]
    pred, _ = load_predictions(args.pred_dir, cnt["silence_id"], cnt["keep_silence"])
    names = sorted(pred)
    zs = np.array([pred[n][0] for n in names], float)
    d = np.array([pred[n][1] for n in names], float)
    est = apply(cal, zs, d)[args.estimator]

    out = Path("output/estimates") / f"{args.name}_{tag(args.pred_dir, args.name)}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "duration_s", "zerosyl_count", "estimated_syllables",
                    "estimated_syllables_per_s"])
        for n, z, dd, e in zip(names, zs, d, est):
            w.writerow([n, f"{dd:.2f}", int(z), f"{e:.1f}", f"{e / dd if dd else 0:.3f}"])

    print(f"Estimator '{args.estimator}' ({DESCRIPTION[args.estimator]}) on {len(names)} files")
    print(f"  total estimated syllables: {est.sum():.0f}  "
          f"({est.sum() / d.sum():.2f} per second)")
    print(f"Saved: {out}")


# ------------------------------------------------------------------ evaluation


def metrics(gt, est, dur):
    err = est - gt
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.abs(err) / gt * 100
    return {
        "MAE": np.mean(np.abs(err)),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "bias": np.mean(err),
        "MAPE": np.mean(pct[np.isfinite(pct)]),
        "total_err_pct": (est.sum() - gt.sum()) / gt.sum() * 100,
        "rate_MAE": np.mean(np.abs(err / dur)),
        "r": np.corrcoef(gt, est)[0, 1] if np.std(est) > 0 else np.nan,
    }


def bootstrap(gt, ests, n_boot, seed=0):
    """95% CI of MAE per estimator, and of MAE(estimator) - MAE(duration)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gt), size=(n_boot, len(gt)))
    abs_err = {k: np.abs(v - gt) for k, v in ests.items()}
    mae = {k: abs_err[k][idx].mean(1) for k in ests}
    ci = {k: np.percentile(mae[k], [2.5, 97.5]) for k in ests}
    diff = {k: np.percentile(mae[k] - mae["duration"], [2.5, 97.5]) for k in ests}
    return ci, diff


def cmd_evaluate(args):
    cal = load_cal(args.name)
    cnt = cal["counting"]
    names, gt, zs, d, _ = paired(args.gt_dir, args.pred_dir, cnt["tier"],
                                 cnt["silence_id"], cnt["keep_silence"])
    ests = apply(cal, zs, d)
    res = {k: metrics(gt, v, d) for k, v in ests.items()}
    ci, diff = bootstrap(gt, ests, args.bootstrap)

    best = min(ESTIMATORS, key=lambda k: res[k]["MAE"])
    L = [f"Syllable estimator evaluation: calibration '{args.name}' "
         f"(fitted on {cal['fitted_on']['files']} files)",
         f"Evaluated on {len(names)} files in {args.pred_dir}",
         f"Ground truth: {gt.sum():.0f} syllables, mean {gt.mean():.1f} per file, "
         f"{gt.sum() / d.sum():.2f} per second", "",
         "Per-file syllable count (error = estimate - ground truth)", ""]
    hdr = (f"{'estimator':<38} {'MAE':>6} {'95% CI':>15} {'RMSE':>6} {'bias':>7} "
           f"{'MAPE%':>6} {'total%':>7} {'r':>6} {'rateMAE':>8}")
    L += [hdr, "-" * len(hdr)]
    for k in ESTIMATORS:
        m = res[k]
        L.append(f"{DESCRIPTION[k]:<38} {m['MAE']:6.1f} "
                 f"{f'[{ci[k][0]:.1f}, {ci[k][1]:.1f}]':>15} {m['RMSE']:6.1f} "
                 f"{m['bias']:+7.1f} {m['MAPE']:6.1f} {m['total_err_pct']:+7.1f} "
                 f"{m['r']:6.3f} {m['rate_MAE']:8.3f}")
    L += ["",
          "MAE     mean absolute error in syllables per file (lower is better)",
          "95% CI  bootstrap interval of the MAE over files "
          f"({args.bootstrap} resamples)",
          "bias    mean signed error: + overcounts, - undercounts",
          "MAPE%   mean absolute error as % of the file's true count",
          "total%  error of the summed count over the whole set",
          "rateMAE mean absolute error in syllables per second", "",
          f"Best estimator on this set: {best} ({DESCRIPTION[best]}), "
          f"MAE {res[best]['MAE']:.1f} syllables/file", "",
          "Does ZeroSyl beat the duration-only estimator?  "
          "(95% CI of MAE difference; negative = better than duration)"]
    for k in ["raw", "inverse", "direct", "rate"]:
        lo, hi = diff[k]
        verdict = ("better" if hi < 0 else "worse" if lo > 0 else "no clear difference")
        L.append(f"  {k:<9} {res[k]['MAE'] - res['duration']['MAE']:+6.1f}  "
                 f"[{lo:+.1f}, {hi:+.1f}]  -> {verdict}")
    table = "\n".join(L)
    print(table)

    t = tag(args.pred_dir, args.name)
    res_dir = Path("output/results")
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / f"SyllableEstimator_{args.name}_{t}.txt").write_text(table + "\n")
    with open(res_dir / f"SyllableEstimator_{args.name}_{t}_perfile.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "duration_s", "ground_truth", "zerosyl_count"]
                   + [f"est_{k}" for k in ESTIMATORS])
        for i, n in enumerate(names):
            w.writerow([n, f"{d[i]:.2f}", int(gt[i]), int(zs[i])]
                       + [f"{ests[k][i]:.1f}" for k in ESTIMATORS])

    # ---- plot: estimate vs ground truth per estimator + MAE bars
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(14, 9.5))
    lim = [0, max(gt.max(), max(v.max() for v in ests.values())) * 1.05]
    for ax, k in zip(axes.flat, ESTIMATORS):
        ax.scatter(gt, ests[k], s=26, alpha=0.8, color="#2a6fdb", edgecolors="white",
                   linewidths=0.5, zorder=3)
        ax.plot(lim, lim, ls="--", color="grey", lw=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(DESCRIPTION[k], fontsize=10)
        ax.set_xlabel("Ground-truth syllables per file")
        ax.set_ylabel("Estimated syllables per file")
        m = res[k]
        ax.text(0.03, 0.97, f"MAE {m['MAE']:.1f}\nbias {m['bias']:+.1f}\nr {m['r']:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="#ccc"))

    ax = axes.flat[-1]
    maes = [res[k]["MAE"] for k in ESTIMATORS]
    err = np.array([[res[k]["MAE"] - ci[k][0], ci[k][1] - res[k]["MAE"]] for k in ESTIMATORS]).T
    colors = ["#adb5bd" if k == "duration" else "#2a6fdb" for k in ESTIMATORS]
    ax.bar(ESTIMATORS, maes, yerr=err, color=colors, capsize=4)
    ax.set_ylabel("MAE (syllables per file)")
    ax.set_title("Mean absolute error with 95% CI\n(grey = duration-only, no ZeroSyl)",
                 fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Syllable count estimation on {args.pred_dir} "
                 f"(calibrated on '{args.name}')", fontsize=11)
    fig.tight_layout()
    plot = Path("output/plots") / f"SyllableEstimator_{args.name}_{t}.png"
    plot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot, dpi=150)

    print(f"\nSaved: {res_dir / f'SyllableEstimator_{args.name}_{t}.txt'}\n"
          f"       {res_dir / f'SyllableEstimator_{args.name}_{t}_perfile.csv'}\n"
          f"       {plot}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="learn the correction on a set with ground truth")
    f.add_argument("gt_dir", type=Path)
    f.add_argument("pred_dir", type=Path)
    f.add_argument("--name", required=True, help="calibration name, e.g. Babaloon")
    f.add_argument("--tier", default="syllables")
    f.add_argument("--sil-id", type=int)
    f.add_argument("--keep-silence", action="store_true")

    pr = sub.add_parser("predict", help="estimate syllables, no ground truth needed")
    pr.add_argument("pred_dir", type=Path)
    pr.add_argument("--name", required=True)
    pr.add_argument("--estimator", choices=ESTIMATORS, default="direct")

    e = sub.add_parser("evaluate", help="score all estimators against ground truth")
    e.add_argument("gt_dir", type=Path)
    e.add_argument("pred_dir", type=Path)
    e.add_argument("--name", required=True)
    e.add_argument("--bootstrap", type=int, default=5000)

    args = p.parse_args()
    {"fit": cmd_fit, "predict": cmd_predict, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
