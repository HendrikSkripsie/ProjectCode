"""Dev-fit plots with ground truth on the y axis and the predicted count on the x axis.

DEV plot   x = predicted syllables per file (ZeroSyl or a baseline)
           y = ground-truth syllables per file
           orange line = least-squares fit on dev:  GT = a * predicted + b
           -> this line IS the estimator: read a file's predicted count on the x axis,
              go up to the line, read the estimated true count on the y axis.

TEST plot  x = predicted syllables per file on test
           y = ground-truth syllables per file on test ("hidden" in real use)
           orange line = the SAME dev line, not refitted
           faint line  = the best line this test set would ideally have (fitted on test)
           MSE = mean squared vertical distance between the dots and the dev line,
                 i.e. how well the dev estimator generalises to unseen files.

Counting is identical to syllable_count_plot.py.

Usage (from ~/ZeroSyl-master):

  # dev only
  python devfit_plots.py data/babaloonAlignments/dev/ output/segments/ZeroSylCollapsed/Babaloon/dev/ \
      --label ZeroSyl

  # dev + test
  python devfit_plots.py data/babaloonAlignments/dev/ output/segments/ZeroSylCollapsed/Babaloon/dev/ \
      --test data/babaloonAlignments/test/ output/segments/ZeroSylCollapsed/Babaloon/test/ \
      --label ZeroSyl

All images go to output/plots/DevFit/ (change with --outdir).
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tgt
import torch
from scipy.stats import pearsonr

BLUE, ORANGE, GREY = "#2a6fdb", "#d9480f", "grey"


def counts(gt_dir: Path, pred_dir: Path):
    """(names, ground truth, predicted) syllables per file, counted like syllable_count_plot."""
    gt_files = {p.stem: p for p in gt_dir.rglob("*.TextGrid")}
    pred_files = {p.stem: p for p in pred_dir.rglob("*.pt")}
    names = sorted(set(gt_files) & set(pred_files))
    if len(names) < 3:
        raise SystemExit(f"Fewer than 3 matching files in {gt_dir} and {pred_dir}")
    if len(names) != len(gt_files) or len(names) != len(pred_files):
        print(f"WARNING: {len(gt_files)} TextGrids, {len(pred_files)} predictions in "
              f"{pred_dir}, using the {len(names)} that match")

    segs = {n: torch.load(pred_files[n], map_location="cpu") for n in names}
    ids = torch.cat([s[:, 2] for s in segs.values() if s.numel()])
    sil_id = int(ids.max()) if ids.unique().numel() > 1 else None  # ZeroSyl silence id

    gt, pred = [], []
    for n in names:
        tg = tgt.read_textgrid(gt_files[n], include_empty_intervals=True)
        gt.append(sum(1 for i in tg.get_tier_by_name("syllables").intervals if i.text.strip()))
        s = segs[n]
        if s.numel() == 0:
            pred.append(0)
        elif sil_id is None:
            pred.append(s.shape[0])
        else:
            pred.append(int((s[:, 2] != sil_id).sum()))
    return names, np.array(gt, float), np.array(pred, float)


def line_label(a, b, prefix):
    return f"{prefix}: y = {a:.3f}x {'+' if b >= 0 else '-'} {abs(b):.1f}"


def new_axes(x, y):
    fig, ax = plt.subplots(figsize=(7, 7.4))
    lim = [0, max(x.max(), y.max()) * 1.05]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.plot(lim, lim, ls="--", color=GREY, lw=1, alpha=0.6,
            label="y = x (predicted count = true count)")
    return fig, ax, lim


def finish(fig, ax, out, title, xlabel, ylabel, footer):
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.text(0.5, 0.015, footer, ha="center", fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("gt_dev", type=Path)
    p.add_argument("pred_dev", type=Path)
    p.add_argument("--test", nargs=2, type=Path, metavar=("GT_TEST", "PRED_TEST"))
    p.add_argument("--label", required=True,
                   help="system name used in titles and file names, e.g. ZeroSyl or 300ms")
    p.add_argument("--dataset", default="Babaloon")
    p.add_argument("--outdir", type=Path, default=Path("output/plots/DevFit"))
    args = p.parse_args()

    xlabel = f"{args.label} predicted syllables per file"

    # ------------------------------------------------ DEV: learn GT = a*pred + b
    _, gt_d, pr_d = counts(args.gt_dev, args.pred_dev)
    a, b = np.polyfit(pr_d, gt_d, 1)
    r_d, p_d = pearsonr(pr_d, gt_d)
    mse_d = np.mean((a * pr_d + b - gt_d) ** 2)
    r2_d = 1 - mse_d / np.var(gt_d)

    fig, ax, lim = new_axes(pr_d, gt_d)
    ax.scatter(pr_d, gt_d, s=28, alpha=0.75, color=BLUE, edgecolors="white",
               linewidths=0.5, label="one dev file", zorder=3)
    ax.plot(lim, [a * v + b for v in lim], color=ORANGE, lw=1.8,
            label=line_label(a, b, "dev fit (the estimator)"))
    ax.text(0.97, 0.03, f"R² on dev = {r2_d:.3f}\nMSE on dev = {mse_d:.1f}\n"
                        f"(RMSE {np.sqrt(mse_d):.1f} syllables)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    out_dev = args.outdir / f"{args.dataset}_dev_{args.label}_devfit.png"
    finish(fig, ax, out_dev, f"DEV: {args.pred_dev}\nvs {args.gt_dev}", xlabel,
           "Ground-truth syllables per file",
           f"Pearson's r = {r_d:.3f}   (p = {p_d:.2e},  n = {len(gt_d)} files)")

    print(f"[{args.label}] dev fit : GT = {a:.3f} * predicted {b:+.1f}   "
          f"({len(gt_d)} files, r = {r_d:.3f}, R² = {r2_d:.3f}, MSE = {mse_d:.1f})")
    print(f"           saved   : {out_dev}")

    if not args.test:
        return

    # ------------------------------------------------ TEST: apply the same line
    names, gt_t, pr_t = counts(*args.test)
    est = a * pr_t + b
    mse = np.mean((est - gt_t) ** 2)
    mae = np.mean(np.abs(est - gt_t))
    r_t, p_t = pearsonr(pr_t, gt_t)
    ia, ib = np.polyfit(pr_t, gt_t, 1)                 # ideal line for this test set
    mse_ideal = np.mean((ia * pr_t + ib - gt_t) ** 2)
    # out-of-sample R²: share of the test files' variation explained by the DEV line.
    # 1 = perfect, 0 = no better than guessing the test mean for every file, < 0 = worse.
    r2 = 1 - mse / np.var(gt_t)
    r2_ideal = 1 - mse_ideal / np.var(gt_t)

    fig, ax, lim = new_axes(pr_t, gt_t)
    ax.scatter(pr_t, gt_t, s=28, alpha=0.75, color=BLUE, edgecolors="white",
               linewidths=0.5, label="one test file", zorder=3)
    ax.plot(lim, [ia * v + ib for v in lim], color="#868e96", lw=1.2, alpha=0.5,
            label=line_label(ia, ib, "ideal fit for test (faint)"))
    ax.plot(lim, [a * v + b for v in lim], color=ORANGE, lw=1.8,
            label=line_label(a, b, "dev fit applied to test"))
    ax.text(0.97, 0.03,
            f"dev line on test : R² = {r2:.3f}  MSE = {mse:.1f}\n"
            f"  (RMSE {np.sqrt(mse):.1f}, MAE {mae:.1f} syllables)\n"
            f"ideal test line  : R² = {r2_ideal:.3f}  MSE = {mse_ideal:.1f}\n"
            f"dev line on dev  : R² = {r2_d:.3f}  MSE = {mse_d:.1f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            family="monospace", bbox=dict(boxstyle="round", fc="white", ec="#ccc"))
    out_test = args.outdir / f"{args.dataset}_test_{args.label}_devfit.png"
    finish(fig, ax, out_test, f"TEST: {args.test[1]}\nvs {args.test[0]} (hidden ground truth)",
           xlabel, "Ground-truth syllables per file (test)",
           f"R² = {r2:.3f}   MSE = {mse:.1f}   Pearson's r = {r_t:.3f}   "
           f"(n = {len(gt_t)} files)")

    with open(out_test.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "predicted_count", "ground_truth", "estimate_dev_line",
                    "error", "squared_error"])
        for n, x, g, e in zip(names, pr_t, gt_t, est):
            w.writerow([n, int(x), int(g), f"{e:.1f}", f"{e - g:+.1f}", f"{(e - g) ** 2:.1f}"])

    print(f"[{args.label}] test    : R² = {r2:.3f}   MSE = {mse:.1f} (RMSE {np.sqrt(mse):.1f}, "
          f"MAE {mae:.1f})   r = {r_t:.3f}")
    print(f"           ideal   : R² = {r2_ideal:.3f}   MSE = {mse_ideal:.1f}")
    print(f"           ideal line: GT = {ia:.3f} * predicted {ib:+.1f}")
    print(f"           saved   : {out_test}\n                     {out_test.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
