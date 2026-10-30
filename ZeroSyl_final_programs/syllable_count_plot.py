"""Scatter plot of ground-truth vs predicted syllable counts, one dot per file.

x axis : number of syllables in the "syllables" tier of each ground-truth TextGrid
         (non-empty intervals only, so silences/pauses are not counted)
y axis : number of syllables ZeroSyl predicted in the matching .pt file
         (rows of the (N, 3) [start, end, id] tensor, excluding silence segments)

Files are matched by name (id_3_main_post_test.TextGrid <-> id_3_main_post_test.pt),
searching all subfolders. Pearson's r is printed under the graph.

Usage (from ~/ZeroSyl-master):

    python syllable_count_plot.py data/babaloonAlignments/dev \
        output/segments/ZeroSylCollapsed/Babaloon/dev

    # works on the stupid baseline output too
    python syllable_count_plot.py data/babaloonAlignments/dev \
        output/segments/StupidBaseline/ZeroSylCollapsed/Babaloon/dev

The plot is saved to output/plots/<model>_<dataset>_<split>_syllable_counts.png
unless --out is given.

Silence: ZeroSylCollapsed gives every silence segment the same id, which is the
largest id in its vocabulary. By default the script takes the largest id seen in the
prediction folder as the silence id and does not count those segments. Use
--sil-id N to set it yourself, or --keep-silence to count every segment.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed on the server
import matplotlib.pyplot as plt
import numpy as np
import tgt
import torch
from scipy.stats import pearsonr


def count_ground_truth(path: Path, tier: str) -> int:
    textgrid = tgt.read_textgrid(path, include_empty_intervals=True)
    return sum(1 for i in textgrid.get_tier_by_name(tier).intervals if i.text.strip())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("ground_truth_dir", type=Path,
                        help="folder of ground-truth TextGrids, e.g. data/babaloonAlignments/dev")
    parser.add_argument("prediction_dir", type=Path,
                        help="folder of ZeroSyl .pt outputs, e.g. "
                             "output/segments/ZeroSylCollapsed/Babaloon/dev")
    parser.add_argument("--out", type=Path, help="where to save the .png")
    parser.add_argument("--tier", default="syllables", help="TextGrid tier to count")
    parser.add_argument("--sil-id", type=int, help="id ZeroSyl uses for silence")
    parser.add_argument("--keep-silence", action="store_true",
                        help="count silence segments as predicted syllables")
    parser.add_argument("--show", action="store_true", help="also open the plot window")
    args = parser.parse_args()

    gt_files = {p.stem: p for p in args.ground_truth_dir.rglob("*.TextGrid")}
    pred_files = {p.stem: p for p in args.prediction_dir.rglob("*.pt")}

    if not gt_files:
        raise SystemExit(f"No .TextGrid files found in {args.ground_truth_dir}")
    if not pred_files:
        raise SystemExit(f"No .pt files found in {args.prediction_dir}")

    only_gt = sorted(set(gt_files) - set(pred_files))
    only_pred = sorted(set(pred_files) - set(gt_files))
    if only_gt or only_pred:
        print(f"WARNING: file counts don't line up "
              f"({len(gt_files)} ground truth, {len(pred_files)} predicted)")
        for s in only_gt[:10]:
            print(f"  no prediction for   {s}")
        if len(only_gt) > 10:
            print(f"  ... and {len(only_gt) - 10} more ground-truth files without a prediction")
        for s in only_pred[:10]:
            print(f"  no ground truth for {s}")
        if len(only_pred) > 10:
            print(f"  ... and {len(only_pred) - 10} more predictions without ground truth")
        print("  Only files present in both folders are plotted.\n")

    names = sorted(set(gt_files) & set(pred_files))
    if len(names) < 2:
        raise SystemExit("Need at least 2 matching files to compute Pearson's r.")

    segments = {n: torch.load(pred_files[n], map_location="cpu") for n in names}

    sil_id = None
    if not args.keep_silence:
        if args.sil_id is not None:
            sil_id = args.sil_id
        else:
            ids = torch.cat([s[:, 2] for s in segments.values() if s.numel()])
            if ids.unique().numel() > 1:
                sil_id = int(ids.max())
        if sil_id is None:
            # e.g. the stupid baseline, where every segment has the same dummy id
            print("All segments share one id -> no silence removal, counting every segment")
        else:
            print(f"Silence id: {sil_id} (segments with this id are not counted)")

    x, y = [], []
    for n in names:
        x.append(count_ground_truth(gt_files[n], args.tier))
        seg = segments[n]
        if seg.numel() == 0:
            y.append(0)
        elif sil_id is None:
            y.append(seg.shape[0])
        else:
            y.append(int((seg[:, 2] != sil_id).sum()))
    x, y = np.array(x), np.array(y)

    r, p = pearsonr(x, y)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(x, y, s=28, alpha=0.75, color="#2a6fdb", edgecolors="white",
               linewidths=0.5, label="one file", zorder=3)

    lim = [0, max(x.max(), y.max()) * 1.05]
    ax.plot(lim, lim, ls="--", color="grey", lw=1, label="y = x (perfect count)")
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(lim, [slope * v + intercept for v in lim], color="#d9480f", lw=1.5,
            label=f"best fit: y = {slope:.2f}x {'+' if intercept >= 0 else '-'} "
                  f"{abs(intercept):.1f}")

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_xlabel("Ground-truth syllables per file")
    ax.set_ylabel("Predicted syllables per file")
    ax.set_title(f"{args.prediction_dir}\nvs {args.ground_truth_dir}", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)

    fig.text(0.5, 0.015,
             f"Pearson's r = {r:.3f}   (p = {p:.2e},  n = {len(names)} files)",
             ha="center", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))

    out = args.out
    if out is None:
        parts = args.prediction_dir.resolve().parts[-3:]
        out = Path("output/plots") / f"{'_'.join(parts)}_syllable_counts.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)

    print(f"Files plotted: {len(names)}")
    print(f"Ground truth : {x.sum()} syllables (mean {x.mean():.1f} per file)")
    print(f"Predicted    : {y.sum()} syllables (mean {y.mean():.1f} per file)")
    print(f"Pearson's r  : {r:.4f}  (p = {p:.2e})")
    print(f"Saved plot   : {out}")

    if args.show:
        matplotlib.use("TkAgg", force=True)
        plt.show()


if __name__ == "__main__":
    main()
