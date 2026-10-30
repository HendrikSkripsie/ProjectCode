"""Prominence sweep for ZeroSyl boundary detection.

ZeroSyl places a syllable boundary at every peak of the (smoothed, z-normalised)
WavLM layer-13 feature norm whose prominence is at least `prominence` (0.45 by default,
see zerosyl/zerosyl.py). This script re-runs ZeroSyl for a range of prominence values
(default 0.00, 0.05, ..., 1.00) and evaluates every run with zerosyl/eval/boundaries.py.

Speed-up: the expensive part (WavLM) does not depend on the prominence, so WavLM runs
ONCE per file and the remaining steps (peak picking, mean pooling, k-means lookup,
silence collapsing) are repeated for every prominence. The result is identical to
running `zerosyl encode` once per prominence, at roughly the cost of a single run.

Place this file in ~/ZeroSyl-master/ and run it from there (zerosyl conda env).

    # full sweep on Babaloon dev (encode + evaluate)
    python prominence_sweep.py data/babaloonWaveforms/dev data/babaloonAlignments/dev \
        --name Babaloon

    # only re-evaluate (outputs already encoded)
    python prominence_sweep.py data/babaloonWaveforms/dev data/babaloonAlignments/dev \
        --name Babaloon --skip-encode

Outputs
    output/segments/ProminenceSweep/<name>/prom_0.45/<file>.pt   collapsed segments
    output/segments/ProminenceSweepDiscrete/<name>/prom_0.45/...   (only with --discrete)
    output/results/ProminenceSweep_<name>.csv                    metrics per prominence
    output/results/ProminenceSweep_<name>.txt                    same, readable table
    output/plots/ProminenceSweep_<name>.png                      metrics vs prominence

Every prom_X.XX folder has the same layout as output/segments/ZeroSylCollapsed/...,
so you can also evaluate one by hand:
    zerosyl evaluate boundaries output/segments/ProminenceSweep/Babaloon/prom_0.45/ \
        data/babaloonAlignments/dev/ --segments-pattern "*.pt" --textgrid-pattern "*.TextGrid"
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

FRAME_RATE = 50.0


# --------------------------------------------------------------------------------------
# Encoding: identical to ZeroSylContinuous/Discrete/Collapsed.encode in zerosyl/zerosyl.py,
# split into "features" (WavLM, once) and "segment" (everything after, per prominence).
# --------------------------------------------------------------------------------------


@torch.inference_mode()
def extract_features(model, wav: torch.Tensor):
    """WavLM part of ZeroSylContinuous.encode (does not depend on the prominence)."""
    assert wav.ndim == 2
    assert wav.size(0) == 1
    if model.cfg.normalize:
        wav = torch.nn.functional.layer_norm(wav, wav.shape)
    wav = torch.nn.functional.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))

    boundary_features, semantic_features = model.extract_dual_features(
        wav, output_layer_1=model.boundary_layer, output_layer_2=model.meanpool_layer
    )
    boundary_features = boundary_features.squeeze(0).cpu().numpy()
    semantic_features = semantic_features.squeeze(0)

    # z-normalised, smoothed feature norm (also independent of the prominence)
    norms = np.linalg.norm(boundary_features, axis=-1)
    norms = (norms - norms.mean()) / norms.std()
    kernel = np.ones(model.window_size) / model.window_size
    pad_len = model.window_size // 2
    norms_padded = np.pad(norms, (pad_len, pad_len), mode="edge")
    norms_smooth = np.convolve(norms_padded, kernel, mode="valid")

    return norms_smooth, len(boundary_features), semantic_features, wav.device


@torch.inference_mode()
def segment(model, norms_smooth, n_frames, semantic_features, device, prominence):
    """Everything in ZeroSyl*.encode after WavLM, for one prominence value.

    Returns (collapsed, discrete) tensors of shape (N, 3) = [start, end, id],
    exactly what `zerosyl encode` saves for --output collapsed / discrete.
    """
    # boundary detection (ZeroSylContinuous)
    peaks, _ = find_peaks(norms_smooth, prominence=prominence)
    boundaries = [0] + peaks.tolist() + [n_frames]

    starts = torch.tensor(boundaries[:-1], device=device)
    ends = torch.tensor(boundaries[1:], device=device)
    embeddings = [
        semantic_features[start:end].mean(dim=0) for start, end in zip(starts, ends)
    ]
    embeddings = torch.stack(embeddings)

    # k-means assignment (ZeroSylDiscrete)
    import faiss

    embeddings = embeddings.cpu().numpy()
    faiss.normalize_L2(embeddings)
    _, ids = model.index.search(embeddings, 1)
    ids = torch.from_numpy(ids).squeeze(1).to(device)
    discrete = torch.stack([starts, ends, ids], dim=1).cpu()

    # silence collapsing (ZeroSylCollapsed)
    if model.mapping.device != ids.device:
        model.mapping = model.mapping.to(ids.device)
    ids = model.mapping[ids]

    if len(ids) == 1:
        collapsed = torch.stack([starts, ends, ids], dim=1).cpu()
        return collapsed, discrete

    not_repeated = torch.ones_like(ids, dtype=torch.bool)
    not_repeated[1:] = ~torch.logical_and(ids[1:] == ids[:-1], ids[1:] == model.SIL)
    is_end = torch.ones_like(ids, dtype=torch.bool)
    is_end[:-1] = ~torch.logical_and(ids[1:] == ids[:-1], ids[1:] == model.SIL)
    collapsed = torch.stack(
        [starts[not_repeated], ends[is_end], ids[not_repeated]], dim=1
    ).cpu()
    return collapsed, discrete


def prom_dir(root: Path, name: str, p: float) -> Path:
    return root / name / f"prom_{p:.2f}"


def run_encode(args, prominences, out_root, disc_root):
    from torchcodec.decoders import AudioDecoder

    from zerosyl.zerosyl import ZeroSylCollapsed

    input_dir = args.wav_dir
    input_paths = sorted(input_dir.rglob(args.input_pattern))
    assert input_paths, f"No files in {input_dir} matching {args.input_pattern}"

    print(f"Loading ZeroSylCollapsed on {args.device} ...")
    model = ZeroSylCollapsed.from_remote().to(args.device)

    print(f"{len(input_paths)} files x {len(prominences)} prominence values "
          f"({prominences[0]:.2f} .. {prominences[-1]:.2f})")
    t0 = time.time()
    for k, input_path in enumerate(input_paths, 1):
        rel = input_path.relative_to(input_dir).with_suffix(".pt")
        targets = [prom_dir(out_root, args.name, p) / rel for p in prominences]
        if args.discrete:
            targets += [prom_dir(disc_root, args.name, p) / rel for p in prominences]
        if not args.overwrite and all(t.exists() for t in targets):
            print(f"[{k}/{len(input_paths)}] {rel} already done, skipping")
            continue

        decoder = AudioDecoder(input_path, sample_rate=16000, num_channels=1)
        wav = decoder.get_all_samples().data.to(args.device)

        feats = extract_features(model, wav)

        # safety check: our split pipeline must equal the real ZeroSyl at its default
        if k == 1 and not args.no_verify:
            ref = torch.stack(model.encode(wav), dim=1).cpu()
            ours, _ = segment(model, *feats, prominence=model.prominence)
            if not torch.equal(ref, ours):
                raise SystemExit("ERROR: sweep output differs from ZeroSylCollapsed.encode "
                                 "at the default prominence. Please report this.")
            print(f"Check passed: output at prominence {model.prominence} is identical "
                  "to `zerosyl encode`.")

        for p in prominences:
            collapsed, discrete = segment(model, *feats, prominence=p)
            out = prom_dir(out_root, args.name, p) / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save(collapsed, out)
            if args.discrete:
                out = prom_dir(disc_root, args.name, p) / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                torch.save(discrete, out)

        el = time.time() - t0
        print(f"[{k}/{len(input_paths)}] {rel}  ({el / 60:.1f} min elapsed)")


# --------------------------------------------------------------------------------------
# Evaluation: zerosyl.eval.boundaries.evaluate_boundary_metrics, unchanged
# --------------------------------------------------------------------------------------

METRICS = ["precision", "recall", "f1", "os", "rvalue",
           "token_precision", "token_recall", "token_f1"]
LABELS = ["Precision", "Recall", "F1", "OS", "R-value",
          "Token Precision", "Token Recall", "Token F1"]


def segments_per_second(seg_dir: Path, pattern: str) -> float:
    n_seg, n_frames = 0, 0
    for p in seg_dir.glob(pattern):
        s = torch.load(p)
        n_seg += s.shape[0]
        n_frames += int(s[-1, 1])
    return n_seg / (n_frames / FRAME_RATE) if n_frames else float("nan")


def run_evaluate(args, prominences, out_root):
    from zerosyl.eval.boundaries import evaluate_boundary_metrics

    rows = []
    for p in prominences:
        seg_dir = prom_dir(out_root, args.name, p)
        if not seg_dir.exists():
            print(f"prominence {p:.2f}: {seg_dir} missing, skipping")
            continue
        print(f"Evaluating prominence {p:.2f} ...")
        values = evaluate_boundary_metrics(
            segments_dir=seg_dir,
            textgrid_dir=args.textgrid_dir,
            segments_pattern=args.segments_pattern,
            textgrid_pattern=args.textgrid_pattern,
            frame_rate=FRAME_RATE,
            constant_shift=args.constant_shift,
            tolerance=args.tolerance,
        )
        row = {"prominence": p, "segments_per_s": segments_per_second(
            seg_dir, args.segments_pattern)}
        row.update({m: float(v) * 100 for m, v in zip(METRICS, values)})
        rows.append(row)

    if not rows:
        raise SystemExit("Nothing to evaluate.")

    res_dir = Path("output/results")
    res_dir.mkdir(parents=True, exist_ok=True)
    csv_path = res_dir / f"ProminenceSweep_{args.name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prominence", "segments_per_s"] + METRICS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.2f}" if k == "prominence" else f"{v:.4f}")
                        for k, v in r.items()})

    # readable table, same numbers and rounding as `zerosyl evaluate boundaries`
    header = f"{'prom':>5} {'seg/s':>6} " + " ".join(f"{l:>9}" for l in
             ["Prec", "Recall", "F1", "OS", "R-value", "TokPrec", "TokRec", "TokF1"])
    lines = [f"Prominence sweep: {args.name}  (constant shift {args.constant_shift}, "
             f"tolerance {args.tolerance})", "", header, "-" * len(header)]
    for r in rows:
        mark = "  <- ZeroSyl default" if abs(r["prominence"] - 0.45) < 1e-9 else ""
        lines.append(f"{r['prominence']:5.2f} {r['segments_per_s']:6.2f} "
                     + " ".join(f"{r[m]:9.2f}" for m in METRICS) + mark)
    finite = [r for r in rows if np.isfinite(r["rvalue"])]
    best_f1 = max(rows, key=lambda r: r["f1"])
    lines += ["", f"Best F1      : {best_f1['f1']:.2f} at prominence {best_f1['prominence']:.2f}"]
    if finite:
        best_r = max(finite, key=lambda r: r["rvalue"])
        lines.append(f"Best R-value : {best_r['rvalue']:.2f} at prominence "
                     f"{best_r['prominence']:.2f}")
    table = "\n".join(lines)
    txt_path = res_dir / f"ProminenceSweep_{args.name}.txt"
    txt_path.write_text(table + "\n")
    print("\n" + table)

    # plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [r["prominence"] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    for m, label, color in [("precision", "Precision", "#2a6fdb"),
                            ("recall", "Recall", "#2f9e44"),
                            ("f1", "F1", "#d9480f"),
                            ("rvalue", "R-value", "#7048e8")]:
        y = [r[m] if np.isfinite(r[m]) else np.nan for r in rows]
        ax1.plot(x, y, marker="o", ms=4, lw=1.8, label=label, color=color)
    ax1.axvline(0.45, color="grey", ls="--", lw=1)
    ax1.text(0.455, 2, "ZeroSyl default (0.45)", color="grey", fontsize=8)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Score (%)")
    ax1.set_title(f"ZeroSyl prominence sweep - {args.name}")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower right")
    ax1.text(0.01, 0.98, "R-value below 0 is not shown", transform=ax1.transAxes,
             fontsize=8, color="grey", va="top")

    ax2.plot(x, [r["segments_per_s"] for r in rows], marker="o", ms=4, color="#495057")
    ax2.axvline(0.45, color="grey", ls="--", lw=1)
    ax2.set_ylabel("Segments / s")
    ax2.set_xlabel("Prominence threshold")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{v:.2f}" for v in x], rotation=90, fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()

    plot_path = Path("output/plots") / f"ProminenceSweep_{args.name}.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)

    print(f"\nSaved: {csv_path}\n       {txt_path}\n       {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wav_dir", type=Path, help="folder of input audio")
    parser.add_argument("textgrid_dir", type=Path, help="folder of ground-truth TextGrids")
    parser.add_argument("--name", required=True,
                        help="dataset name used for output folders, e.g. Babaloon")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--stop", type=float, default=1.0)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--input-pattern", default="*.wav")
    parser.add_argument("--segments-pattern", default="*.pt",
                        help='same as for `zerosyl evaluate boundaries` (use "**/*.pt" '
                             "for nested folders)")
    parser.add_argument("--textgrid-pattern", default="*.TextGrid")
    parser.add_argument("--constant-shift", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=None,
                        help="CPU threads for WavLM (default: torch decides)")
    parser.add_argument("--discrete", action="store_true",
                        help="also save discrete (non-collapsed) outputs, for "
                             "`zerosyl evaluate clustering`")
    parser.add_argument("--skip-encode", action="store_true", help="only evaluate")
    parser.add_argument("--skip-evaluate", action="store_true", help="only encode")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-encode files that are already done")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the identical-to-zerosyl check on the first file")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    n = int(round((args.stop - args.start) / args.step)) + 1
    prominences = [round(args.start + i * args.step, 4) for i in range(n)]

    out_root = Path("output/segments/ProminenceSweep")
    disc_root = Path("output/segments/ProminenceSweepDiscrete")

    if not args.skip_encode:
        run_encode(args, prominences, out_root, disc_root)
    if not args.skip_evaluate:
        run_evaluate(args, prominences, out_root)


if __name__ == "__main__":
    main()
