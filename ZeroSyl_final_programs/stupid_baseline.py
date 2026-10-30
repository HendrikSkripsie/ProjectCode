"""Stupid baseline for ZeroSyl syllable boundaries.

Reads ZeroSyl segment outputs (.pt files, each a tensor of shape (N, 3) holding
[start_frame, end_frame, id] per segment at 50 Hz), measures the average rate at
which ZeroSyl places segments across the whole dataset, and writes a dummy version
of every file where boundaries are placed at a fixed interval instead.

The output mirrors the input folder structure and file names, so it can be fed
straight into zerosyl/eval/boundaries.py in place of the real ZeroSyl output.

Examples (run from ~/ZeroSyl-master):

    # one dataset
    python stupid_baseline.py output/segments/ZeroSylCollapsed/Babaloon

    # every dataset under a model folder
    python stupid_baseline.py output/segments/ZeroSylCollapsed --all

    # also write TextGrids so you can look at the dummy boundaries in Praat
    python stupid_baseline.py output/segments/ZeroSylCollapsed/fleurs --textgrid

Output goes to output/segments/StupidBaseline/<model>/<dataset>/..., e.g.
    output/segments/StupidBaseline/ZeroSylCollapsed/Babaloon/dev/id_3_main_pre_test.pt
"""

import argparse
import json
from pathlib import Path

import torch

FRAME_RATE = 50.0  # ZeroSyl feature frames per second (20 ms per frame)


def load_segments(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")


def measure_rate(paths):
    """Total number of segments and total number of frames over all files."""
    n_segments, n_frames = 0, 0
    for p in paths:
        seg = load_segments(p)
        if seg.numel() == 0:
            continue
        n_segments += seg.shape[0]
        n_frames += int(seg[-1, 1])  # end frame of last segment = utterance length
    return n_segments, n_frames


def uniform_segments(total_frames: int, period: float, dtype) -> torch.Tensor:
    """Segments of constant length `period` frames covering [0, total_frames]."""
    bounds = [0]
    k = 1
    while True:
        b = int(round(k * period))
        if b >= total_frames:
            break
        if b > bounds[-1]:
            bounds.append(b)
        k += 1
    bounds.append(total_frames)

    starts = torch.tensor(bounds[:-1], dtype=dtype)
    ends = torch.tensor(bounds[1:], dtype=dtype)
    ids = torch.zeros_like(starts)  # dummy id, ignored by boundaries.py
    return torch.stack([starts, ends, ids], dim=1)


def write_textgrid(segments: torch.Tensor, path: Path):
    import tgt

    tier = tgt.IntervalTier(name="syllables")
    for start, end, _ in segments.tolist():
        if end > start:
            tier.add_interval(
                tgt.Interval(start / FRAME_RATE, end / FRAME_RATE, "syl")
            )
    tg = tgt.TextGrid()
    tg.add_tier(tier)
    path.parent.mkdir(parents=True, exist_ok=True)
    tgt.write_to_file(tg, str(path), format="long")


def run_dataset(input_dir: Path, output_root: Path, pattern: str, scale: float,
                textgrid: bool):
    paths = sorted(input_dir.glob(pattern))
    if not paths:
        print(f"[skip] no files matching '{pattern}' in {input_dir}")
        return

    n_segments, n_frames = measure_rate(paths)
    if n_segments == 0 or n_frames == 0:
        print(f"[skip] {input_dir} has no usable segments")
        return

    # average frames per ZeroSyl segment -> constant baseline period
    period = (n_frames / n_segments) * scale
    period_ms = period / FRAME_RATE * 1000
    rate = FRAME_RATE / period

    # <output_root>/<model>/<dataset>
    output_dir = output_root / input_dir.parent.name / input_dir.name
    dtype = load_segments(paths[0]).dtype

    n_out = 0
    for p in paths:
        seg = load_segments(p)
        total = int(seg[-1, 1]) if seg.numel() else 0
        dummy = uniform_segments(total, period, dtype)

        rel = p.relative_to(input_dir)
        out_path = output_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dummy, out_path)
        n_out += dummy.shape[0]

        if textgrid:
            write_textgrid(dummy, output_dir.parent / f"{input_dir.name}_textgrids"
                           / rel.with_suffix(".TextGrid"))

    info = {
        "source": str(input_dir),
        "files": len(paths),
        "zerosyl_segments": n_segments,
        "total_seconds": n_frames / FRAME_RATE,
        "avg_zerosyl_segments_per_file": n_segments / len(paths),
        "zerosyl_segments_per_second": n_segments / (n_frames / FRAME_RATE),
        "scale": scale,
        "baseline_period_frames": period,
        "baseline_period_ms": period_ms,
        "baseline_segments_per_second": rate,
        "baseline_segments": n_out,
    }
    # stored next to (not inside) the dataset folder so it can't confuse globbing
    with open(output_dir.parent / f"{input_dir.name}_baseline_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(
        f"[done] {input_dir} -> {output_dir}\n"
        f"       {len(paths)} files, ZeroSyl: {n_segments} segments "
        f"({info['zerosyl_segments_per_second']:.2f}/s)\n"
        f"       baseline: one boundary every {period_ms:.1f} ms "
        f"({rate:.2f}/s), {n_out} segments"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path,
                        help="ZeroSyl output folder for one dataset, or a model "
                             "folder when used with --all")
    parser.add_argument("--all", action="store_true",
                        help="treat input_dir as a model folder and process every "
                             "dataset subfolder in it")
    parser.add_argument("--output-root", type=Path,
                        default=Path("output/segments/StupidBaseline"))
    parser.add_argument("--pattern", default="**/*.pt",
                        help="glob for segment files inside a dataset folder")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiply the period: >1 = fewer boundaries, "
                             "<1 = more. 1.0 matches ZeroSyl's average rate")
    parser.add_argument("--textgrid", action="store_true",
                        help="also write TextGrids of the dummy boundaries")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if args.all:
        datasets = sorted(d for d in input_dir.iterdir() if d.is_dir())
    else:
        datasets = [input_dir]

    for d in datasets:
        run_dataset(d, args.output_root, args.pattern, args.scale, args.textgrid)


if __name__ == "__main__":
    main()
