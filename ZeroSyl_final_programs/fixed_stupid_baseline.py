"""Fixed-rate stupid baseline for syllable boundaries.

Like stupid_baseline.py, but instead of matching ZeroSyl's average rate it places a
boundary every N milliseconds, for a fixed set of N (default 50 100 200 300 400 500).

ZeroSyl's output is only used to know how long each utterance is (the end frame of
its last segment), so every dummy file covers exactly the same time span as the real
prediction and lines up with the ground-truth TextGrid.

Each rate gets its own folder, mirroring the input structure and file names, so
zerosyl/eval/boundaries.py (and syllable_count_plot.py) can read it directly:

    output/segments/FixedStupidBaseline/<N>ms/<dataset>/<split>/<file>.pt
    e.g. output/segments/FixedStupidBaseline/100ms/Babaloon/dev/id_3_main_post_test.pt

Usage (from ~/ZeroSyl-master):

    python fixed_stupid_baseline.py output/segments/ZeroSylCollapsed/Babaloon
    python fixed_stupid_baseline.py output/segments/ZeroSylCollapsed/Babaloon --rates 100 250
"""

import argparse
from pathlib import Path

import torch

FRAME_RATE = 50.0  # ZeroSyl feature frames per second (20 ms per frame)


def fixed_segments(total_frames: int, period_frames: float, dtype) -> torch.Tensor:
    """Segments of (on average) `period_frames` frames covering [0, total_frames].

    50 ms is 2.5 frames, so boundaries are placed at round(k * period) and the
    segment lengths alternate 2/3 frames, averaging exactly 50 ms.
    """
    bounds = [0]
    k = 1
    while True:
        b = int(round(k * period_frames))
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_dir", type=Path,
                        help="ZeroSyl output folder for one dataset, "
                             "e.g. output/segments/ZeroSylCollapsed/Babaloon")
    parser.add_argument("--rates", type=int, nargs="+",
                        default=[50, 100, 200, 300, 400, 500],
                        help="boundary spacing(s) in milliseconds")
    parser.add_argument("--output-root", type=Path,
                        default=Path("output/segments/FixedStupidBaseline"))
    parser.add_argument("--pattern", default="**/*.pt")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    paths = sorted(input_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files matching '{args.pattern}' in {input_dir}")

    # load once: we only need each utterance's length in frames
    lengths, dtype = {}, None
    for p in paths:
        seg = torch.load(p, map_location="cpu")
        dtype = dtype or seg.dtype
        lengths[p] = int(seg[-1, 1]) if seg.numel() else 0
    total_s = sum(lengths.values()) / FRAME_RATE
    print(f"{len(paths)} files from {input_dir} ({total_s:.1f} s of audio)")

    for ms in args.rates:
        period = ms / 1000 * FRAME_RATE  # in frames
        out_dir = args.output_root / f"{ms}ms" / input_dir.name
        n_seg = 0
        for p in paths:
            dummy = fixed_segments(lengths[p], period, dtype)
            out_path = out_dir / p.relative_to(input_dir)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dummy, out_path)
            n_seg += dummy.shape[0]
        print(f"  {ms:>4} ms -> {out_dir}  ({n_seg} segments, "
              f"{n_seg / total_s:.2f} per second)")


if __name__ == "__main__":
    main()
