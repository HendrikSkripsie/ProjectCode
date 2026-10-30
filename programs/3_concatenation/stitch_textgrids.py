#!/usr/bin/env python3
"""
stitch_textgrids.py - build one TextGrid per concatenated speaker file.

Takes the per-recording TextGrids made by zerosyl_align.py (tiers words / phones / syllables)
and places each one at its position in the joined wav, using the <speaker>_segments.csv files
written by concat_speakers.py.

Output per speaker:  OUT/<speaker>.TextGrid  with tiers  words | phones | syllables | utterance
(the extra "utterance" tier shows which original recording each stretch came from)

Example
  python stitch_textgrids.py --concat ~/Conrad_Data/concatenated_20 \
                             --textgrids ~/Conrad_Data/textgrids_aligned \
                             --out ~/Conrad_Data/concatenated_20
"""

import argparse
import csv
import re
import sys
from pathlib import Path


def read_textgrid(path: Path):
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    tiers, cur, iv, gxmax = [], None, {}, None
    for line in text.splitlines():
        if (m := re.match(r'^\s*name\s*=\s*"(.*)"', line)):
            cur = (m.group(1), [])
            tiers.append(cur)
        elif (m := re.match(r"^\s*(xmin|xmax)\s*=\s*([-0-9.eE+]+)", line)):
            if cur is None:
                if m.group(1) == "xmax":
                    gxmax = float(m.group(2))
            else:
                iv[m.group(1)] = float(m.group(2))
        elif cur is not None and (m := re.match(r'^\s*text\s*=\s*"(.*)"\s*$', line)):
            cur[1].append((iv["xmin"], iv["xmax"], m.group(1).replace('""', '"')))
    return gxmax, tiers


def _fmt(x):
    s = f"{x:.6f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def write_textgrid(path: Path, dur: float, tiers):
    L = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0 ",
         f"xmax = {_fmt(dur)} ", "tiers? <exists> ", f"size = {len(tiers)} ", "item []: "]
    for i, (name, ivs) in enumerate(tiers, 1):
        filled, t = [], 0.0
        for s, e, lab in sorted(ivs):
            s, e = max(s, t), min(e, dur)
            if e - s < 1e-6:
                continue
            if s > t + 1e-6:
                filled.append((t, s, ""))
            filled.append((s, e, lab))
            t = e
        if dur > t + 1e-6:
            filled.append((t, dur, ""))
        filled = filled or [(0.0, dur, "")]
        L += [f"    item [{i}]:", '        class = "IntervalTier" ', f'        name = "{name}" ',
              "        xmin = 0 ", f"        xmax = {_fmt(dur)} ", f"        intervals: size = {len(filled)} "]
        for j, (s, e, lab) in enumerate(filled, 1):
            L += [f"        intervals [{j}]:", f"            xmin = {_fmt(s)} ",
                  f"            xmax = {_fmt(e)} ", f'            text = "{lab.replace(chr(34), chr(34) * 2)}" ']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--concat", required=True, help="folder from concat_speakers.py (has <speaker>_segments.csv)")
    p.add_argument("--textgrids", required=True, help="output folder of zerosyl_align.py batch")
    p.add_argument("--out", help="where to write the stitched TextGrids (default: the --concat folder)")
    args = p.parse_args()

    concat = Path(args.concat).expanduser()
    tg_root = Path(args.textgrids).expanduser()
    out = Path(args.out).expanduser() if args.out else concat
    seg_files = sorted(concat.glob("*_segments.csv"))
    if not seg_files:
        sys.exit(f"no *_segments.csv files in {concat} - run concat_speakers.py first")
    if not tg_root.is_dir():
        sys.exit(f"TextGrid folder not found: {tg_root}")
    index = {f.stem: f for f in tg_root.rglob("*.TextGrid")}

    report, tier_names = [], None
    for seg_csv in seg_files:
        spk = seg_csv.name[: -len("_segments.csv")]
        with open(seg_csv, newline="", encoding="utf-8") as f:
            segs = list(csv.DictReader(f))
        merged, utt_tier, missing = {}, [], []
        dur = float(segs[-1]["end_s"])
        for s in segs:
            utt = Path(s["file"]).stem
            off, end = float(s["start_s"]), float(s["end_s"])
            utt_tier.append((off, end, utt))
            tg = index.get(utt)
            if tg is None:
                missing.append(utt)
                continue
            _, tiers = read_textgrid(tg)
            if tier_names is None:
                tier_names = [n for n, _ in tiers]
            for name, ivs in tiers:
                merged.setdefault(name, [])
                for a, b, lab in ivs:
                    if lab.strip():
                        merged[name].append((min(off + a, end), min(off + b, end), lab))
        names = tier_names or ["words", "phones", "syllables"]
        tiers_out = [(n, merged.get(n, [])) for n in names] + [("utterance", utt_tier)]
        write_textgrid(out / f"{spk}.TextGrid", dur, tiers_out)
        report.append({"speaker": spk, "recordings": len(segs), "aligned": len(segs) - len(missing),
                       "missing": " ".join(missing)})
        msg = f"{spk}: {len(segs) - len(missing)}/{len(segs)} recordings aligned -> {spk}.TextGrid"
        print(msg + (f"   MISSING: {', '.join(missing)}" if missing else ""))

    with open(out / "stitch_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["speaker", "recordings", "aligned", "missing"])
        w.writeheader()
        w.writerows(report)
    n_missing = sum(len(r["missing"].split()) for r in report)
    print(f"\ndone: {len(report)} TextGrids in {out}"
          + (f"  ({n_missing} recordings had no alignment - their stretch is left empty, see stitch_report.csv)"
             if n_missing else ""))


if __name__ == "__main__":
    main()
