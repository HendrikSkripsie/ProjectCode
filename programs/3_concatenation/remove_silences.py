#!/usr/bin/env python3
"""
remove_silences.py - cut the silences out of wav files, using their TextGrids, and shift the
TextGrids to match.

A silence is an empty (or "sil"/"sp"/"<eps>") interval on the reference tier (default: words).
Every silence longer than --keep-ms is shortened to --keep-ms (half kept at each side, with a
fade over that kept part so the joins don't click). Speech audio is never changed, and every
word/phone/syllable keeps its exact duration; only the gaps between them shrink.

Files are matched by name: WAV_DIR/G0001.wav <-> TG_DIR/G0001.TextGrid

Output (in --out):
  Waveform/<name>.wav           audio without the silences
  TextGrid/<name>.TextGrid      the same tiers, shifted to the new timeline
  time_maps/<name>_removed.csv  which stretches of the ORIGINAL file were removed
  summary.csv                   per file: original / new duration, seconds removed

Example
  python3 remove_silences.py \
      --wav-dir  "~/Conrad_Data/Neat Output/ConradConcatWaveform" \
      --tg-dir   "~/Conrad_Data/Neat Output/ConradConcatTextGrid" \
      --out      "~/Conrad_Data/Neat Output/ConradConcatNoSilence"
"""

import argparse
import csv
import re
import sys
import wave
from pathlib import Path

import numpy as np

SILENCE = {"", "sil", "sp", "<eps>", "<sil>", "silence", "spn_sil"}

# --------------------------------------------------------------------------- #
# TextGrid reading (long and short format) and writing (long format)
# --------------------------------------------------------------------------- #

_TOKEN = re.compile(r'"(?:[^"]|"")*"|\[[^\]\n]*\]|<[A-Za-z]+>|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def read_textgrid(path: Path):
    """Return (xmin, xmax, [(class, name, items)]); items = (start, end, text) or (time, mark)."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("latin-1")
    toks = [t for t in _TOKEN.findall(text) if not t.startswith(("[", "<"))]
    strings = lambda t: t[1:-1].replace('""', '"')
    i = 0

    def nxt():
        nonlocal i
        i += 1
        return toks[i - 1]

    nxt(); nxt()  # "ooTextFile" "TextGrid"
    xmin, xmax = float(nxt()), float(nxt())
    n_tiers = int(float(nxt()))
    tiers = []
    for _ in range(n_tiers):
        cls, name = strings(nxt()), strings(nxt())
        float(nxt()); float(nxt())
        n = int(float(nxt()))
        items = []
        for _ in range(n):
            if cls == "IntervalTier":
                items.append((float(nxt()), float(nxt()), strings(nxt())))
            else:
                items.append((float(nxt()), strings(nxt())))
        tiers.append((cls, name, items))
    return xmin, xmax, tiers


def _fmt(x):
    s = f"{x:.6f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def _esc(s):
    return s.replace('"', '""')


def write_textgrid(path: Path, dur: float, tiers):
    L = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0 ",
         f"xmax = {_fmt(dur)} ", "tiers? <exists> ", f"size = {len(tiers)} ", "item []: "]
    for k, (cls, name, items) in enumerate(tiers, 1):
        L += [f"    item [{k}]:", f'        class = "{cls}" ', f'        name = "{_esc(name)}" ',
              "        xmin = 0 ", f"        xmax = {_fmt(dur)} "]
        if cls == "IntervalTier":
            L.append(f"        intervals: size = {len(items)} ")
            for j, (s, e, t) in enumerate(items, 1):
                L += [f"        intervals [{j}]:", f"            xmin = {_fmt(s)} ",
                      f"            xmax = {_fmt(e)} ", f'            text = "{_esc(t)}" ']
        else:
            L.append(f"        points: size = {len(items)} ")
            for j, (t, m) in enumerate(items, 1):
                L += [f"        points [{j}]:", f"            number = {_fmt(t)} ", f'            mark = "{_esc(m)}" ']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #

def silence_regions(tier_items, dur, keep, min_sil):
    """Stretches to REMOVE, in seconds. Silence = any time not covered by a non-silent
    interval of the reference tier (empty intervals and unannotated gaps alike)."""
    speech = sorted((max(0.0, s), min(dur, e)) for s, e, lab in tier_items
                    if lab.strip().lower() not in SILENCE and e > s)
    silences, t = [], 0.0
    for s, e in speech:
        if s > t:
            silences.append((t, s))
        t = max(t, e)
    if dur > t:
        silences.append((t, dur))
    regions = []
    for s, e in silences:
        length = e - s
        if length <= max(keep, min_sil):
            continue
        a = s if s <= 1e-6 else s + keep / 2           # leading silence: cut from 0
        b = e if e >= dur - 1e-6 else e - keep / 2     # trailing silence: cut to the end
        if b > a:
            regions.append([a, b])
    regions.sort()
    merged = []
    for a, b in regions:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def make_mapper(removed):
    """Original time -> new time (times inside a removed stretch map to the cut point)."""
    starts = np.array([a for a, _ in removed]) if removed else np.zeros(0)
    ends = np.array([b for _, b in removed]) if removed else np.zeros(0)

    def f(t):
        if not len(starts):
            return t
        return t - float(np.clip(t - starts, 0, ends - starts).sum())
    return f


def process(wav_path: Path, tg_path: Path, out: Path, args):
    with wave.open(str(wav_path), "rb") as w:
        params = w.getparams()
        sr, ch, sw = params.framerate, params.nchannels, params.sampwidth
        raw = w.readframes(params.nframes)
    if sw != 2:
        raise ValueError(f"only 16-bit PCM wav supported (this file is {8 * sw}-bit)")
    audio = np.frombuffer(raw, dtype="<i2").reshape(-1, ch).astype(np.float32)
    n = audio.shape[0]
    dur = n / sr

    _, tg_xmax, tiers = read_textgrid(tg_path)
    note = ""
    if abs(tg_xmax - dur) > 0.05:
        note = f"TextGrid length {tg_xmax:.3f}s differs from wav {dur:.3f}s"
    ref = next((t for t in tiers if t[1].strip().lower() == args.tier.lower() and t[0] == "IntervalTier"), None)
    if ref is None:
        raise ValueError(f"no interval tier named '{args.tier}' (tiers: {[t[1] for t in tiers]})")

    keep = args.keep_ms / 1000
    regions = silence_regions(ref[2], dur, keep, args.min_silence_ms / 1000)
    # snap to samples so audio and TextGrid agree exactly
    regions = [[round(a * sr) / sr, round(b * sr) / sr] for a, b in regions]
    regions = [r for r in regions if r[1] > r[0]]

    # ---- audio: fades in the kept silence next to each cut, then drop the removed stretches
    half = int(round(keep / 2 * sr))
    if half > 0:
        ramp = np.linspace(1.0, 0.0, half, dtype=np.float32)[:, None]
        for a, b in regions:
            ia, ib = int(round(a * sr)), int(round(b * sr))
            if ia > 0:                                   # fade out the silence before the cut
                lo = max(0, ia - half)
                audio[lo:ia] *= ramp[half - (ia - lo):]
            if ib < n:                                   # fade in the silence after the cut
                hi = min(n, ib + half)
                audio[ib:hi] *= ramp[::-1][: hi - ib]
    keep_mask = np.ones(n, bool)
    for a, b in regions:
        keep_mask[int(round(a * sr)):int(round(b * sr))] = False
    new_audio = audio[keep_mask]
    new_dur = new_audio.shape[0] / sr

    wav_out = out / "Waveform" / wav_path.name
    wav_out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(wav_out), "wb") as w:
        w.setnchannels(ch)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.clip(np.round(new_audio), -32768, 32767).astype("<i2").tobytes())

    # ---- TextGrid: shift every boundary with the same mapping
    f = make_mapper(regions)
    new_tiers = []
    for cls, name, items in tiers:
        if cls == "IntervalTier":
            ivs = []
            for s, e, lab in items:
                ns, ne = min(f(min(s, dur)), new_dur), min(f(min(e, dur)), new_dur)
                if ne - ns > 1e-7:
                    ivs.append([ns, ne, lab])
            # merge neighbouring silences, close tiny gaps, cover 0..new_dur exactly
            filled, t = [], 0.0
            for iv in sorted(ivs):
                if iv[0] > t + 1e-6:                    # unannotated gap -> explicit empty interval
                    filled.append([t, iv[0], ""])
                else:
                    iv[0] = max(iv[0], t)
                if iv[1] - iv[0] > 1e-7:
                    filled.append(iv)
                    t = iv[1]
            if new_dur > t + 1e-6:
                filled.append([t, new_dur, ""])
            clean = []
            for iv in filled:
                if clean and iv[2].strip().lower() in SILENCE and clean[-1][2].strip().lower() in SILENCE:
                    clean[-1][1] = iv[1]
                else:
                    clean.append(iv)
            if clean:
                clean[-1][1] = new_dur
            else:
                clean = [[0.0, new_dur, ""]]
            new_tiers.append((cls, name, [tuple(c) for c in clean]))
        else:
            new_tiers.append((cls, name, [(min(f(t), new_dur), m) for t, m in items]))
    write_textgrid(out / "TextGrid" / f"{tg_path.stem}.TextGrid", new_dur, new_tiers)

    tm = out / "time_maps" / f"{wav_path.stem}_removed.csv"
    tm.parent.mkdir(parents=True, exist_ok=True)
    with open(tm, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["orig_start", "orig_end", "new_time"])
        for a, b in regions:
            wr.writerow([round(a, 6), round(b, 6), round(f(a), 6)])

    removed = dur - new_dur
    return {"file": wav_path.stem, "original_s": round(dur, 3), "new_s": round(new_dur, 3),
            "removed_s": round(removed, 3), "removed_pct": round(100 * removed / dur, 1) if dur else 0,
            "pauses_shortened": len(regions), "note": note}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wav-dir", required=True, help="folder with the wav files")
    p.add_argument("--tg-dir", required=True, help="folder with the matching TextGrids")
    p.add_argument("--out", required=True, help="new output folder")
    p.add_argument("--tier", default="words", help="tier whose empty intervals are silence (default: words)")
    p.add_argument("--keep-ms", type=float, default=20.0,
                   help="each pause is shortened to this many ms (default 20; 0 = cut completely, no fades)")
    p.add_argument("--min-silence-ms", type=float, default=0.0,
                   help="leave pauses shorter than this untouched (default 0 = shorten every pause)")
    p.add_argument("--overwrite", action="store_true", help="allow writing into an existing output folder")
    args = p.parse_args()

    wav_dir, tg_dir = Path(args.wav_dir).expanduser(), Path(args.tg_dir).expanduser()
    out = Path(args.out).expanduser()
    for d in (wav_dir, tg_dir):
        if not d.is_dir():
            sys.exit(f"folder not found: {d}")
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        sys.exit(f"{out} already exists and is not empty (use --overwrite)")
    if out.resolve() in (wav_dir.resolve(), tg_dir.resolve()):
        sys.exit("--out must be a NEW folder, not one of the input folders")

    tgs = {t.stem: t for t in tg_dir.rglob("*") if t.suffix.lower() == ".textgrid"}
    wavs = sorted(w for w in wav_dir.rglob("*") if w.suffix.lower() == ".wav")
    pairs = [(w, tgs[w.stem]) for w in wavs if w.stem in tgs]
    no_tg = [w.name for w in wavs if w.stem not in tgs]
    no_wav = sorted(set(tgs) - {w.stem for w in wavs})
    print(f"{len(pairs)} wav/TextGrid pairs found")
    if no_tg:
        print(f"WARNING: no TextGrid for {len(no_tg)} wav(s): {', '.join(no_tg[:10])}")
    if no_wav:
        print(f"WARNING: no wav for {len(no_wav)} TextGrid(s): {', '.join(no_wav[:10])}")
    if not pairs:
        sys.exit("nothing to do - wav and TextGrid names must match (G0001.wav <-> G0001.TextGrid)")

    rows = []
    for w, t in pairs:
        try:
            r = process(w, t, out, args)
            rows.append(r)
            print(f"{r['file']}: {r['original_s']:.1f} s -> {r['new_s']:.1f} s "
                  f"({r['removed_pct']}% removed, {r['pauses_shortened']} pauses)"
                  + (f"  NOTE: {r['note']}" if r["note"] else ""))
        except Exception as exc:
            print(f"ERROR {w.name}: {exc}")
            rows.append({"file": w.stem, "note": f"ERROR: {exc}"})
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["file", "original_s", "new_s", "removed_s", "removed_pct",
                                            "pauses_shortened", "note"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"\ndone: {sum('new_s' in r for r in rows)} files written to {out}")


if __name__ == "__main__":
    main()
