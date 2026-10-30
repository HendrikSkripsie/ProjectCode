#!/usr/bin/env python3
"""
concat_speakers.py - make one long wav per speaker by joining that speaker's first N recordings.

For the first S speaker folders (sorted by name: G0001, G0002, ...) it takes the first N
.wav files (sorted by name: G0001S0001.wav, G0001S0002.wav, ...) and joins them, in that
order, into  OUT/<speaker>.wav

It also writes, in OUT/:
  <speaker>_segments.csv   which file went where: file, start_s, end_s (+ transcription if --manifest)
  <speaker>.txt            the transcriptions joined (only with --manifest)
  concat_summary.csv       one row per speaker

Uses only the Python standard library (wave, csv).

Example
  python concat_speakers.py --src ~/Skripsie/DATA/Conrad_english/british_english/british_english/16kHz_Mono \
                            --out ~/Skripsie/DATA/Conrad_english/concatenated_20 \
                            --manifest ~/Skripsie/DATA/Conrad_english/british_english/british_english/manifest.csv
"""

import argparse
import csv
import sys
import wave
from pathlib import Path


def read_manifest(path):
    if not path:
        return {}
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, newline="", encoding=enc) as f:
                return {r["filename"].strip(): r.get("transcription", "").strip() for r in csv.DictReader(f)}
        except UnicodeDecodeError:
            continue
    sys.exit(f"could not read {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="the 16kHz_Mono folder (contains G0001, G0002, ...)")
    p.add_argument("--out", required=True, help="new folder for the joined files (created if needed)")
    p.add_argument("--speakers", type=int, default=32, help="how many speaker folders to use (default 32)")
    p.add_argument("--files", type=int, default=20, help="how many recordings per speaker (default 20)")
    p.add_argument("--gap-ms", type=float, default=0.0,
                   help="silence inserted between recordings, in ms (default 0 = joined directly)")
    p.add_argument("--manifest", help="optional manifest.csv, to save the transcriptions alongside")
    p.add_argument("--overwrite", action="store_true", help="replace existing output files")
    args = p.parse_args()

    src = Path(args.src).expanduser()
    out = Path(args.out).expanduser()
    if not src.is_dir():
        sys.exit(f"source folder not found: {src}")
    if out.resolve() == src.resolve() or src.resolve() in out.resolve().parents:
        sys.exit("choose an output folder OUTSIDE the source folder")

    speakers = sorted(d for d in src.iterdir() if d.is_dir())[: args.speakers]
    if len(speakers) < args.speakers:
        print(f"WARNING: only {len(speakers)} speaker folders found")
    print("speakers:", ", ".join(d.name for d in speakers))
    transcripts = read_manifest(args.manifest)
    out.mkdir(parents=True, exist_ok=True)

    summary = []
    for spk in speakers:
        wavs = sorted(spk.glob("*.wav"))[: args.files]
        target = out / f"{spk.name}.wav"
        if len(wavs) < args.files:
            print(f"WARNING {spk.name}: only {len(wavs)} wav files, using all of them")
        if not wavs:
            continue
        if target.exists() and not args.overwrite:
            sys.exit(f"{target} already exists (use --overwrite to replace it)")

        params, rows, pos = None, [], 0
        with wave.open(str(target), "wb") as w_out:
            for i, f in enumerate(wavs):
                with wave.open(str(f), "rb") as w_in:
                    p_in = w_in.getparams()
                    fmt = (p_in.nchannels, p_in.sampwidth, p_in.framerate, p_in.comptype)
                    if params is None:
                        params = fmt
                        w_out.setnchannels(p_in.nchannels)
                        w_out.setsampwidth(p_in.sampwidth)
                        w_out.setframerate(p_in.framerate)
                        gap = b"\x00" * (int(p_in.framerate * args.gap_ms / 1000) * p_in.nchannels * p_in.sampwidth)
                    elif fmt != params:
                        sys.exit(f"{f} has a different format {fmt} than {wavs[0].name} {params}; "
                                 "all files must share sample rate, channels and bit depth")
                    if i > 0 and gap:
                        w_out.writeframes(gap)
                        pos += len(gap) // (p_in.nchannels * p_in.sampwidth)
                    frames = w_in.readframes(w_in.getnframes())
                    n = w_in.getnframes()
                    w_out.writeframes(frames)
                    rows.append({"file": f.name, "start_s": round(pos / p_in.framerate, 4),
                                 "end_s": round((pos + n) / p_in.framerate, 4),
                                 "transcription": transcripts.get(f.name, "")})
                    pos += n
        dur = pos / params[2]

        with open(out / f"{spk.name}_segments.csv", "w", newline="", encoding="utf-8") as fh:
            fields = ["file", "start_s", "end_s"] + (["transcription"] if transcripts else [])
            wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(rows)
        if transcripts:
            (out / f"{spk.name}.txt").write_text(" ".join(r["transcription"] for r in rows) + "\n", encoding="utf-8")

        summary.append({"speaker": spk.name, "files_used": len(wavs), "first_file": wavs[0].name,
                        "last_file": wavs[-1].name, "duration_s": round(dur, 2), "output": target.name})
        print(f"{spk.name}: {len(wavs)} files ({wavs[0].name} .. {wavs[-1].name}) -> {target.name}  {dur:.1f} s")

    with open(out / "concat_summary.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["speaker", "files_used", "first_file", "last_file", "duration_s", "output"])
        wr.writeheader()
        wr.writerows(summary)
    print(f"\ndone: {len(summary)} files in {out}")


if __name__ == "__main__":
    main()
