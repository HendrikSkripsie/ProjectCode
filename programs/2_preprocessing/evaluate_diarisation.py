#!/usr/bin/env python3
"""
evaluate_diarisation.py - score preprocess_audio.py against ground-truth TextGrids.

Ground truth: one TextGrid per recording with an ADULT tier and a CHILD tier
(any non-empty interval = that person is speaking; the two may overlap).
Predictions: the output folder of preprocess_audio.py (one sub-folder per recording).

Everything is compared on a 10 ms grid over the annotated part of the
recording (0 .. xmax of the ground-truth TextGrid).

What is measured (per file, and over all files):

  VAD (step 1)            speech = ADULT or CHILD in the ground truth
      vad_recall          share of true speech the VAD kept
      vad_precision       share of VAD speech that really was speech

  Diarisation (step 2)
      DER                 Diarisation Error Rate = (missed + false alarm + confusion) / true speech
                          with the best possible matching of SPEAKER_xx to ADULT/CHILD
                          (the standard metric, comparable with the literature)
      role_error_rate     the same, but using the program's OWN child/adult decision
                          (what actually matters for this pipeline)
      child_id_correct    did the program pick the right speaker as the child?

  Final result (step 3, the "kept" audio)
      child_recall        share of the child's speech that ended up in the output
      child_precision     share of the output that is the child speaking
      adult_in_output_s   seconds of adult speech that leaked into the output
      nonspeech_in_output_s  seconds of silence/noise in the output

Usage
  python evaluate_diarisation.py --gt GROUNDTRUTH_FOLDER --pred preprocessed/ --out eval/
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

STEP = 0.01  # 10 ms frames


def log(msg):
    print(f"[evaluate] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

def read_textgrid(path: Path):
    """Long-format TextGrid -> (xmax, {tier name: [(start, end, label)]})."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    tiers, cur, iv, gxmax = {}, None, {}, None
    for line in text.splitlines():
        if (m := re.match(r'^\s*name\s*=\s*"(.*)"', line)):
            cur = m.group(1)
            tiers[cur] = []
        elif (m := re.match(r"^\s*(xmin|xmax)\s*=\s*([-0-9.eE+]+)", line)):
            if cur is None:
                if m.group(1) == "xmax":
                    gxmax = float(m.group(2))
            else:
                iv[m.group(1)] = float(m.group(2))
        elif cur is not None and (m := re.match(r'^\s*text\s*=\s*"(.*)"\s*$', line)):
            tiers[cur].append((iv["xmin"], iv["xmax"], m.group(1).strip()))
    if gxmax is None:
        raise ValueError(f"{path}: not a long-format TextGrid")
    return gxmax, tiers


def find_tier(tiers, name):
    for k, v in tiers.items():
        if k.strip().lower() == name.lower():
            return v
    return None


def read_csv_segments(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_frames(segs, n):
    """List of (start, end) -> boolean array of n 10-ms frames."""
    a = np.zeros(n, bool)
    for s, e in segs:
        i, j = max(0, int(round(s / STEP))), min(n, int(round(e / STEP)))
        if j > i:
            a[i:j] = True
    return a


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def der_counts(ref: dict, hyp: dict, mapping: dict, mask):
    """Frame counts for DER. mapping: hyp speaker -> ref speaker (or absent)."""
    n_ref = sum(r.astype(int) for r in ref.values())
    n_hyp = sum(h.astype(int) for h in hyp.values()) if hyp else np.zeros_like(n_ref)
    n_cor = np.zeros_like(n_ref)
    for h, r in mapping.items():
        n_cor += (hyp[h] & ref[r]).astype(int)
    miss = np.maximum(0, n_ref - n_hyp)
    fa = np.maximum(0, n_hyp - n_ref)
    conf = np.minimum(n_ref, n_hyp) - n_cor
    return {"ref": int(n_ref[mask].sum()), "miss": int(miss[mask].sum()),
            "fa": int(fa[mask].sum()), "conf": int(conf[mask].sum()),
            "_miss": miss, "_fa": fa, "_conf": conf}


def best_mapping(ref: dict, hyp: dict, mask):
    """Hyp->ref assignment that maximises overlapping time (each used at most once)."""
    rs, hs = list(ref), list(hyp)
    best, best_score = {}, -1
    if len(hs) <= len(rs):
        for perm in permutations(rs, len(hs)):
            score = sum(int((hyp[h] & ref[r] & mask).sum()) for h, r in zip(hs, perm))
            if score > best_score:
                best, best_score = dict(zip(hs, perm)), score
    else:
        for perm in permutations(hs, len(rs)):
            score = sum(int((hyp[h] & ref[r] & mask).sum()) for h, r in zip(perm, rs))
            if score > best_score:
                best, best_score = dict(zip(perm, rs)), score
    return best


def collar_mask(ref_segs_all, n, collar):
    mask = np.ones(n, bool)
    if collar <= 0:
        return mask
    c = int(round(collar / STEP))
    for s, e in ref_segs_all:
        for t in (s, e):
            k = int(round(t / STEP))
            mask[max(0, k - c):min(n, k + c)] = False
    return mask


def ratio(a, b):
    return a / b if b else None


def runs_to_intervals(labels: np.ndarray):
    """Array of strings per frame -> [(start, end, label)] for non-empty labels."""
    out = []
    if len(labels) == 0:
        return out
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    for a, b in zip(np.r_[0, change], np.r_[change, len(labels)]):
        if labels[a]:
            out.append((a * STEP, b * STEP, str(labels[a])))
    return out


# --------------------------------------------------------------------------- #
# TextGrid writer (for the error file)
# --------------------------------------------------------------------------- #

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
                  f"            xmax = {_fmt(e)} ", f'            text = "{lab}" ']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Per-file evaluation
# --------------------------------------------------------------------------- #

def evaluate_file(gt_path: Path, pred_dir: Path, collar: float, err_dir: Path | None):
    stem = gt_path.stem
    gt_end, gt_tiers = read_textgrid(gt_path)
    adult_t, child_t = find_tier(gt_tiers, "ADULT"), find_tier(gt_tiers, "CHILD")
    if adult_t is None or child_t is None:
        raise ValueError(f"needs tiers ADULT and CHILD, found {list(gt_tiers)}")
    n = int(round(gt_end / STEP))
    ref_segs = {"ADULT": [(s, e) for s, e, l in adult_t if l],
                "CHILD": [(s, e) for s, e, l in child_t if l]}
    ref = {k: to_frames(v, n) for k, v in ref_segs.items()}
    ref_speech = ref["ADULT"] | ref["CHILD"]
    mask = collar_mask([seg for v in ref_segs.values() for seg in v], n, collar)
    sec = lambda frames: round(float(frames) * STEP, 2)

    row = {"file": stem, "annotated_s": round(gt_end, 2),
           "ref_speech_s": sec((ref_speech & mask).sum()),
           "ref_child_s": sec((ref["CHILD"] & mask).sum()),
           "ref_adult_s": sec((ref["ADULT"] & mask).sum()),
           "ref_overlap_s": sec((ref["ADULT"] & ref["CHILD"] & mask).sum())}

    cfg_path = pred_dir / "run_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        row["diariser"] = cfg.get("diariser") if cfg.get("diarise") else "off"

    review = pred_dir / "review.TextGrid"
    audio_end = None
    if review.exists():
        audio_end, rv = read_textgrid(review)
        row["audio_s"] = round(audio_end, 2)
        if abs(audio_end - gt_end) > 2.0:
            row["note"] = f"ground truth covers {gt_end:.1f} s of {audio_end:.1f} s audio; scored up to {gt_end:.1f} s"

    # ---- VAD ---------------------------------------------------------------
    vad_csv = pred_dir / "01_vad" / "speech_segments.csv"
    if vad_csv.exists():
        vad = to_frames([(float(r["start"]), float(r["end"])) for r in read_csv_segments(vad_csv)], n)
        tp = (vad & ref_speech & mask).sum()
        row.update(vad_recall=ratio(tp, (ref_speech & mask).sum()), vad_precision=ratio(tp, (vad & mask).sum()),
                   vad_missed_s=sec((ref_speech & ~vad & mask).sum()),
                   vad_false_alarm_s=sec((vad & ~ref_speech & mask).sum()))

    # ---- diarisation -------------------------------------------------------
    turns_csv = pred_dir / "02_diarisation" / "speaker_turns.csv"
    hyp_role_frames = None
    if turns_csv.exists():
        turns = read_csv_segments(turns_csv)
        spk_segs, roles = {}, {}
        for r in turns:
            spk_segs.setdefault(r["speaker"], []).append((float(r["start"]), float(r["end"])))
            roles[r["speaker"]] = r.get("role", "")
        hyp = {k: to_frames(v, n) for k, v in spk_segs.items()}

        m_best = best_mapping(ref, hyp, mask)
        d = der_counts(ref, hyp, m_best, mask)
        role_map = {h: ("CHILD" if roles[h] == "child" else "ADULT") for h in hyp}
        # with >1 hyp speaker on the same role, only the best-overlapping one may count as correct
        dr = der_counts(ref, hyp, _unique(role_map, ref, hyp, mask), mask)
        child_spk = [h for h, r in roles.items() if r == "child"]
        row.update(
            hyp_speakers=len(hyp),
            DER=ratio(d["miss"] + d["fa"] + d["conf"], d["ref"]),
            der_missed=ratio(d["miss"], d["ref"]), der_false_alarm=ratio(d["fa"], d["ref"]),
            der_confusion=ratio(d["conf"], d["ref"]),
            role_error_rate=ratio(dr["miss"] + dr["fa"] + dr["conf"], dr["ref"]),
            child_id_correct=bool(child_spk and m_best.get(child_spk[0]) == "CHILD"),
            _der=d, _role=dr,
        )
        hyp_role_frames = {"CHILD": np.zeros(n, bool), "ADULT": np.zeros(n, bool)}
        for h, frames in hyp.items():
            hyp_role_frames[role_map[h]] |= frames

    # ---- final kept audio --------------------------------------------------
    kept = None
    if review.exists():
        kept_t = find_tier(rv, "kept")
        if kept_t is not None:
            kept = to_frames([(s, e) for s, e, l in kept_t if l], n)
            tp = (kept & ref["CHILD"] & mask).sum()
            row.update(child_recall=ratio(tp, (ref["CHILD"] & mask).sum()),
                       child_precision=ratio(tp, (kept & mask).sum()),
                       kept_s=sec((kept & mask).sum()),
                       adult_in_output_s=sec((kept & ref["ADULT"] & ~ref["CHILD"] & mask).sum()),
                       overlap_in_output_s=sec((kept & ref["ADULT"] & ref["CHILD"] & mask).sum()),
                       nonspeech_in_output_s=sec((kept & ~ref_speech & mask).sum()),
                       child_missing_s=sec((~kept & ref["CHILD"] & mask).sum()))
            p, r = row["child_precision"], row["child_recall"]
            row["child_f1"] = 2 * p * r / (p + r) if p and r else 0.0

    # ---- error TextGrid for Praat -----------------------------------------
    if err_dir is not None:
        tiers = [("GT_ADULT", [(s, e, "ADULT") for s, e in ref_segs["ADULT"]]),
                 ("GT_CHILD", [(s, e, "CHILD") for s, e in ref_segs["CHILD"]])]
        if hyp_role_frames is not None:
            lab = np.where(hyp_role_frames["CHILD"] & hyp_role_frames["ADULT"], "both",
                           np.where(hyp_role_frames["CHILD"], "child", np.where(hyp_role_frames["ADULT"], "adult", "")))
            tiers.append(("predicted_role", runs_to_intervals(lab)))
            dr = row["_role"]
            err = np.where(dr["_conf"] > 0, "confusion", np.where(dr["_miss"] > 0, "missed",
                           np.where(dr["_fa"] > 0, "false alarm", "")))
            err = np.where(mask, err, "")
            tiers.append(("diarisation_errors", runs_to_intervals(err)))
        if kept is not None:
            tiers.append(("kept", runs_to_intervals(np.where(kept, "keep", ""))))
            out_err = np.where(kept & ~ref["CHILD"] & ref["ADULT"], "adult kept",
                               np.where(kept & ~ref_speech, "non-speech kept",
                                        np.where(~kept & ref["CHILD"], "child lost", "")))
            tiers.append(("output_errors", runs_to_intervals(np.where(mask, out_err, ""))))
        write_textgrid(err_dir / f"{stem}.errors.TextGrid", audio_end or gt_end, tiers)
    return row


def _unique(role_map, ref, hyp, mask):
    """If several hyp speakers share a role, map only the one overlapping that ref speaker most."""
    out = {}
    for role in set(role_map.values()):
        cands = [h for h, r in role_map.items() if r == role]
        best = max(cands, key=lambda h: int((hyp[h] & ref[role] & mask).sum()))
        out[best] = role
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

FIELDS = ["file", "diariser", "annotated_s", "audio_s", "ref_speech_s", "ref_child_s", "ref_adult_s",
          "ref_overlap_s", "vad_recall", "vad_precision", "vad_missed_s", "vad_false_alarm_s",
          "hyp_speakers", "DER", "der_missed", "der_false_alarm", "der_confusion", "role_error_rate",
          "child_id_correct", "kept_s", "child_recall", "child_precision", "child_f1",
          "adult_in_output_s", "overlap_in_output_s", "nonspeech_in_output_s", "child_missing_s", "note"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt", required=True, help="folder with ground-truth TextGrids (searched recursively) or one file")
    p.add_argument("--pred", required=True, help="output folder of preprocess_audio.py")
    p.add_argument("--out", default="evaluation", help="where to write the results (default: evaluation/)")
    p.add_argument("--collar", type=float, default=0.0,
                   help="seconds ignored around every ground-truth boundary (0.25 = standard NIST DER)")
    p.add_argument("--no-error-textgrids", action="store_true", help="don't write the per-file error TextGrids")
    args = p.parse_args(argv)

    gt = Path(args.gt).expanduser()
    pred = Path(args.pred).expanduser()
    out = Path(args.out).expanduser()
    gt_files = sorted(gt.rglob("*.TextGrid")) if gt.is_dir() else [gt]
    pairs = [(g, pred / g.stem) for g in gt_files if (pred / g.stem).is_dir()]
    if not pairs:
        sys.exit(f"[evaluate] no matches: none of the {len(gt_files)} ground-truth names has a folder in {pred}")
    log(f"{len(pairs)} recording(s) have both ground truth and predictions "
        f"({len(gt_files) - len(pairs)} ground-truth files without predictions skipped)")

    rows = []
    for g, d in pairs:
        try:
            rows.append(evaluate_file(g, d, args.collar, None if args.no_error_textgrids else out / "error_textgrids"))
        except Exception as exc:
            log(f"WARNING {g.name}: {exc}")
    if not rows:
        sys.exit("[evaluate] nothing could be evaluated")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_file.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    # ---- totals: summed over all files (time-weighted) ----------------------
    S = lambda k: sum(r.get(k) or 0 for r in rows)
    tot = {"files": len(rows), "collar_s": args.collar}
    have_der = [r for r in rows if "_der" in r]
    if have_der:
        d = {k: sum(r["_der"][k] for r in have_der) for k in ("ref", "miss", "fa", "conf")}
        dr = {k: sum(r["_role"][k] for r in have_der) for k in ("ref", "miss", "fa", "conf")}
        tot.update(DER=ratio(d["miss"] + d["fa"] + d["conf"], d["ref"]),
                   der_missed=ratio(d["miss"], d["ref"]), der_false_alarm=ratio(d["fa"], d["ref"]),
                   der_confusion=ratio(d["conf"], d["ref"]),
                   role_error_rate=ratio(dr["miss"] + dr["fa"] + dr["conf"], dr["ref"]),
                   child_id_accuracy=ratio(sum(bool(r["child_id_correct"]) for r in have_der), len(have_der)))
    have_vad = [r for r in rows if "vad_missed_s" in r]
    if have_vad:
        ref_sp = sum(r["ref_speech_s"] for r in have_vad)
        tot.update(vad_recall=ratio(ref_sp - sum(r["vad_missed_s"] for r in have_vad), ref_sp),
                   vad_missed_s=round(sum(r["vad_missed_s"] for r in have_vad), 1),
                   vad_false_alarm_s=round(sum(r["vad_false_alarm_s"] for r in have_vad), 1))
    have_kept = [r for r in rows if "kept_s" in r]
    if have_kept:
        child = sum(r["ref_child_s"] for r in have_kept)
        kept = sum(r["kept_s"] for r in have_kept)
        good = child - sum(r["child_missing_s"] for r in have_kept)
        pr, rc = ratio(good, kept), ratio(good, child)
        tot.update(child_recall=rc, child_precision=pr,
                   child_f1=(2 * pr * rc / (pr + rc)) if pr and rc else 0.0,
                   adult_in_output_s=round(sum(r["adult_in_output_s"] for r in have_kept), 1),
                   nonspeech_in_output_s=round(sum(r["nonspeech_in_output_s"] for r in have_kept), 1),
                   child_missing_s=round(sum(r["child_missing_s"] for r in have_kept), 1))
    (out / "overall.json").write_text(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                                  for k, v in tot.items()}, indent=2))

    pct = lambda v: "  n/a" if v is None else f"{100 * v:5.1f}%"
    print()
    print(f"  Evaluated {len(rows)} recording(s), collar {args.collar} s")
    if have_vad:
        print(f"  VAD          recall {pct(tot['vad_recall'])}   missed {tot['vad_missed_s']} s, "
              f"false alarm {tot['vad_false_alarm_s']} s")
    if have_der:
        print(f"  Diarisation  DER {pct(tot['DER'])}  (missed {pct(tot['der_missed'])}, false alarm "
              f"{pct(tot['der_false_alarm'])}, confusion {pct(tot['der_confusion'])})")
        print(f"               role error rate {pct(tot['role_error_rate'])}   "
              f"child picked correctly in {pct(tot['child_id_accuracy'])} of files")
    if have_kept:
        print(f"  Output       child recall {pct(tot['child_recall'])}   precision {pct(tot['child_precision'])}"
              f"   F1 {pct(tot['child_f1'])}")
        print(f"               adult speech kept {tot['adult_in_output_s']} s, non-speech kept "
              f"{tot['nonspeech_in_output_s']} s, child speech lost {tot['child_missing_s']} s")
    print()
    log(f"per-file results: {out / 'per_file.csv'}")
    log(f"totals:           {out / 'overall.json'}")
    if not args.no_error_textgrids:
        log(f"error TextGrids:  {out / 'error_textgrids'}/  (open with the original wav in Praat)")
    for r in rows:
        if r.get("note"):
            log(f"note {r['file']}: {r['note']}")


if __name__ == "__main__":
    main()
