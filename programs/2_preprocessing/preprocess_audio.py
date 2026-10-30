#!/usr/bin/env python3
"""
preprocess_audio.py - audio preprocessing for Afrikaans child speech (adult + child recordings).

Steps (each can be switched off; each writes its own inspectable output):

    1. VAD          remove non-speech (silence, noise, rustling)     -> 01_vad/
    2. Diarisation  who speaks when; find the child; drop the adult  -> 02_diarisation/
    3. Render       keep (speech AND child) - cut or mute the rest   -> 03_final/

All decisions are made as time segments on the ORIGINAL timeline; audio is only
edited at the very end. A review.TextGrid (tiers: vad | speaker | role | kept)
lets you check every decision in Praat on top of the original wav.

Backends
    VAD:          Silero VAD (PyTorch, model ships with the pip package)
    Diarisation:  pyannote  - pyannote.audio community-1 pipeline (best; needs a HuggingFace token)
                  simple    - speaker embeddings (Resemblyzer GE2E encoder) + clustering
                              (no token, quick; no overlap detection)

Usage
    python preprocess_audio.py run  recording.wav --out preprocessed/
    python preprocess_audio.py run  audio_folder/ --out preprocessed/ --no-vad
    python preprocess_audio.py map-textgrid final.TextGrid time_map.csv original.TextGrid
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")  # no usage telemetry from pyannote
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_SR = 16000  # all models run at 16 kHz mono
Seg = tuple[float, float]

# --------------------------------------------------------------------------- #
# Default settings (any of these can be put in a JSON config file)
# --------------------------------------------------------------------------- #

DEFAULTS = {
    # step switches
    "vad": True,
    "diarise": True,
    # output
    "mode": "cut",               # cut | mute
    "fade_ms": 10.0,             # fade at every edit point (prevents clicks)
    "join_gap_ms": 0.0,          # cut mode: silence inserted between kept chunks
    "min_keep_ms": 100.0,        # drop kept pieces shorter than this
    "listen_gap_ms": 150.0,      # gap between pieces in the removed_*/adult_only listening files
    # VAD (Silero)
    "vad_threshold": 0.5,
    "min_speech_ms": 250,
    "min_silence_ms": 300,
    "speech_pad_ms": 100,
    # diarisation
    "diariser": "pyannote",      # pyannote | simple
    "pyannote_model": "pyannote/speaker-diarization-community-1",
    "hf_token": None,            # or env HF_TOKEN
    "num_speakers": 2,           # 0 = let the diariser decide
    "keep_overlap": False,       # keep regions where child and adult speak at once
    # child selection
    "child_rule": "auto",        # auto | pitch | talktime
    "pitch_margin": 0.15,        # auto: pitch decides alone only if speakers differ by >= 15 %
    "f0_min": 70.0,
    "f0_max": 700.0,
    "overrides": None,           # CSV: file,child_speaker
    # misc
    "reuse_diarisation": True,   # reuse cached diarisation when its settings are unchanged
}

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #


def log(msg: str) -> None:
    print(f"[preprocess] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[preprocess] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), sr  # mono


def to_model_rate(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == MODEL_SR:
        return audio
    import librosa
    return librosa.resample(audio, orig_sr=sr, target_sr=MODEL_SR).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1, 1), sr, subtype="PCM_16")


# ---- segment algebra ------------------------------------------------------ #

def merge(segs: list[Seg], gap: float = 0.0) -> list[Seg]:
    out: list[list[float]] = []
    for s, e in sorted(segs):
        if e <= s:
            continue
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def intersect(a: list[Seg], b: list[Seg]) -> list[Seg]:
    a, b = merge(a), merge(b)
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(a: list[Seg], b: list[Seg]) -> list[Seg]:
    out = []
    b = merge(b)
    for s, e in merge(a):
        cur = s
        for bs, be in b:
            if be <= cur or bs >= e:
                continue
            if bs > cur:
                out.append((cur, bs))
            cur = max(cur, be)
        if cur < e:
            out.append((cur, e))
    return out


def complement(segs: list[Seg], dur: float) -> list[Seg]:
    return subtract([(0.0, dur)], segs)


def total(segs: list[Seg]) -> float:
    return float(sum(e - s for s, e in merge(segs)))


# ---- rendering ------------------------------------------------------------ #

def _fade(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))).astype(np.float32) if n > 0 else np.ones(0, np.float32)


def render_cut(audio, sr, segs, fade_ms, gap_ms):
    """Concatenate segments. Returns (audio, time_map rows)."""
    nf = int(sr * fade_ms / 1000)
    gap = np.zeros(int(sr * gap_ms / 1000), np.float32)
    pieces, rows, pos = [], [], 0
    for k, (s, e) in enumerate(merge(segs)):
        a, b = int(round(s * sr)), int(round(e * sr))
        chunk = audio[a:b].copy()
        if len(chunk) == 0:
            continue
        f = min(nf, len(chunk) // 2)
        if f:
            chunk[:f] *= _fade(f)
            chunk[-f:] *= _fade(f)[::-1]
        if k and len(gap):
            pieces.append(gap)
            pos += len(gap)
        pieces.append(chunk)
        rows.append({"out_start": pos / sr, "out_end": (pos + len(chunk)) / sr,
                     "orig_start": a / sr, "orig_end": b / sr})
        pos += len(chunk)
    out = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
    return out, rows


def render_mute(audio, sr, segs, fade_ms):
    """Keep segments, silence the rest (same length as the input)."""
    gain = np.zeros(len(audio), np.float32)
    nf = int(sr * fade_ms / 1000)
    ramp = _fade(nf)
    for s, e in merge(segs):
        a, b = max(0, int(round(s * sr))), min(len(audio), int(round(e * sr)))
        gain[a:b] = 1.0
        if nf:  # ramps sit in the removed region, so kept audio is untouched
            lo = max(0, a - nf)
            gain[lo:a] = np.maximum(gain[lo:a], ramp[nf - (a - lo):])
            hi = min(len(audio), b + nf)
            gain[b:hi] = np.maximum(gain[b:hi], ramp[::-1][: hi - b])
    rows = [{"out_start": s, "out_end": e, "orig_start": s, "orig_end": e} for s, e in merge(segs)]
    return audio * gain, rows


def render(audio, sr, segs, cfg, listening=False):
    if listening:  # removed/adult listening files: always cut, with audible gaps
        return render_cut(audio, sr, segs, cfg["fade_ms"], cfg["listen_gap_ms"])[0]
    if cfg["mode"] == "mute":
        return render_mute(audio, sr, segs, cfg["fade_ms"])[0]
    return render_cut(audio, sr, segs, cfg["fade_ms"], cfg["join_gap_ms"])[0]


# ---- TextGrid ------------------------------------------------------------- #

def _fmt(x: float) -> str:
    s = f"{x:.6f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def fill_tier(ivs: list[tuple[float, float, str]], dur: float) -> list[tuple[float, float, str]]:
    out, t = [], 0.0
    for s, e, lab in sorted(ivs):
        s, e = max(s, t), min(e, dur)
        if e - s < 1e-6:
            continue
        if s > t + 1e-6:
            out.append((t, s, ""))
        out.append((s, e, lab))
        t = e
    if dur > t + 1e-6:
        out.append((t, dur, ""))
    return out or [(0.0, dur, "")]


def write_textgrid(path: Path, dur: float, tiers: list[tuple[str, list[tuple[float, float, str]]]]) -> None:
    L = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "", "xmin = 0 ",
         f"xmax = {_fmt(dur)} ", "tiers? <exists> ", f"size = {len(tiers)} ", "item []: "]
    for i, (name, ivs) in enumerate(tiers, 1):
        ivs = fill_tier(ivs, dur)
        L += [f"    item [{i}]:", '        class = "IntervalTier" ', f'        name = "{name}" ',
              "        xmin = 0 ", f"        xmax = {_fmt(dur)} ", f"        intervals: size = {len(ivs)} "]
        for j, (s, e, t) in enumerate(ivs, 1):
            L += [f"        intervals [{j}]:", f"            xmin = {_fmt(s)} ",
                  f"            xmax = {_fmt(e)} ", f'            text = "{t.replace(chr(34), chr(34) * 2)}" ']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def read_textgrid(path: Path):
    """Minimal long-format TextGrid reader -> (xmax, [(name, [(s, e, text)])])."""
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


# --------------------------------------------------------------------------- #
# Step 1: VAD (Silero)
# --------------------------------------------------------------------------- #

_SILERO = None


def run_vad(audio16: np.ndarray, cfg) -> list[Seg]:
    global _SILERO
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    if _SILERO is None:
        _SILERO = load_silero_vad()
    ts = get_speech_timestamps(
        torch.from_numpy(audio16), _SILERO, sampling_rate=MODEL_SR,
        threshold=cfg["vad_threshold"], min_speech_duration_ms=int(cfg["min_speech_ms"]),
        min_silence_duration_ms=int(cfg["min_silence_ms"]), speech_pad_ms=int(cfg["speech_pad_ms"]),
        return_seconds=False,
    )
    return merge([(t["start"] / MODEL_SR, t["end"] / MODEL_SR) for t in ts])


# --------------------------------------------------------------------------- #
# Step 2: diarisation
# --------------------------------------------------------------------------- #

_PYANNOTE = {}


def diarise_pyannote(audio16: np.ndarray, cfg) -> list[tuple[float, float, str]]:
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError:
        die("pyannote.audio is not installed (pip install pyannote.audio) - or use --diariser simple")
    token = cfg.get("hf_token") or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    name = cfg["pyannote_model"]
    if name not in _PYANNOTE:
        log(f"loading {name} (first time downloads the model from HuggingFace)")
        try:
            pipe = Pipeline.from_pretrained(name, token=token)
        except TypeError:  # pyannote.audio 3.x
            pipe = Pipeline.from_pretrained(name, use_auth_token=token)
        if pipe is None:
            die(f"could not load {name}. Check your HF token and that you accepted the model's "
                "conditions on huggingface.co (see README).")
        pipe.to(torch.device("cpu"))
        _PYANNOTE[name] = pipe
    kwargs = {"num_speakers": int(cfg["num_speakers"])} if int(cfg["num_speakers"]) > 0 else {}
    out = _PYANNOTE[name]({"waveform": torch.from_numpy(audio16)[None], "sample_rate": MODEL_SR}, **kwargs)
    ann = getattr(out, "speaker_diarization", out)  # 4.x returns DiarizeOutput, 3.x an Annotation
    turns = [(float(t.start), float(t.end), str(spk)) for t, _, spk in ann.itertracks(yield_label=True)]
    return normalise_labels(turns)


class _GE2E:
    """Resemblyzer's GE2E speaker encoder (weights from the resemblyzer pip package)."""

    def __init__(self):
        import importlib.util
        import torch
        from torch import nn
        spec = importlib.util.find_spec("resemblyzer")
        if spec is None or not spec.submodule_search_locations:
            die("the simple diariser needs the encoder weights: pip install --no-deps resemblyzer")
        weights = Path(list(spec.submodule_search_locations)[0]) / "pretrained.pt"

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.lstm = nn.LSTM(40, 256, 3, batch_first=True)
                s.linear = nn.Linear(256, 256)
                s.relu = nn.ReLU()

            def forward(s, mels):
                _, (h, _) = s.lstm(mels)
                e = s.relu(s.linear(h[-1]))
                return e / torch.norm(e, dim=1, keepdim=True).clamp_min(1e-9)

        self.net = Net()
        self.net.load_state_dict(torch.load(weights, map_location="cpu")["model_state"], strict=False)
        self.net.eval()
        self.torch = torch

    def mel(self, audio16):
        import librosa
        rms = np.sqrt(np.mean(audio16 ** 2)) + 1e-9
        y = audio16 * (10 ** (-30 / 20) / rms)  # -30 dBFS, as the encoder was trained
        return librosa.feature.melspectrogram(y=y, sr=MODEL_SR, n_fft=400, hop_length=160,
                                              n_mels=40).astype(np.float32).T  # 10 ms frames

    def embed(self, mels_list):
        with self.torch.no_grad():
            return np.stack([self.net(self.torch.from_numpy(m)[None])[0].numpy() for m in mels_list])


_ENCODER = None


def diarise_simple(audio16: np.ndarray, speech: list[Seg], cfg) -> list[tuple[float, float, str]]:
    """Embeddings on 1.5 s windows (0.5 s hop) inside speech, clustered; frame-level voting."""
    global _ENCODER
    from sklearn.cluster import AgglomerativeClustering
    if _ENCODER is None:
        _ENCODER = _GE2E()
    mel = _ENCODER.mel(audio16)
    win, hop, minlen = 150, 50, 50  # frames (10 ms)
    windows, spans = [], []
    for s, e in speech:
        a, b = int(s * 100), min(len(mel), int(e * 100))
        if b - a < minlen:
            continue
        starts = list(range(a, b - win + 1, hop)) if b - a >= win else [a]
        if b - a >= win and starts[-1] + win < b:
            starts.append(b - win)
        for st in starts:
            windows.append(mel[st:min(b, st + win)])
            spans.append((st, min(b, st + win)))
    if len(windows) < 2:
        return [(s, e, "SPEAKER_00") for s, e in speech]
    emb = _ENCODER.embed(windows)
    n = int(cfg["num_speakers"])
    clus = (AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average") if n > 0 else
            AgglomerativeClustering(n_clusters=None, distance_threshold=0.35, metric="cosine", linkage="average"))
    labels = clus.fit_predict(emb)
    k = labels.max() + 1
    votes = np.zeros((len(mel), k), np.float32)
    for (a, b), lab in zip(spans, labels):
        votes[a:b, lab] += 1
    centres = np.array([(a + b) / 2 for a, b in spans])
    turns = []
    for s, e in speech:
        a, b = int(s * 100), min(len(mel), int(e * 100))
        if b <= a:
            continue
        v = votes[a:b]
        frame_lab = v.argmax(axis=1)
        empty = v.sum(axis=1) == 0  # short segments: label of the nearest window
        if empty.any():
            idx = np.where(empty)[0]
            nearest = np.abs(centres[None, :] - (a + idx)[:, None]).argmin(axis=1)
            frame_lab[idx] = labels[nearest]
        cut = np.flatnonzero(np.diff(frame_lab)) + 1
        for r0, r1 in zip(np.r_[0, cut], np.r_[cut, b - a]):
            turns.append(((a + r0) / 100, (a + r1) / 100, f"S{frame_lab[r0]}"))
    # absorb very short turns (< 0.3 s) into the previous turn
    cleaned = []
    for t in turns:
        if cleaned and t[1] - t[0] < 0.3 and abs(cleaned[-1][1] - t[0]) < 0.011:
            cleaned[-1] = (cleaned[-1][0], t[1], cleaned[-1][2])
        elif cleaned and cleaned[-1][2] == t[2] and abs(cleaned[-1][1] - t[0]) < 0.011:
            cleaned[-1] = (cleaned[-1][0], t[1], t[2])
        else:
            cleaned.append(t)
    return normalise_labels(cleaned)


def normalise_labels(turns):
    """Rename speakers SPEAKER_00, SPEAKER_01 ... in order of first appearance."""
    order = {}
    for s, e, lab in sorted(turns):
        order.setdefault(lab, f"SPEAKER_{len(order):02d}")
    return sorted((s, e, order[lab]) for s, e, lab in turns)


def speaker_regions(turns) -> dict[str, list[Seg]]:
    regs: dict[str, list[Seg]] = {}
    for s, e, lab in turns:
        regs.setdefault(lab, []).append((s, e))
    return {k: merge(v) for k, v in regs.items()}


def overlap_regions(regs: dict[str, list[Seg]]) -> list[Seg]:
    labs = list(regs)
    ov = []
    for i in range(len(labs)):
        for j in range(i + 1, len(labs)):
            ov += intersect(regs[labs[i]], regs[labs[j]])
    return merge(ov)


# ---- deciding which speaker is the child ---------------------------------- #

def pitch_track(audio16: np.ndarray, cfg):
    import librosa
    f0, voiced, _ = librosa.pyin(audio16, fmin=cfg["f0_min"], fmax=cfg["f0_max"], sr=MODEL_SR,
                                 frame_length=1024, hop_length=160)
    times = librosa.times_like(f0, sr=MODEL_SR, hop_length=160)
    return times, np.where(voiced, f0, np.nan)


def median_f0(times, f0, segs: list[Seg]) -> float | None:
    if not segs:
        return None
    mask = np.zeros(len(times), bool)
    for s, e in segs:
        mask |= (times >= s) & (times < e)
    vals = f0[mask]
    vals = vals[~np.isnan(vals)]
    return float(np.median(vals)) if len(vals) >= 20 else None


def load_overrides(path) -> dict[str, str]:
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        die(f"overrides file {p} not found")
    with open(p, newline="", encoding="utf-8-sig") as f:
        return {Path(r["file"].strip()).stem: r["child_speaker"].strip() for r in csv.DictReader(f)
                if r.get("file") and r.get("child_speaker")}


def choose_child(stem, regs, overlap, times, f0, cfg, overrides):
    """Return (child_label, info dict per speaker, decision dict)."""
    info = {}
    for lab, segs in regs.items():
        excl = subtract(segs, overlap)
        info[lab] = {"talk_time_s": round(total(excl), 2), "median_f0_hz": None, "turns": len(segs)}
        mf = median_f0(times, f0, excl)
        info[lab]["median_f0_hz"] = round(mf, 1) if mf else None
    labs = list(regs)
    decision = {"rule": cfg["child_rule"], "flag": ""}
    if stem in overrides:
        child = overrides[stem]
        if child not in regs:
            die(f"override for {stem} names {child}, but speakers are {labs}")
        decision.update(basis="manual override")
        return child, info, decision
    if len(labs) == 1:
        decision.update(basis="only one speaker found", flag="only one speaker found - check")
        return labs[0], info, decision

    by_talk = max(labs, key=lambda l: info[l]["talk_time_s"])
    with_f0 = [l for l in labs if info[l]["median_f0_hz"]]
    by_pitch = max(with_f0, key=lambda l: info[l]["median_f0_hz"]) if with_f0 else None
    margin = None
    if len(with_f0) >= 2:
        f = sorted(info[l]["median_f0_hz"] for l in with_f0)
        margin = (f[-1] - f[-2]) / f[-2]
    decision.update(by_pitch=by_pitch, by_talktime=by_talk,
                    pitch_difference=round(margin, 3) if margin is not None else None)

    rule = cfg["child_rule"]
    if rule == "pitch" and by_pitch:
        child, basis = by_pitch, "highest median pitch"
    elif rule == "talktime" or by_pitch is None:
        child, basis = by_talk, "most talk time"
    elif by_pitch == by_talk:
        child, basis = by_pitch, "pitch and talk time agree"
    elif margin is not None and margin >= cfg["pitch_margin"]:
        child, basis = by_pitch, "pitch (clear difference) - disagrees with talk time"
    else:
        child, basis = by_talk, "talk time (pitch difference too small) - disagrees with pitch"
    flags = []
    if by_pitch and by_pitch != by_talk:
        flags.append("pitch and talk time disagree")
    if margin is not None and margin < cfg["pitch_margin"]:
        flags.append(f"small pitch difference ({margin:.0%})")
    if len(labs) > 2:
        flags.append(f"{len(labs)} speakers found")
    decision.update(basis=basis, flag="; ".join(flags))
    return child, info, decision


# --------------------------------------------------------------------------- #
# Per-file pipeline
# --------------------------------------------------------------------------- #

def diar_cache_key(cfg, speech_for_simple) -> str:
    keys = {k: cfg[k] for k in ("diariser", "pyannote_model", "num_speakers")}
    if cfg["diariser"] == "simple":
        keys.update({k: cfg[k] for k in ("vad_threshold", "min_speech_ms", "min_silence_ms", "speech_pad_ms")})
    return hashlib.md5(json.dumps(keys, sort_keys=True).encode()).hexdigest()


def write_segments_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("start,end\n")
            return
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})


def process_file(wav: Path, out_root: Path, cfg, overrides) -> dict:
    stem = wav.stem
    out = out_root / stem
    for d in ("01_vad", "02_diarisation", "03_final"):
        shutil.rmtree(out / d, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    audio, sr = load_audio(wav)
    audio16 = to_model_rate(audio, sr)
    dur = len(audio) / sr
    everything = [(0.0, dur)]
    summary = {"file": wav.name, "duration_s": round(dur, 2), "vad": cfg["vad"], "diarise": cfg["diarise"],
               "mode": cfg["mode"]}
    tiers = []

    # ---- step 1: VAD --------------------------------------------------------
    speech = None
    if cfg["vad"] or (cfg["diarise"] and cfg["diariser"] == "simple"):
        speech = run_vad(audio16, cfg)
    if cfg["vad"]:
        d = out / "01_vad"
        write_segments_csv(d / "speech_segments.csv",
                           [{"start": s, "end": e, "duration": e - s} for s, e in speech])
        write_wav(d / "speech_only.wav", render(audio, sr, speech, cfg), sr)
        write_wav(d / "removed_by_vad.wav", render(audio, sr, complement(speech, dur), cfg, listening=True), sr)
        tiers.append(("vad", [(s, e, "speech") for s, e in speech]))
        summary["speech_s"] = round(total(speech), 2)
        log(f"{stem}: VAD kept {total(speech):.1f} s of {dur:.1f} s in {len(speech)} segments")
    vad_keep = speech if cfg["vad"] else everything

    # ---- step 2: diarisation ------------------------------------------------
    child_keep = everything
    if cfg["diarise"]:
        d = out / "02_diarisation"
        cache = out / ".cache" / "diarisation.json"
        key = diar_cache_key(cfg, speech)
        turns = None
        if cfg["reuse_diarisation"] and cache.exists():
            c = json.loads(cache.read_text())
            if c.get("key") == key:
                turns = [tuple(t) for t in c["turns"]]
                log(f"{stem}: reusing cached diarisation ({cache})")
        if turns is None:
            td = time.time()
            if cfg["diariser"] == "pyannote":
                turns = diarise_pyannote(audio16, cfg)
            elif cfg["diariser"] == "simple":
                turns = diarise_simple(audio16, speech, cfg)
            else:
                die(f"unknown diariser {cfg['diariser']}")
            log(f"{stem}: diarisation ({cfg['diariser']}) took {time.time() - td:.0f} s")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"key": key, "diariser": cfg["diariser"], "turns": turns}))

        regs = speaker_regions(turns)
        overlap = overlap_regions(regs)
        times, f0 = pitch_track(audio16, cfg)
        child, info, decision = choose_child(stem, regs, overlap, times, f0, cfg, overrides)
        child_segs = regs[child]
        adult_segs = merge([s for lab, segs in regs.items() if lab != child for s in segs])
        child_keep = child_segs if cfg["keep_overlap"] else subtract(child_segs, overlap)

        for lab in info:
            info[lab]["role"] = "child" if lab == child else "adult"
        write_segments_csv(d / "speaker_turns.csv",
                           [{"start": s, "end": e, "speaker": lab, "role": info[lab]["role"]} for s, e, lab in turns])
        with open(d / "speaker_turns.rttm", "w") as f:
            for s, e, lab in turns:
                f.write(f"SPEAKER {stem} 1 {s:.3f} {e - s:.3f} <NA> <NA> {lab} <NA> <NA>\n")
        (d / "speakers.json").write_text(json.dumps(
            {"child_speaker": child, "decision": decision, "speakers": info,
             "overlap_s": round(total(overlap), 2), "diariser": cfg["diariser"]}, indent=2))
        write_wav(d / "child_only.wav", render(audio, sr, child_keep, cfg), sr)
        write_wav(d / "adult_only.wav", render(audio, sr, subtract(adult_segs, overlap), cfg, listening=True), sr)
        if overlap:
            write_wav(d / "overlap.wav", render(audio, sr, overlap, cfg, listening=True), sr)

        tiers.append(("speaker", _speaker_tier(regs, overlap)))
        role_ivs = ([(s, e, "child") for s, e in subtract(child_segs, overlap)] +
                    [(s, e, "adult") for s, e in subtract(adult_segs, overlap)] +
                    [(s, e, "overlap") for s, e in overlap])
        tiers.append(("role", role_ivs))
        summary.update(child_speaker=child, child_s=round(total(subtract(child_segs, overlap)), 2),
                       adult_s=round(total(subtract(adult_segs, overlap)), 2),
                       overlap_s=round(total(overlap), 2), decision=decision["basis"], flag=decision["flag"])
        for lab, i in info.items():
            log(f"{stem}: {lab} ({i['role']}): talk {i['talk_time_s']} s, median F0 {i['median_f0_hz']} Hz")
        log(f"{stem}: child = {child} ({decision['basis']})" + (f"  CHECK: {decision['flag']}" if decision["flag"] else ""))

    # ---- step 3: combine + render -------------------------------------------
    kept = [(s, e) for s, e in intersect(vad_keep, child_keep) if (e - s) * 1000 >= cfg["min_keep_ms"]]
    d = out / "03_final"
    if cfg["mode"] == "mute":
        final, rows = render_mute(audio, sr, kept, cfg["fade_ms"])
    else:
        final, rows = render_cut(audio, sr, kept, cfg["fade_ms"], cfg["join_gap_ms"])
    write_wav(d / f"{stem}.wav", final, sr)
    write_wav(d / "removed.wav", render(audio, sr, complement(kept, dur), cfg, listening=True), sr)
    write_segments_csv(d / "time_map.csv", rows)
    final_dur = len(final) / sr
    write_textgrid(d / f"{stem}.source.TextGrid", final_dur,
                   [("source", [(r["out_start"], r["out_end"], f"{r['orig_start']:.2f}-{r['orig_end']:.2f}")
                                for r in rows])])
    tiers.append(("kept", [(s, e, "keep") for s, e in kept]))
    write_textgrid(out / "review.TextGrid", dur, tiers)
    (out / "run_config.json").write_text(json.dumps(
        {k: v for k, v in cfg.items() if k != "hf_token"}, indent=2))

    summary.update(kept_s=round(total(kept), 2), final_duration_s=round(final_dur, 2),
                   seconds_taken=round(time.time() - t0, 1))
    log(f"{stem}: kept {total(kept):.1f} s -> {d / (stem + '.wav')} ({final_dur:.1f} s, {cfg['mode']} mode)")
    return summary


def _speaker_tier(regs, overlap):
    ivs = []
    for lab, segs in regs.items():
        ivs += [(s, e, lab) for s, e in subtract(segs, overlap)]
    labs_at = lambda t: "+".join(sorted(l for l, segs in regs.items() if any(s <= t < e for s, e in segs)))
    ivs += [(s, e, labs_at((s + e) / 2)) for s, e in overlap]
    return ivs


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

SUMMARY_FIELDS = ["file", "duration_s", "vad", "diarise", "mode", "speech_s", "child_speaker", "child_s",
                  "adult_s", "overlap_s", "kept_s", "final_duration_s", "decision", "flag", "seconds_taken"]


def cmd_run(args) -> None:
    cfg = dict(DEFAULTS)
    if args.config:
        user = json.loads(Path(args.config).read_text())
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            die(f"unknown keys in {args.config}: {sorted(unknown)}")
        cfg.update(user)
    for k in DEFAULTS:  # command-line flags override the config file
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v
    if cfg["mode"] not in ("cut", "mute"):
        die("mode must be cut or mute")

    src = Path(args.input).expanduser()
    files = sorted(src.rglob("*.wav")) if src.is_dir() else [src]
    files = [f for f in files if f.exists()]
    if not files:
        die(f"no .wav files found at {src}")
    out_root = Path(args.out).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    overrides = load_overrides(cfg["overrides"])
    log(f"{len(files)} file(s); steps: VAD={'on' if cfg['vad'] else 'off'}, "
        f"diarisation={'on (' + cfg['diariser'] + ')' if cfg['diarise'] else 'off'}, mode={cfg['mode']}")

    rows = []
    for f in files:
        try:
            rows.append(process_file(f, out_root, cfg, overrides))
        except Exception as exc:
            if len(files) == 1:
                raise
            log(f"ERROR in {f.name}: {exc}")
            rows.append({"file": f.name, "flag": f"ERROR: {exc}"})
    with open(out_root / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    flagged = [r for r in rows if r.get("flag")]
    log(f"done. summary: {out_root / 'summary.csv'}" + (f" ({len(flagged)} file(s) flagged to check)" if flagged else ""))


def cmd_map_textgrid(args) -> None:
    """Map a TextGrid made on the final (cut) audio back onto the original recording's timeline."""
    with open(args.time_map, newline="") as f:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        die("time map is empty")

    def to_orig(t, is_start):
        # at a cut point, an interval START belongs to the next chunk, an END to the previous one
        for r in rows:
            if (t < r["out_end"] - 1e-6) if is_start else (t <= r["out_end"] + 1e-6):
                if t < r["out_start"]:      # inside an inserted join gap
                    return r["orig_start"]
                return r["orig_start"] + (t - r["out_start"])
        return rows[-1]["orig_end"]

    _, tiers = read_textgrid(Path(args.textgrid))
    dur = args.original_duration or rows[-1]["orig_end"]
    mapped = [(name, [(to_orig(s, True), to_orig(e, False), t) for s, e, t in ivs if t.strip()])
              for name, ivs in tiers]
    write_textgrid(Path(args.output), dur, mapped)
    log(f"wrote {args.output} (original timeline)")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="preprocess one wav or a folder of wavs")
    r.add_argument("input", help="a .wav file or a folder (searched recursively)")
    r.add_argument("--out", default="preprocessed", help="output folder (default: preprocessed/)")
    r.add_argument("--config", help="JSON file with any of the settings (flags override it)")
    g = r.add_argument_group("steps")
    g.add_argument("--vad", dest="vad", action="store_true", default=None, help="step 1 on (default)")
    g.add_argument("--no-vad", dest="vad", action="store_false", help="step 1 off")
    g.add_argument("--diarise", dest="diarise", action="store_true", default=None, help="step 2 on (default)")
    g.add_argument("--no-diarise", dest="diarise", action="store_false", help="step 2 off")
    g = r.add_argument_group("output")
    g.add_argument("--mode", choices=["cut", "mute"], help="cut (default) or mute removed audio")
    g.add_argument("--fade-ms", dest="fade_ms", type=float)
    g.add_argument("--join-gap-ms", dest="join_gap_ms", type=float, help="cut mode: silence between chunks")
    g.add_argument("--min-keep-ms", dest="min_keep_ms", type=float)
    g = r.add_argument_group("VAD")
    g.add_argument("--vad-threshold", dest="vad_threshold", type=float)
    g.add_argument("--min-speech-ms", dest="min_speech_ms", type=int)
    g.add_argument("--min-silence-ms", dest="min_silence_ms", type=int)
    g.add_argument("--speech-pad-ms", dest="speech_pad_ms", type=int)
    g = r.add_argument_group("diarisation")
    g.add_argument("--diariser", choices=["pyannote", "simple"])
    g.add_argument("--pyannote-model", dest="pyannote_model")
    g.add_argument("--hf-token", dest="hf_token", help="HuggingFace token (or set HF_TOKEN)")
    g.add_argument("--num-speakers", dest="num_speakers", type=int, help="default 2; 0 = automatic")
    g.add_argument("--keep-overlap", dest="keep_overlap", action="store_true", default=None)
    g.add_argument("--child-rule", dest="child_rule", choices=["auto", "pitch", "talktime"])
    g.add_argument("--pitch-margin", dest="pitch_margin", type=float)
    g.add_argument("--overrides", help="CSV with columns file,child_speaker")
    g.add_argument("--no-reuse", dest="reuse_diarisation", action="store_false", default=None,
                   help="always re-run diarisation instead of using the cache")
    r.set_defaults(func=cmd_run)

    m = sub.add_parser("map-textgrid", help="map a TextGrid made on cut audio back to the original timeline")
    m.add_argument("textgrid")
    m.add_argument("time_map", help="03_final/time_map.csv of that file")
    m.add_argument("output")
    m.add_argument("--original-duration", type=float, help="duration of the original wav (s)")
    m.set_defaults(func=cmd_map_textgrid)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
