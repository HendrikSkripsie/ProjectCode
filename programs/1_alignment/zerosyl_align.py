#!/usr/bin/env python3
"""
zerosyl_align.py - forced alignment pipeline for the Nexdata British English
child reading-speech corpus (Conrad's data), producing Praat TextGrids with
three tiers:

    1. words      - word boundaries         (from Montreal Forced Aligner)
    2. phones     - phone boundaries        (from Montreal Forced Aligner)
    3. syllables  - syllable boundaries     (derived from the phones tier)

The syllable tier is built by grouping each word's aligned phones into
syllables (one vowel / syllabic consonant per syllable, consonants between
vowels split by the Maximal Onset Principle). Syllables never cross word
boundaries, so every syllable boundary is also an MFA phone boundary.

Models (all from the MFA model repository, all share one IPA phone set):
    acoustic model : english_mfa       (trained on US + UK + other varieties)
    dictionary     : english_uk_mfa    (British English pronunciations)
    G2P model      : english_uk_mfa    (British English, for missing words)

Sub-commands
------------
    setup      download the three MFA models (run once)
    lexicon    build the combined lexicon (UK dict + dict.local + G2P for OOVs)
    one        align ONE wav file  -> one TextGrid
    batch      align the whole corpus (or chosen sessions) -> one TextGrid per wav
    syllabify  add the syllable tier to existing MFA TextGrids (no MFA needed)

Run `python zerosyl_align.py <command> -h` for the options of each command.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from collections import OrderedDict
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ACOUSTIC_MODEL = "english_mfa"
DICTIONARY_MODEL = "english_uk_mfa"
G2P_MODEL = "english_uk_mfa"

DEFAULT_WORK_DIR = Path("mfa_work")
COMBINED_LEXICON_NAME = "english_uk_children.dict"

# Wider beams than MFA's defaults (10 / 40): child speech is acoustically far
# from the adult training data, and narrow beams make MFA drop utterances.
DEFAULT_BEAM = 100
DEFAULT_RETRY_BEAM = 400

SILENCE_LABELS = {"", "sil", "sp", "<eps>", "<sil>", "silence"}

# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def log(msg: str) -> None:
    print(f"[zerosyl_align] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[zerosyl_align] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def require_mfa() -> str:
    exe = shutil.which("mfa")
    if exe is None:
        die(
            "The `mfa` command was not found. Activate the conda environment first:\n"
            "    conda activate aligner\n"
            "(see setup_mfa.sh / README.md for how to create it)"
        )
    return exe


def run(cmd: list[str]) -> None:
    log("running: " + " ".join(str(c) for c in cmd))
    res = subprocess.run([str(c) for c in cmd])
    if res.returncode != 0:
        die(f"command failed with exit code {res.returncode}: {cmd[0]} {cmd[1]}")


def mfa_root() -> Path:
    return Path(os.environ.get("MFA_ROOT_DIR", Path.home() / "Documents" / "MFA"))


def find_pretrained_dictionary(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.exists():
            die(f"--base-dict {p} does not exist")
        return p
    p = mfa_root() / "pretrained_models" / "dictionary" / f"{DICTIONARY_MODEL}.dict"
    if p.exists():
        return p
    hits = sorted(mfa_root().glob(f"**/{DICTIONARY_MODEL}*.dict"))
    if hits:
        return hits[0]
    die(
        f"Could not find the {DICTIONARY_MODEL} dictionary under {mfa_root()}.\n"
        "Run `python zerosyl_align.py setup` first (or pass --base-dict PATH)."
    )
    raise AssertionError  # unreachable


# --------------------------------------------------------------------------- #
# Manifest + transcript normalisation
# --------------------------------------------------------------------------- #


def load_manifest(path: str | Path) -> "OrderedDict[str, dict]":
    """Return {utterance_id (filename without .wav): row dict}."""
    path = Path(path).expanduser()
    if not path.exists():
        die(f"manifest not found: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            import pandas as pd
        except ImportError:
            die("Reading .xlsx needs pandas + openpyxl; use manifest.csv instead.")
        rows = pd.read_excel(path, dtype=str).fillna("").to_dict("records")
    else:
        rows = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                with open(path, newline="", encoding=enc) as f:
                    rows = list(csv.DictReader(f))
                break
            except UnicodeDecodeError:
                continue
        if rows is None:
            die(f"could not decode {path}")
    manifest: "OrderedDict[str, dict]" = OrderedDict()
    for r in rows:
        r = {str(k).strip().lower(): ("" if v is None else str(v).strip()) for k, v in r.items()}
        if not r.get("filename"):
            continue
        utt = Path(r["filename"]).stem
        if not r.get("session"):
            r["session"] = utt.split("S")[0] if "S" in utt else ""
        manifest[utt] = r
    if not manifest:
        die(f"no rows with a 'filename' column found in {path}")
    return manifest


def tokenize(text: str) -> list[tuple[str, bool]]:
    """
    Turn a manifest transcription into aligner tokens.

    Returns [(lower-case word, looks_like_acronym), ...]
      - punctuation removed, hyphens split words ("Pop-up" -> pop up)
      - apostrophes inside words kept ("that's", "mum's")
      - ALL-CAPS tokens are flagged so that unknown ones (KFC, PTA, GP...)
        can be spelled out letter by letter in the lexicon
    """
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = re.sub(r"[-–—_/+&]", " ", text)
    out = []
    for raw in re.findall(r"[A-Za-z0-9']+", text):
        w = raw.strip("'")
        if not w:
            continue
        acronym = w.isalpha() and w.isupper() and len(w) >= 2
        out.append((w.lower(), acronym))
    return out


def lab_text(transcription: str) -> str:
    return " ".join(w for w, _ in tokenize(transcription))


# --------------------------------------------------------------------------- #
# Lexicon
# --------------------------------------------------------------------------- #

_PROB = re.compile(r"^(\d+\.\d+|1|0)$")

# British letter names in the MFA phone set (used to spell out acronyms).
LETTER_NAMES = {
    "a": "ej", "b": "bʲ iː", "c": "s iː", "d": "dʲ iː", "e": "iː", "f": "ɛ f",
    "g": "dʒ iː", "h": "ej tʃ", "i": "aj", "j": "dʒ ej", "k": "cʰ ej",
    "l": "ɛ ɫ", "m": "ɛ m", "n": "ɛ n", "o": "əw", "p": "pʰ iː", "q": "cʰ ʉː",
    "r": "ɑː", "s": "ɛ s", "t": "tʲ iː", "u": "j ʉː", "v": "vʲ iː",
    "w": "d ɐ b ə ɫ j ʉː", "x": "ɛ k s", "y": "w aj", "z": "z ɛ d",
}


def parse_lexicon_line(line: str) -> tuple[str, list[str]] | None:
    parts = line.strip().split()
    if len(parts) < 2:
        return None
    word = unicodedata.normalize("NFKC", parts[0]).lower()
    rest = parts[1:]
    # MFA dictionaries may carry up to 4 probability columns before the phones
    n = 0
    while rest and n < 4 and _PROB.match(rest[0]):
        rest = rest[1:]
        n += 1
    if not rest:
        return None
    return word, rest


def read_lexicon(path: Path) -> dict[str, list[list[str]]]:
    lex: dict[str, list[list[str]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parsed = parse_lexicon_line(line)
            if parsed:
                lex.setdefault(parsed[0], []).append(parsed[1])
    return lex


def collect_words(manifest) -> "OrderedDict[str, bool]":
    words: "OrderedDict[str, bool]" = OrderedDict()
    for row in manifest.values():
        for w, acr in tokenize(row.get("transcription", "")):
            words[w] = words.get(w, False) or acr
    return words


def build_lexicon(manifest, args) -> Path:
    """
    Combined lexicon = english_uk_mfa
                       + dict.local (overrides the UK dict for the same word)
                       + spelled-out acronyms
                       + G2P (english_uk_mfa) pronunciations for remaining OOVs
    """
    work = Path(args.work_dir)
    lex_dir = work / "lexicon"
    lex_dir.mkdir(parents=True, exist_ok=True)
    combined = lex_dir / COMBINED_LEXICON_NAME

    base_path = find_pretrained_dictionary(getattr(args, "base_dict", None))
    dict_local = Path(args.dict_local).expanduser() if args.dict_local else None
    if dict_local and not dict_local.exists():
        log(f"dict.local {dict_local} not found - continuing without it")
        dict_local = None

    words = collect_words(manifest)

    # Reuse an existing combined lexicon if it already covers every word
    if combined.exists() and not args.rebuild_lexicon:
        stale = dict_local is not None and dict_local.stat().st_mtime > combined.stat().st_mtime
        if not stale:
            have = read_lexicon(combined)
            if all(w in have for w in words):
                log(f"using existing lexicon {combined}")
                return combined

    log(f"base dictionary: {base_path}")
    base = read_lexicon(base_path)
    phone_set = {p for prons in base.values() for pron in prons for p in pron}

    local: dict[str, list[list[str]]] = {}
    if dict_local:
        local = read_lexicon(dict_local)
        bad = {p for prons in local.values() for pron in prons for p in pron} - phone_set
        if bad:
            die(
                f"{dict_local} uses phones that are not in the {DICTIONARY_MODEL} phone set: "
                f"{' '.join(sorted(bad))}\n"
                "dict.local must use the MFA IPA phone set (e.g. 'tʰ ɛ ɫ'), not ARPAbet."
            )
        log(f"dict.local: {len(local)} words")

    oov = [w for w in words if w not in base and w not in local]
    acronyms: dict[str, list[str]] = {}
    for w in oov:
        if words[w] and all(c in LETTER_NAMES for c in w):
            acronyms[w] = " ".join(LETTER_NAMES[c] for c in w).split()
    g2p_words = [w for w in oov if w not in acronyms]
    log(
        f"{len(words)} distinct words in manifest; {len(oov)} not in {DICTIONARY_MODEL} "
        f"({len(acronyms)} acronyms spelled out, {len(g2p_words)} sent to G2P)"
    )

    g2p: dict[str, list[list[str]]] = {}
    if g2p_words:
        require_mfa()
        oov_file = lex_dir / "oov_words.txt"
        g2p_out = lex_dir / "g2p_output.txt"
        oov_file.write_text("\n".join(g2p_words) + "\n", encoding="utf-8")
        run(["mfa", "g2p", oov_file, G2P_MODEL, g2p_out, "--clean",
             "--num_pronunciations", "1"])
        g2p = read_lexicon(g2p_out)
        missing = [w for w in g2p_words if w not in g2p]
        if missing:
            log("WARNING: G2P produced nothing for: " + ", ".join(missing))

    # Human-reviewable list of every generated pronunciation
    review = lex_dir / "generated_pronunciations.txt"
    with open(review, "w", encoding="utf-8") as f:
        f.write("# Pronunciations NOT from the english_uk_mfa dictionary.\n"
                "# Check them; copy corrected lines into your dict.local and re-run with\n"
                "# --rebuild-lexicon. Phones must come from the MFA English phone set.\n")
        for w, pron in acronyms.items():
            f.write(f"{w}\t{' '.join(pron)}\t# acronym\n")
        for w, prons in g2p.items():
            for pron in prons:
                f.write(f"{w}\t{' '.join(pron)}\t# g2p\n")

    with open(combined, "w", encoding="utf-8") as out:
        with open(base_path, encoding="utf-8") as f:
            for line in f:
                parsed = parse_lexicon_line(line)
                if parsed and parsed[0] in local:
                    continue  # dict.local overrides
                out.write(line if line.endswith("\n") else line + "\n")
        for w, prons in local.items():
            for pron in prons:
                out.write(f"{w}\t{' '.join(pron)}\n")
        for w, pron in acronyms.items():
            out.write(f"{w}\t{' '.join(pron)}\n")
        for w, prons in g2p.items():
            for pron in prons:
                out.write(f"{w}\t{' '.join(pron)}\n")
    log(f"wrote combined lexicon {combined}")
    log(f"review generated pronunciations in {review}")
    return combined


# --------------------------------------------------------------------------- #
# Syllabification
# --------------------------------------------------------------------------- #

VOWEL_CHARS = set("aeiouyæøœɐɑɒɔəɘɚɛɜɝɞɤɨɪʉʊʌʏɵɯ")
SYLLABIC_MARK = "̩"  # combining vertical line below: n̩ ɫ̩ m̩
MODIFIERS = re.compile("[ʰʲʷʼˠˤ̪̩̥̊ːː]")
BASE_MAP = {"ɫ": "l", "ʎ": "l", "ɲ": "n", "c": "k", "ɟ": "ɡ", "g": "ɡ",
            "ç": "h", "r": "ɹ", "ɾ": "t", "ʔ": "t"}

_ONSETS_2 = """
p l, p ɹ, p j, b l, b ɹ, b j, t ɹ, t j, t w, d ɹ, d j, d w,
k l, k ɹ, k j, k w, ɡ l, ɡ ɹ, ɡ j, ɡ w, f l, f ɹ, f j, v j, θ ɹ, θ w, θ j,
ʃ ɹ, s p, s t, s k, s m, s n, s l, s w, s f, s j, m j, n j, h j, l j,
tʃ ɹ, dʒ ɹ, s tʃ, ʃ tʃ
"""
_ONSETS_3 = """
s p l, s p ɹ, s p j, s t ɹ, s t j, s k l, s k ɹ, s k w, s k j,
s tʃ ɹ, ʃ tʃ ɹ, ʃ t ɹ
"""
LEGAL_ONSETS = {tuple(o.split()) for o in (_ONSETS_2 + "," + _ONSETS_3).replace("\n", " ").split(",") if o.strip()}


def is_nucleus(phone: str) -> bool:
    if not phone:
        return False
    return phone[0] in VOWEL_CHARS or SYLLABIC_MARK in phone


def onset_base(phone: str) -> str:
    b = MODIFIERS.sub("", phone)
    return BASE_MAP.get(b, b)


def is_legal_onset(cluster: list[str]) -> bool:
    if not cluster:
        return True
    bases = tuple(onset_base(p) for p in cluster)
    if len(bases) == 1:
        return bases[0] != "ŋ"
    return bases in LEGAL_ONSETS


def syllabify(phones: list[tuple[float, float, str]]) -> list[list[tuple[float, float, str]]]:
    """Group one word's phones into syllables (Maximal Onset Principle)."""
    if not phones:
        return []
    nuclei = [i for i, p in enumerate(phones) if is_nucleus(p[2])]
    if len(nuclei) <= 1:
        return [phones]
    starts = [0]
    for a, b in zip(nuclei, nuclei[1:]):
        cluster = [p[2] for p in phones[a + 1:b]]
        for j in range(len(cluster) + 1):  # j=0 -> longest possible onset
            if is_legal_onset(cluster[j:]):
                starts.append(a + 1 + j)
                break
    starts.append(len(phones))
    return [phones[s:e] for s, e in zip(starts, starts[1:]) if e > s]


# --------------------------------------------------------------------------- #
# TextGrid I/O
# --------------------------------------------------------------------------- #

Interval = tuple[float, float, str]


def read_textgrid(path: Path) -> tuple[float, float, "OrderedDict[str, list[Interval]]"]:
    """Minimal reader for long-format ("ooTextFile") interval-tier TextGrids."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("latin-1")
    lines = text.splitlines()
    tiers: "OrderedDict[str, list[Interval]]" = OrderedDict()
    gxmin = gxmax = None
    tier = None
    cur: dict = {}
    in_interval = False
    num = re.compile(r"^\s*(xmin|xmax)\s*=\s*([-0-9.eE+]+)")
    txt = re.compile(r'^\s*text\s*=\s*"(.*)"\s*$')
    name_re = re.compile(r'^\s*name\s*=\s*"(.*)"\s*$')
    for line in lines:
        if (m := name_re.match(line)):
            tier = m.group(1).replace('""', '"')
            tiers[tier] = []
            in_interval = False
            continue
        if re.match(r"^\s*intervals\s*\[\d+\]\s*:", line):
            in_interval = True
            cur = {}
            continue
        if (m := num.match(line)):
            val = float(m.group(2))
            if tier is None:
                if m.group(1) == "xmin" and gxmin is None:
                    gxmin = val
                elif m.group(1) == "xmax" and gxmax is None:
                    gxmax = val
            elif in_interval:
                cur[m.group(1)] = val
            continue
        if in_interval and (m := txt.match(line)):
            tiers[tier].append((cur["xmin"], cur["xmax"], m.group(1).replace('""', '"')))
            in_interval = False
    if gxmax is None:
        raise ValueError(f"{path} does not look like a long-format TextGrid")
    return gxmin or 0.0, gxmax, tiers


def _fmt(x: float) -> str:
    s = f"{x:.6f}".rstrip("0")
    return s + "0" if s.endswith(".") else s


def write_textgrid(path: Path, xmax: float, tiers: list[tuple[str, list[Interval]]]) -> None:
    """Write a long-format TextGrid laid out like the reference file."""
    L = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
         "xmin = 0 ", f"xmax = {_fmt(xmax)} ", "tiers? <exists> ",
         f"size = {len(tiers)} ", "item []: "]
    for i, (name, ivs) in enumerate(tiers, 1):
        L += [f"    item [{i}]:",
              '        class = "IntervalTier" ',
              f'        name = "{name}" ',
              "        xmin = 0 ",
              f"        xmax = {_fmt(xmax)} ",
              f"        intervals: size = {len(ivs)} "]
        for j, (s, e, t) in enumerate(ivs, 1):
            t = t.replace('"', '""')
            L += [f"        intervals [{j}]:",
                  f"            xmin = {_fmt(s)} ",
                  f"            xmax = {_fmt(e)} ",
                  f'            text = "{t}" ']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def clean_tier(ivs: list[Interval], xmax: float, eps: float = 1e-6) -> list[Interval]:
    """Normalise silence labels to "", merge adjacent silences, fill gaps, cover [0, xmax]."""
    items = sorted(
        ((s, e, "" if t.strip().lower() in SILENCE_LABELS else t.strip()) for s, e, t in ivs if e - s > eps),
        key=lambda x: x[0],
    )
    out: list[Interval] = []
    t = 0.0
    for s, e, lab in items:
        if s < t - eps:  # overlap - clip
            s = t
            if e - s <= eps:
                continue
        if s > t + eps:
            out.append((t, s, ""))
        elif out:
            s = t  # snap tiny gaps
        else:
            s = 0.0 if s < eps else s
        out.append((s, e, lab))
        t = e
    if xmax > t + eps:
        out.append((t, xmax, ""))
    elif out:
        out[-1] = (out[-1][0], xmax, out[-1][2])
    merged: list[Interval] = []
    for iv in out:
        if merged and iv[2] == "" and merged[-1][2] == "":
            merged[-1] = (merged[-1][0], iv[1], "")
        else:
            merged.append(iv)
    return merged or [(0.0, xmax, "")]


def find_tier(tiers, kind: str) -> list[Interval]:
    for name, ivs in tiers.items():
        n = name.strip().lower()
        if n == kind or n.endswith(f"- {kind}") or n.endswith(f"-{kind}"):
            return ivs
    for name, ivs in tiers.items():
        if kind.rstrip("s") in name.lower():
            return ivs
    raise KeyError(f"no '{kind}' tier (tiers: {list(tiers)})")


def build_three_tiers(mfa_textgrid: Path) -> tuple[float, list[Interval], list[Interval], list[Interval], dict]:
    _, xmax, tiers = read_textgrid(mfa_textgrid)
    words = clean_tier(find_tier(tiers, "words"), xmax)
    phones = clean_tier(find_tier(tiers, "phones"), xmax)

    syl: list[Interval] = []
    stats = {"words": 0, "phones": 0, "syllables": 0, "spn_words": 0}
    real_phones = [p for p in phones if p[2]]
    stats["phones"] = len(real_phones)
    for ws, we, wl in words:
        if not wl:
            continue
        stats["words"] += 1
        wp = [p for p in real_phones if ws - 1e-6 <= (p[0] + p[1]) / 2 <= we + 1e-6]
        if not wp or any(p[2] == "spn" for p in wp):
            syl.append((ws, we, "spn"))
            stats["spn_words"] += 1
            stats["syllables"] += 1
            continue
        groups = syllabify(wp)
        for k, g in enumerate(groups):
            s = ws if k == 0 else g[0][0]
            e = we if k == len(groups) - 1 else groups[k + 1][0][0]
            syl.append((s, e, "".join(p[2] for p in g)))
            stats["syllables"] += 1
    syllables = clean_tier(syl, xmax)
    return xmax, words, phones, syllables, stats


def convert_textgrid(mfa_tg: Path, out_tg: Path) -> dict:
    xmax, words, phones, syllables, stats = build_three_tiers(mfa_tg)
    write_textgrid(out_tg, xmax, [("words", words), ("phones", phones), ("syllables", syllables)])
    stats["first_word_start"] = next((w[0] for w in words if w[2]), None)
    stats["last_word_end"] = next((w[1] for w in reversed(words) if w[2]), None)
    stats["duration"] = xmax
    return stats


# --------------------------------------------------------------------------- #
# QC summary
# --------------------------------------------------------------------------- #

QC_FIELDS = ["utterance", "session", "status", "transcript", "expected_words", "aligned_words",
             "phones", "syllables", "spn_words", "manifest_start_s", "aligned_start_s",
             "manifest_end_s", "aligned_end_s", "flag"]


def qc_row(utt: str, row: dict, stats: dict | None, tolerance: float) -> dict:
    expected = len(tokenize(row.get("transcription", "")))
    r = {"utterance": utt, "session": row.get("session", ""),
         "transcript": row.get("transcription", ""), "expected_words": expected}
    if stats is None:
        r.update(status="NOT_ALIGNED", flag="MFA produced no alignment (try larger beams)")
        return r

    def ms(key):
        try:
            return float(row.get(key, "")) / 1000.0
        except ValueError:
            return None

    m_start, m_end = ms("start_ms"), ms("end_ms")
    flags = []
    if stats["words"] != expected:
        flags.append("word count mismatch")
    if stats["spn_words"]:
        flags.append("spoken-noise (OOV) words")
    if m_start is not None and stats["first_word_start"] is not None and abs(stats["first_word_start"] - m_start) > tolerance:
        flags.append("start differs from manifest")
    if m_end is not None and stats["last_word_end"] is not None and abs(stats["last_word_end"] - m_end) > tolerance:
        flags.append("end differs from manifest")
    r.update(status="OK" if not flags else "CHECK", aligned_words=stats["words"],
             phones=stats["phones"], syllables=stats["syllables"], spn_words=stats["spn_words"],
             manifest_start_s=m_start, aligned_start_s=stats["first_word_start"],
             manifest_end_s=m_end, aligned_end_s=stats["last_word_end"], flag="; ".join(flags))
    return r


def write_qc(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=QC_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else (round(r[k], 3) if isinstance(r.get(k), float) else r[k])) for k in QC_FIELDS})
    n_ok = sum(r["status"] == "OK" for r in rows)
    log(f"QC report: {path}  ({n_ok}/{len(rows)} OK, {len(rows) - n_ok} to check)")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_setup(args) -> None:
    require_mfa()
    run(["mfa", "model", "download", "acoustic", ACOUSTIC_MODEL])
    run(["mfa", "model", "download", "dictionary", DICTIONARY_MODEL])
    run(["mfa", "model", "download", "g2p", G2P_MODEL])
    log("models downloaded to " + str(mfa_root() / "pretrained_models"))


def cmd_lexicon(args) -> None:
    build_lexicon(load_manifest(args.manifest), args)


def mfa_align_options(args) -> list[str]:
    opts = ["--clean", "--beam", str(args.beam), "--retry_beam", str(args.retry_beam),
            "--output_format", "long_textgrid"]
    return opts


def cmd_one(args) -> None:
    require_mfa()
    wav = Path(args.wav).expanduser().resolve()
    if not wav.exists():
        die(f"{wav} not found")
    manifest = load_manifest(args.manifest)
    utt = wav.stem
    if args.text:
        row = {"filename": wav.name, "transcription": args.text, "session": ""}
    elif utt in manifest:
        row = manifest[utt]
    else:
        die(f"{wav.name} is not in the manifest (pass --text \"...\" to give the transcript)")
    log(f"{utt}: \"{row['transcription']}\"  ->  \"{lab_text(row['transcription'])}\"")

    lexicon = build_lexicon(manifest, args)
    if args.text:  # make sure the custom text is covered too
        have = read_lexicon(lexicon)
        if any(w not in have for w, _ in tokenize(args.text)):
            args.rebuild_lexicon = True
            combined = OrderedDict(manifest)
            combined[f"__custom__{utt}"] = row
            lexicon = build_lexicon(combined, args)

    tmp = Path(args.work_dir) / "single" / utt
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    lab = tmp / f"{utt}.lab"
    lab.write_text(lab_text(row["transcription"]) + "\n", encoding="utf-8")
    mfa_tg = tmp / f"{utt}.mfa.TextGrid"

    model = args.acoustic_model or ACOUSTIC_MODEL
    run(["mfa", "align_one", wav, lab, lexicon, model, mfa_tg] + mfa_align_options(args))
    if not mfa_tg.exists():
        die("MFA did not write a TextGrid - alignment failed (try --beam 200 --retry_beam 800)")

    out = Path(args.out).expanduser()
    out_tg = out / f"{utt}.TextGrid" if (out.is_dir() or out.suffix == "") else out
    stats = convert_textgrid(mfa_tg, out_tg)
    r = qc_row(utt, row, stats, args.tolerance)
    log(f"wrote {out_tg}: {stats['words']} words, {stats['phones']} phones, {stats['syllables']} syllables")
    if r["status"] != "OK":
        log("QC: " + r["flag"])


def select_utterances(manifest, args):
    sessions = set(args.sessions) if args.sessions else None
    items = [(u, r) for u, r in manifest.items() if sessions is None or r.get("session") in sessions]
    if args.limit:
        items = items[: args.limit]
    return items


def cmd_batch(args) -> None:
    require_mfa()
    manifest = load_manifest(args.manifest)
    audio_root = Path(args.audio_root).expanduser().resolve()
    work = Path(args.work_dir).resolve()
    out_dir = Path(args.out).expanduser().resolve()
    items = select_utterances(manifest, args)
    if not items:
        die("no utterances selected")

    # 1. corpus folder: <session>/<utt>.wav (symlink) + <utt>.lab  -> MFA treats
    #    each session (= one child) as a speaker and adapts to it (fMLLR)
    corpus = work / "corpus"
    if corpus.exists():
        shutil.rmtree(corpus)
    missing = []
    for utt, row in items:
        session = row.get("session") or utt[:5]
        wav = audio_root / session / row["filename"]
        if not wav.exists():
            alt = list(audio_root.glob(f"**/{row['filename']}"))
            if not alt:
                missing.append(utt)
                continue
            wav = alt[0]
        d = corpus / session
        d.mkdir(parents=True, exist_ok=True)
        dst = d / wav.name
        if args.copy_audio:
            shutil.copy2(wav, dst)
        else:
            try:
                dst.symlink_to(wav)
            except OSError:
                shutil.copy2(wav, dst)
        (d / f"{utt}.lab").write_text(lab_text(row["transcription"]) + "\n", encoding="utf-8")
    if missing:
        log(f"WARNING: {len(missing)} wav files listed in the manifest were not found "
            f"(e.g. {', '.join(missing[:5])})")
    selected = [(u, r) for u, r in items if u not in set(missing)]
    log(f"corpus: {len(selected)} utterances in {len({r.get('session') for _, r in selected})} sessions")

    # 2. lexicon
    lexicon = build_lexicon(OrderedDict(selected), args)

    # 3. optional: adapt the adult acoustic model to these children's voices
    model = args.acoustic_model or ACOUSTIC_MODEL
    if args.adapt:
        adapted = work / "english_mfa_child_adapted.zip"
        run(["mfa", "adapt", corpus, lexicon, model, adapted, "--clean", "-j", str(args.jobs),
             "--beam", str(args.beam), "--retry_beam", str(args.retry_beam)])
        model = str(adapted)

    # 4. align
    mfa_out = work / "mfa_output"
    if mfa_out.exists():
        shutil.rmtree(mfa_out)
    run(["mfa", "align", corpus, lexicon, model, mfa_out, "-j", str(args.jobs)] + mfa_align_options(args))

    # 5. words + phones + syllables TextGrids, QC report
    qc = []
    for utt, row in selected:
        session = row.get("session") or utt[:5]
        hits = [mfa_out / session / f"{utt}.TextGrid"]
        mfa_tg = next((h for h in hits if h.exists()), None)
        if mfa_tg is None:
            found = list(mfa_out.glob(f"**/{utt}.TextGrid"))
            mfa_tg = found[0] if found else None
        stats = None
        if mfa_tg is not None:
            try:
                dest = out_dir / session / f"{utt}.TextGrid" if not args.flat else out_dir / f"{utt}.TextGrid"
                stats = convert_textgrid(mfa_tg, dest)
            except Exception as exc:  # keep going, report in QC
                log(f"WARNING: could not convert {mfa_tg}: {exc}")
        qc.append(qc_row(utt, row, stats, args.tolerance))
    write_qc(out_dir / "alignment_qc.csv", qc)
    log(f"TextGrids written to {out_dir}")


def cmd_syllabify(args) -> None:
    src = Path(args.input).expanduser()
    dst = Path(args.output).expanduser()
    files = sorted(src.rglob("*.TextGrid")) if src.is_dir() else [src]
    if not files:
        die(f"no TextGrids found in {src}")
    for f in files:
        out = (dst / f.relative_to(src)) if src.is_dir() else (dst / f.name if dst.is_dir() or dst.suffix == "" else dst)
        stats = convert_textgrid(f, out)
        log(f"{out}: {stats['words']} words, {stats['phones']} phones, {stats['syllables']} syllables")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def lexicon_opts(sp):
        sp.add_argument("--manifest", required=True, help="manifest.csv (or manifest.xlsx)")
        sp.add_argument("--dict-local", default="dict.local",
                        help="extra/override pronunciations in MFA IPA phones (default: ./dict.local if present)")
        sp.add_argument("--base-dict", help=f"path to {DICTIONARY_MODEL}.dict (found automatically after `setup`)")
        sp.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR), help="scratch folder (default: mfa_work)")
        sp.add_argument("--rebuild-lexicon", action="store_true", help="force rebuilding the combined lexicon")

    def align_opts(sp):
        sp.add_argument("--beam", type=int, default=DEFAULT_BEAM)
        sp.add_argument("--retry-beam", dest="retry_beam", type=int, default=DEFAULT_RETRY_BEAM)
        sp.add_argument("--acoustic-model", help=f"acoustic model name or .zip (default: {ACOUSTIC_MODEL})")
        sp.add_argument("--tolerance", type=float, default=0.3,
                        help="QC: max allowed difference (s) from manifest start_ms/end_ms (default 0.3)")

    sp = sub.add_parser("setup", help="download the MFA models (once)")
    sp.set_defaults(func=cmd_setup)

    sp = sub.add_parser("lexicon", help="build the combined lexicon only")
    lexicon_opts(sp)
    sp.set_defaults(func=cmd_lexicon)

    sp = sub.add_parser("one", help="align a single wav file")
    sp.add_argument("wav", help="path to e.g. 16kHz_Mono/G0001/G0001S0001.wav")
    sp.add_argument("--out", default="textgrids", help="output folder or .TextGrid path (default: textgrids/)")
    sp.add_argument("--text", help="transcript to use instead of looking it up in the manifest")
    lexicon_opts(sp)
    align_opts(sp)
    sp.set_defaults(func=cmd_one)

    sp = sub.add_parser("batch", help="align many files (whole corpus or chosen sessions)")
    sp.add_argument("--audio-root", required=True, help="the 16kHz_Mono folder")
    sp.add_argument("--out", default="textgrids", help="output folder (default: textgrids/)")
    sp.add_argument("--sessions", nargs="+", help="only these sessions, e.g. G0001 G0002")
    sp.add_argument("--limit", type=int, help="only the first N utterances (for testing)")
    sp.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    sp.add_argument("--adapt", action="store_true",
                    help="first adapt english_mfa to this child corpus (slower, usually more accurate)")
    sp.add_argument("--flat", action="store_true", help="write all TextGrids into one folder")
    sp.add_argument("--copy-audio", action="store_true", help="copy wavs instead of symlinking")
    lexicon_opts(sp)
    align_opts(sp)
    sp.set_defaults(func=cmd_batch)

    sp = sub.add_parser("syllabify", help="add a syllable tier to existing MFA TextGrid(s)")
    sp.add_argument("input", help="MFA TextGrid file or folder")
    sp.add_argument("output", help="output file or folder")
    sp.set_defaults(func=cmd_syllabify)

    args = p.parse_args(argv)
    if getattr(args, "dict_local", None) == "dict.local" and not Path("dict.local").exists():
        args.dict_local = None
    args.func(args)


if __name__ == "__main__":
    main()
