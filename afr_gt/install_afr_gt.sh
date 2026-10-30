#!/usr/bin/env bash
# =============================================================================
# install_afr_gt.sh  -  one-file installer for the Afrikaans ground-truth
# alignment pipeline (MFA + Dutch CV model + rule-based Afrikaans syllables).
#
#   bash ~/Downloads/install_afr_gt.sh
#
# What it does (safe to run again; it skips what's already there):
#   1. writes the pipeline files to ~/Skripsie/code/afr_gt
#      (afr_g2p.py, afr_gt_align.py, afr_add_syllables.py, dict.local, README.md)
#   2. installs Miniforge (conda) in ~/miniforge3 if you have no conda
#   3. creates the conda env "aligner" with Montreal Forced Aligner
#   4. downloads MFA's Dutch acoustic model to ~/Skripsie/code/afr_gt/models/dutch_cv.zip
#   5. tests the installation
# =============================================================================
set -euo pipefail

INSTALL_DIR="${AFR_GT_DIR:-$HOME/Skripsie/code/afr_gt}"
ENV_NAME="aligner"
MODEL_URL="https://github.com/MontrealCorpusTools/mfa-models/releases/download/acoustic-dutch_cv-v2.0.0/dutch_cv.zip"

download() {  # download URL FILE  (curl or wget)
    if command -v curl >/dev/null 2>&1; then curl -L --fail -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then wget -O "$2" "$1"
    else echo "ERROR: need curl or wget:  sudo apt install curl"; exit 1; fi
}

echo "==> 1/5  Writing pipeline files to $INSTALL_DIR"
mkdir -p "$INSTALL_DIR/models"
cd "$INSTALL_DIR"
cat > afr_g2p.py <<'__AFR_GT_EOF__'
#!/usr/bin/env python3
"""
Rule-based Afrikaans grapheme-to-phoneme conversion and syllabification.

Pronunciations use the phone set of MFA's Dutch Common Voice acoustic model
(`dutch_cv`, Epitran phones), because that is the acoustic model used for the
alignment. Each word comes out as a list of syllables, each syllable a list of
phones, e.g.

    babaloon -> [['b', 'aː'], ['b', 'aː'], ['l', 'oː', 'n']]

Syllables are decided here, from the spelling, before alignment. After MFA has
aligned the phones, the syllable tier is made by grouping the aligned phones
with these same syllable sizes, so every syllable boundary is a phone boundary.

Syllabification rules (Afrikaans):
  * every vowel / diphthong is one nucleus ("aai", "ooi", "oei", "ui", "ou",
    "ei", "y", "ie", "oe", "ee", "aa", "oo", "uu", "eu", "a", "e", ...)
  * a diaeresis (geëet, oë) starts a new syllable: ge.eet
  * separable / compound prefixes (op-, uit-, af-, oor-, aan-, weg-, in-,
    ver-, daar-, hier-, ...) end in a syllable boundary: op.eet, weg.spring
  * consonants between two nuclei: the longest legal Afrikaans onset goes to
    the next syllable (Maximal Onset), but
      - s + consonant clusters are not used as onsets inside a word
        (kas.te, vens.ters, vis.lyn), as in Afrikaans hyphenation
      - ŋ is never an onset (hon.ger -> hɔŋ.ɛr)
  * a word without any vowel (fragments such as "s", "g", "hmm") is one syllable
  * "unk" is one syllable of spoken noise (phone "spn")

Run on its own to inspect pronunciations:
    python afr_g2p.py babaloon seuntjie geëet hieroor
    python afr_g2p.py --file words.txt
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

# --------------------------------------------------------------------------
# Phone set of the dutch_cv acoustic model (MFA, Epitran phones) + spn
# --------------------------------------------------------------------------
DUTCH_CV_PHONES = set(
    "a aː b c d eː f h i iː j k l m n oː p r s t u uː v w x y yː z øː ĵ ŋ œ ɑ ɔ "
    "ɔ̈ ɛ ɛ̈ ɣ ɥ ɪ ʃ ʋ ʏ ʏ̈".split()
) | {"spn"}

VOWEL_PHONES = {"a", "aː", "eː", "i", "iː", "oː", "u", "uː", "y", "yː", "øː", "œ",
                "ɑ", "ɔ", "ɔ̈", "ɛ", "ɛ̈", "ɪ", "ʏ", "ʏ̈"}

SYL_SEP = "."  # separates syllables in lexicon files

# --------------------------------------------------------------------------
# Words that the spelling rules can't handle: English words, fillers, a few
# compounds. Syllables are separated by " . ". Phones must be dutch_cv phones.
# Extend or override with a dict.local file (same format).
# --------------------------------------------------------------------------
BUILTIN_EXCEPTIONS = """
unk         spn
uh          ɑ
um          ɑ m
hmm         h m
huh         h ɑ
mm          m
joh         j oː
ball        b ɔ l
balloon     b ɑ . l u n
baloon      b ɑ . l u n
butterfly   b ɑ . t ɛ . f l ɑ j
cause       k ɔ s
chase       t ʃ eː s
dog         d ɔ k
feel        f i l
feeling     f i . l ɪ ŋ
get         k ɛ t
good        k u t
grumpy      x r ɑ m . p i
happy       h ɛ . p i
hes         h i s
jump        c ɑ m p
gejump      x ɛ . c ɑ m p
ingejump    ɪ n . x ɛ . c ɑ m p
mad         m ɛ t
mouse       m ɑ w s
party       p aː . t i
partytjie   p aː . t i . k i
polony      p ɔ . l oː . n i
sad         s ɛ t
the         d ɛ
to          t u
tree        t r i
ouchies     ɑ w . t ʃ i s
rondom      r ɔ n t . ɔ m
n           ɛ
"""

# --------------------------------------------------------------------------
# Graphemes
# --------------------------------------------------------------------------
# vowel grapheme -> phones (None = depends on context: a, e, o, u)
VOWELS = {
    "aai": ["aː", "j"], "ooi": ["oː", "j"], "oei": ["u", "j"],
    "eeu": ["i", "w"], "ieu": ["i", "w"],
    "ai": ["ɑ", "j"], "oi": ["ɔ", "j"], "au": ["ɑ", "w"], "ou": ["ɑ", "w"],
    "ui": ["œ", "ɥ"], "ei": ["ɛ", "j"],
    "ie": ["i"], "oe": ["u"], "ee": ["eː"], "aa": ["aː"], "oo": ["oː"],
    "uu": ["y"], "eu": ["øː"],
    "ê": ["ɛ"], "ô": ["ɔ"], "î": ["ɪ"], "û": ["œ"],
    "y": ["ɛ", "j"], "i": ["ɪ"],
    "a": None, "e": None, "o": None, "u": None,
}
VOWEL_KEYS = sorted(VOWELS, key=len, reverse=True)

CONSONANTS = {
    "ng": ["ŋ"], "sj": ["ʃ"], "ch": ["ʃ"], "ck": ["k"], "qu": ["k", "ʋ"],
    "tj": ["c"], "dj": ["c"],
    "b": ["b"], "c": ["k"], "d": ["d"], "f": ["f"], "g": ["x"], "h": ["h"],
    "j": ["j"], "k": ["k"], "l": ["l"], "m": ["m"], "n": ["n"], "p": ["p"],
    "q": ["k"], "r": ["r"], "s": ["s"], "t": ["t"], "v": ["f"], "w": ["v"],
    "x": ["k", "s"], "z": ["z"],
}
CONS_KEYS = sorted(CONSONANTS, key=len, reverse=True)

# prefixes followed by a syllable boundary (checked longest first)
PREFIX_ANY = ["daar", "hier", "waar", "terug", "neer", "saam", "weg", "uit",
              "oor", "aan", "ont", "op", "af"]        # before vowel or consonant
PREFIX_CONS = ["ver", "her", "in", "om", "on", "mee",   # only before a consonant
               "ge", "be"]
SCHWA_PREFIXES = ("ge", "be", "ver", "te")              # unstressed e

# legal onsets inside a word (s-clusters excluded on purpose, see docstring)
LEGAL_ONSETS = {
    ("p", "r"), ("p", "l"), ("b", "r"), ("b", "l"), ("t", "r"), ("d", "r"),
    ("k", "r"), ("k", "l"), ("k", "n"), ("k", "ʋ"), ("f", "r"), ("f", "l"),
    ("x", "r"), ("x", "l"), ("t", "ʋ"), ("d", "ʋ"), ("ʃ", "r"),
}


def normalise(word: str) -> str:
    return unicodedata.normalize("NFC", word.strip().lower())


class Unit:
    """One grapheme unit: a vowel nucleus (V), consonant (C) or boundary (|)."""

    def __init__(self, kind, graph, phones=None, geminate=False):
        self.kind, self.graph, self.phones, self.geminate = kind, graph, phones, geminate

    def __repr__(self):
        return f"{self.kind}:{self.graph}:{self.phones}"


def _split_prefix(w: str) -> str:
    """Insert '|' after a separable/compound prefix (recursively)."""
    def has_vowel(s):
        return any(ch in "aeiouyêôîûë" for ch in s)
    for p in sorted(PREFIX_ANY + PREFIX_CONS, key=len, reverse=True):
        if w.startswith(p) and len(w) > len(p):
            rest = w[len(p):]
            if rest[0] == "|" or not has_vowel(rest) or len(rest) < 2:
                continue
            if p in PREFIX_CONS and rest[0] in "aeiouyêôîûë":
                continue
            # "ge" after a prefix is itself unstressed; keep scanning the rest
            return p + "|" + _split_prefix(rest)
    return w


def _tokenise(w: str) -> list[Unit]:
    # accents: diaeresis = new syllable; acute/grave = emphasis only
    repl = {"ë": "|e", "ï": "|i", "ü": "|u", "ö": "|o",
            "é": "e", "è": "e", "á": "a", "à": "a", "ó": "o", "ú": "u",
            "í": "i", "ý": "y"}
    w = "".join(repl.get(ch, ch) for ch in w)
    w = w.replace("'", "").replace("-", "|")
    w = "|".join(_split_prefix(part) for part in w.split("|"))
    w = w.strip("|")

    units: list[Unit] = []
    i = 0
    while i < len(w):
        if w[i] == "|":
            if units and units[-1].kind != "|":
                units.append(Unit("|", "|"))
            i += 1
            continue
        # diminutive -tjie / -djie: [ki], and a preceding n becomes ŋ
        if w.startswith(("tjie", "djie"), i):
            if units and units[-1].kind == "C" and units[-1].graph == "n":
                units[-1].phones = ["ŋ"]
            units.append(Unit("C", w[i:i + 2], ["k"]))
            i += 2
            continue
        for g in VOWEL_KEYS:
            if w.startswith(g, i):
                units.append(Unit("V", g, VOWELS[g]))
                i += len(g)
                break
        else:
            for g in CONS_KEYS:
                if w.startswith(g, i):
                    # double consonant -> one phone, closes the previous vowel
                    if units and units[-1].kind == "C" and units[-1].graph == g:
                        units[-1].geminate = True
                    else:
                        units.append(Unit("C", g, list(CONSONANTS[g])))
                    i += len(g)
                    break
            else:
                i += 1  # unknown character: skip
    return units


def _assign_vowels(units: list[Unit], word: str) -> None:
    """Decide a/e/o/u quality from open/closed syllable."""
    first_v = next((k for k, u in enumerate(units) if u.kind == "V"), None)
    for k, u in enumerate(units):
        if u.kind != "V" or u.phones is not None:
            continue
        nxt = units[k + 1] if k + 1 < len(units) else None
        nxt2 = units[k + 2] if k + 2 < len(units) else None
        if nxt is None or nxt.kind in ("V", "|"):
            is_open = True
        elif (nxt.kind == "C" and not nxt.geminate and nxt.phones != ["ŋ"]
              and nxt2 is not None and nxt2.kind == "V"):
            is_open = True
        else:
            is_open = False
        word_final = nxt is None
        if u.graph == "a":
            u.phones = ["aː"] if is_open else ["ɑ"]
        elif u.graph == "o":
            u.phones = ["oː"] if is_open else ["ɔ"]
        elif u.graph == "u":
            u.phones = ["y"] if is_open else ["œ"]
        elif u.graph == "e":
            stressed_open = (is_open and not word_final and k == first_v
                             and not word.startswith(SCHWA_PREFIXES))
            u.phones = ["eː"] if stressed_open else ["ɛ"]


def _consonant_rules(units: list[Unit]) -> None:
    for k, u in enumerate(units):
        if u.kind != "C":
            continue
        prev = units[k - 1] if k > 0 else None
        nxt = units[k + 1] if k + 1 < len(units) else None
        # n before k/x -> ŋ
        if u.phones == ["n"] and nxt is not None and nxt.kind == "C" and nxt.phones[0] in ("k",) \
                and nxt.graph not in ("tj", "dj"):
            u.phones = ["ŋ"]
        # w after a consonant (tw, kw, sw, dw) -> ʋ
        if u.graph == "w" and prev is not None and prev.kind == "C":
            u.phones = ["ʋ"]
        # final devoicing: b, d at the end of a word / before a boundary / before a
        # consonant they can't form an onset with (bl, br, dr stay voiced)
        if u.phones in (["b"], ["d"]) and (
                nxt is None or nxt.kind == "|"
                or (nxt.kind == "C" and (u.phones[0], nxt.phones[0]) not in LEGAL_ONSETS)):
            u.phones = ["p"] if u.phones == ["b"] else ["t"]
    # the same phone twice in a row inside a word (goedte: d->t + t) -> one phone
    k = 0
    while k + 1 < len(units):
        a, b = units[k], units[k + 1]
        if a.kind == "C" and b.kind == "C" and a.phones == b.phones:
            del units[k]
            continue
        k += 1
    # identical consonant on both sides of a prefix boundary (weg|gesteek) -> keep one,
    # in the onset of the next syllable
    k = 0
    while k + 2 < len(units):
        a, b, c = units[k], units[k + 1], units[k + 2]
        if a.kind == "C" and b.kind == "|" and c.kind == "C" and a.phones == c.phones:
            del units[k]
            continue
        k += 1


def _legal_onset(cons: list[list[str]]) -> bool:
    """cons: list of consonant units (each a phone list). Word-internal onset check."""
    if not cons:
        return True
    phones = [p for c in cons for p in c]
    if len(cons) == 1:
        return phones[0] != "ŋ" and len(phones) == 1
    if len(cons) == 2 and len(phones) == 2:
        return tuple(phones) in LEGAL_ONSETS
    return False


def syllabify_units(units: list[Unit]) -> list[list[str]]:
    nuclei = [k for k, u in enumerate(units) if u.kind == "V"]
    if not nuclei:
        phones = [p for u in units if u.kind == "C" for p in u.phones]
        return [phones] if phones else []
    # boundary position (unit index where a new syllable starts) between nuclei
    starts = [0]
    for a, b in zip(nuclei, nuclei[1:]):
        between = list(range(a + 1, b))
        bars = [k for k in between if units[k].kind == "|"]
        if bars:
            starts.append(bars[-1] + 1)
            continue
        cons = [units[k].phones for k in between]
        split = b  # default: no consonant goes to the onset
        for n_on in range(len(cons), 0, -1):
            if _legal_onset(cons[len(cons) - n_on:]):
                split = b - n_on
                break
        starts.append(split)
    sylls = []
    for s, e in zip(starts, starts[1:] + [len(units)]):
        ph = [p for u in units[s:e] if u.kind in ("V", "C") for p in u.phones]
        if ph:
            sylls.append(ph)
    return sylls


def rule_g2p(word: str) -> list[list[str]]:
    w = normalise(word)
    units = _tokenise(w)
    _assign_vowels(units, w.replace("|", ""))
    _consonant_rules(units)
    return syllabify_units(units)


# --------------------------------------------------------------------------
# Lexicon handling
# --------------------------------------------------------------------------
def parse_syllabified(text: str, source: str = "") -> dict[str, list[list[str]]]:
    """Parse 'word<TAB or spaces>p p . p p' lines."""
    lex = {}
    for ln, line in enumerate(text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"{source}:{ln}: expected 'word phones', got {line!r}")
        word, pron = normalise(parts[0]), parts[1]
        sylls = [s.split() for s in pron.split(SYL_SEP)]
        sylls = [s for s in sylls if s]
        bad = [p for s in sylls for p in s if p not in DUTCH_CV_PHONES]
        if bad:
            raise ValueError(f"{source}:{ln}: {word}: phones not in the dutch_cv set: {bad}\n"
                             f"allowed: {' '.join(sorted(DUTCH_CV_PHONES))}")
        lex[word] = sylls
    return lex


def build_lexicon(words, dict_local: Path | None = None):
    """Return {word: (syllables, source)} for every word."""
    exc = parse_syllabified(BUILTIN_EXCEPTIONS, "builtin")
    local = {}
    if dict_local is not None and Path(dict_local).exists():
        local = parse_syllabified(Path(dict_local).read_text(encoding="utf-8"), str(dict_local))
    out = {}
    for w in sorted({normalise(x) for x in words if x.strip()}):
        if w in local:
            out[w] = (local[w], "dict.local")
        elif w in exc:
            out[w] = (exc[w], "builtin")
        else:
            out[w] = (rule_g2p(w), "rules")
        bad = [p for s in out[w][0] for p in s if p not in DUTCH_CV_PHONES]
        if bad or not out[w][0]:
            raise ValueError(f"no valid pronunciation for {w!r}: {out[w][0]}")
    return out


def fmt_sylls(sylls) -> str:
    return f" {SYL_SEP} ".join(" ".join(s) for s in sylls)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("words", nargs="*")
    ap.add_argument("--file", help="one word per line (or any text: split on whitespace)")
    ap.add_argument("--dict-local", type=Path)
    a = ap.parse_args(argv)
    words = list(a.words)
    if a.file:
        words += Path(a.file).read_text(encoding="utf-8").split()
    for w, (s, src) in build_lexicon(words, a.dict_local).items():
        print(f"{w}\t{fmt_sylls(s)}\t[{src}]")


if __name__ == "__main__":
    sys.exit(main())
__AFR_GT_EOF__
cat > afr_gt_align.py <<'__AFR_GT_EOF__'
#!/usr/bin/env python3
"""
Ground-truth phone / word / syllable TextGrids for the Afrikaans babaloon child speech.

    corpus (wav + TextGrid with one "child" tier, one interval per utterance)
      -> MFA forced alignment, inside each utterance's time stamps
         (acoustic model: MFA dutch_cv, pronunciations: rule-based Afrikaans G2P)
      -> syllable tier from the rule-based Afrikaans syllabifier (afr_g2p.py)
      -> silences between words *inside* an utterance removed: each gap is split
         at its midpoint between the two neighbouring words
      -> silence before the first word / after the last word of the file is
         given to the first / last syllable
      -> silences *between* utterances are kept as empty intervals
      -> Praat TextGrid, tiers: words, phones, syllables

Commands
    run          prepare + lexicon + MFA + postprocess (the normal way)
    prepare      build the MFA input corpus and the pronunciation lexicon only
    postprocess  rebuild the output TextGrids from an existing MFA output folder
    check        verify output TextGrids (no silences inside utterances, tiers in sync)

Example
    python afr_gt_align.py run \
        --corpus ~/Skripsie/DATA/babaloon_alignments/corpora/afr/dev \
        --out    ~/Skripsie/DATA/babaloon_alignments/alignments/af/dev/Syllable_ground_truth_hendrik_made
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
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import afr_g2p  # noqa: E402

HOME = Path.home()
DEFAULT_CORPUS = HOME / "Skripsie/DATA/babaloon_alignments/corpora/afr/dev"
DEFAULT_OUT = HOME / "Skripsie/DATA/babaloon_alignments/alignments/af/dev/Syllable_ground_truth_hendrik_made"
DEFAULT_WORK = HOME / "Skripsie/DATA/babaloon_alignments/work_hendrik_gt"
_LOCAL_MODEL = Path(__file__).resolve().parent / "models" / "dutch_cv.zip"
DEFAULT_MODEL = str(_LOCAL_MODEL) if _LOCAL_MODEL.exists() else "dutch_cv"

EPS = 1e-6


# ==========================================================================
# TextGrid reading / writing (no dependencies; reads long and short format)
# ==========================================================================
@dataclass
class Interval:
    start: float
    end: float
    text: str


@dataclass
class Tier:
    name: str
    intervals: list[Interval] = field(default_factory=list)


@dataclass
class TextGrid:
    xmin: float
    xmax: float
    tiers: list[Tier]

    def tier(self, name: str) -> Tier:
        for t in self.tiers:
            if t.name == name:
                return t
        raise KeyError(name)


def read_textgrid(path: Path) -> TextGrid:
    """Minimal Praat TextGrid reader for interval tiers (long or short text format)."""
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        txt = raw.decode("utf-16")
    else:
        txt = raw.decode("utf-8-sig")
    # drop long-format index markers like "item [1]:" / "intervals [3]:" (outside quotes)
    parts = re.split(r'("(?:[^"]|"")*")', txt)
    txt = "".join(p if k % 2 else re.sub(r"\[\d*\]", " ", p) for k, p in enumerate(parts))
    # tokens: quoted strings (with "" escapes) or bare numbers / words
    toks = re.findall(r'"((?:[^"]|"")*)"|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)|(<exists>)', txt)
    vals = []
    for q, num, ex in toks:
        if ex:
            vals.append(("ex", ex))
        elif num:
            vals.append(("num", float(num)))
        else:
            vals.append(("str", q.replace('""', '"')))
    i = 0

    def nxt(kind=None):
        nonlocal i
        k, v = vals[i]
        i += 1
        if kind and k != kind:
            raise ValueError(f"{path}: expected {kind}, got {k} {v!r}")
        return v

    assert nxt("str") == "ooTextFile"
    assert nxt("str") == "TextGrid"
    xmin, xmax = nxt("num"), nxt("num")
    if vals[i][0] == "ex":
        i += 1
    n_tiers = int(nxt("num"))
    tiers = []
    for _ in range(n_tiers):
        cls = nxt("str")
        name = nxt("str")
        nxt("num"), nxt("num")
        n = int(nxt("num"))
        t = Tier(name)
        for _ in range(n):
            if cls == "IntervalTier":
                s, e, txt_ = nxt("num"), nxt("num"), nxt("str")
                t.intervals.append(Interval(s, e, txt_))
            else:  # point tier: skip
                nxt("num"), nxt("str")
        tiers.append(t)
    return TextGrid(xmin, xmax, tiers)


def _q(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def _fill(intervals: list[Interval], xmin: float, xmax: float) -> list[Interval]:
    """Add empty intervals in the gaps so the tier covers xmin..xmax."""
    out, t = [], xmin
    for iv in sorted(intervals, key=lambda x: x.start):
        if iv.start > t + EPS:
            out.append(Interval(t, iv.start, ""))
        out.append(Interval(max(iv.start, t), iv.end, iv.text))
        t = iv.end
    if xmax > t + EPS:
        out.append(Interval(t, xmax, ""))
    return out


def write_textgrid(path: Path, xmin: float, xmax: float, tiers: list[Tier]) -> None:
    """Praat long text format (same layout as the reference ground-truth files)."""
    L = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
         f"xmin = {xmin} ", f"xmax = {xmax} ", "tiers? <exists> ", f"size = {len(tiers)} ", "item []: "]
    for ti, tier in enumerate(tiers, 1):
        ivs = _fill(tier.intervals, xmin, xmax)
        L += [f"    item [{ti}]:", '        class = "IntervalTier" ', f"        name = {_q(tier.name)} ",
              f"        xmin = {xmin} ", f"        xmax = {xmax} ", f"        intervals: size = {len(ivs)} "]
        for k, iv in enumerate(ivs, 1):
            L += [f"        intervals [{k}]:", f"            xmin = {round(iv.start, 6)} ",
                  f"            xmax = {round(iv.end, 6)} ", f"            text = {_q(iv.text)} "]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ==========================================================================
# Corpus preparation
# ==========================================================================
def norm_text(s: str) -> list[str]:
    s = unicodedata.normalize("NFC", s.lower())
    s = re.sub(r"[^\w\s'\-]", " ", s)          # drop punctuation, keep letters/digits/'/-
    return [w.strip("'-") for w in s.split() if w.strip("'-")]


def speaker_of(stem: str) -> str:
    m = re.match(r"(id_\d+)", stem)
    return m.group(1) if m else stem


def utterances(tg: TextGrid, tier_name: str | None) -> list[Interval]:
    tier = tg.tier(tier_name) if tier_name else tg.tiers[0]
    return [iv for iv in tier.intervals if norm_text(iv.text)]


def find_pairs(corpus: Path):
    pairs = []
    for tgp in sorted(corpus.rglob("*.TextGrid")):
        wav = tgp.with_suffix(".wav")
        if wav.exists():
            pairs.append((wav, tgp))
        else:
            print(f"  ! no wav for {tgp.name}, skipped")
    return pairs


def prepare(args) -> dict:
    corpus, work = args.corpus.expanduser(), args.work_dir.expanduser()
    mfa_in = work / "mfa_corpus"
    if mfa_in.exists():
        shutil.rmtree(mfa_in)
    mfa_in.mkdir(parents=True)
    pairs = find_pairs(corpus)
    if args.only:
        pairs = [p for p in pairs if p[0].stem in set(args.only)]
    if not pairs:
        sys.exit(f"no wav+TextGrid pairs found in {corpus}")
    vocab = []
    for wav, tgp in pairs:
        tg = read_textgrid(tgp)
        utts = utterances(tg, args.tier)
        spk = speaker_of(wav.stem)
        ivs = []
        for u in utts:
            words = norm_text(u.text)
            vocab += words
            ivs.append(Interval(u.start, min(u.end, tg.xmax), " ".join(words)))
        # tier name = speaker, so MFA adapts per child (pre and post test = same child)
        write_textgrid(mfa_in / f"{wav.stem}.TextGrid", tg.xmin, tg.xmax, [Tier(spk, ivs)])
        shutil.copy2(wav, mfa_in / wav.name)
    print(f"prepared {len(pairs)} files in {mfa_in}")

    # ---------------- lexicon ----------------
    lexdir = work / "lexicon"
    lexdir.mkdir(parents=True, exist_ok=True)
    lex = afr_g2p.build_lexicon(vocab, args.dict_local)
    with open(lexdir / "afrikaans_dutchcv.dict", "w", encoding="utf-8") as f:
        for w, (sylls, _) in lex.items():
            f.write(f"{w}\t{' '.join(p for s in sylls for p in s)}\n")
    with open(lexdir / "afrikaans_syllabified.txt", "w", encoding="utf-8") as f:
        f.write("# word\tsyllabified pronunciation (dutch_cv phones)\tsource\n")
        for w, (sylls, src) in lex.items():
            f.write(f"{w}\t{afr_g2p.fmt_sylls(sylls)}\t{src}\n")
    n_rules = sum(1 for _, (_, s) in lex.items() if s == "rules")
    print(f"lexicon: {len(lex)} words ({n_rules} from spelling rules) -> {lexdir}")
    return {w: s for w, (s, _) in lex.items()}


def load_syllabified(work: Path) -> dict:
    p = work / "lexicon" / "afrikaans_syllabified.txt"
    lex = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        w, pron = line.split("\t")[:2]
        lex[w] = [s.split() for s in pron.split(afr_g2p.SYL_SEP)]
    return lex


# ==========================================================================
# MFA
# ==========================================================================
def run_mfa(args) -> Path:
    work = args.work_dir.expanduser()
    mfa = shutil.which("mfa")
    if not mfa:
        sys.exit("`mfa` not found: activate the conda env first (conda activate aligner)")
    out = work / "mfa_output"
    if out.exists():
        shutil.rmtree(out)
    corpus, dic = work / "mfa_corpus", work / "lexicon" / "afrikaans_dutchcv.dict"
    common = ["--clean", "--beam", str(args.beam), "--retry_beam", str(args.retry_beam),
              "-j", str(args.jobs), "--temporary_directory", str(work / "mfa_tmp")]
    if args.adapt:
        model_out = work / "dutch_cv_adapted_babaloon.zip"
        cmd = [mfa, "adapt", str(corpus), str(dic), args.acoustic_model, str(model_out),
               "--output_directory", str(out)] + common
    else:
        cmd = [mfa, "align", str(corpus), str(dic), args.acoustic_model, str(out)] + common
    print("+ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"MFA failed (exit {r.returncode})")
    return out


# ==========================================================================
# Post-processing: syllables + silence removal
# ==========================================================================
@dataclass
class Word:
    text: str
    phones: list[list]          # [start, end, label] (mutable)
    utt: int = -1
    sylls: list[int] = field(default_factory=list)   # phones per syllable
    fallback: bool = False

    @property
    def start(self):
        return self.phones[0][0]

    @property
    def end(self):
        return self.phones[-1][1]


def generic_syllables(labels: list[str]) -> list[int]:
    """Fallback when the aligned phones don't match the lexicon: MOP on dutch_cv phones."""
    nuc = [i for i, p in enumerate(labels) if p in afr_g2p.VOWEL_PHONES or p == "spn"]
    if len(nuc) <= 1:
        return [len(labels)]
    # glides directly after a vowel belong to the nucleus
    starts = [0]
    for a, b in zip(nuc, nuc[1:]):
        k = a + 1
        while k < b and labels[k] in ("j", "w", "ɥ"):
            k += 1
        cons = labels[k:b]
        split = b
        for n in range(len(cons), 0, -1):
            on = cons[len(cons) - n:]
            if (n == 1 and on[0] != "ŋ") or (n == 2 and tuple(on) in afr_g2p.LEGAL_ONSETS):
                split = b - n
                break
        starts.append(split)
    ends = starts[1:] + [len(labels)]
    return [e - s for s, e in zip(starts, ends)]


def find_tier(tg: TextGrid, suffix: str) -> Tier:
    for t in tg.tiers:
        if t.name == suffix or t.name.endswith(" - " + suffix):
            return t
    raise KeyError(f"no '{suffix}' tier in MFA output (tiers: {[t.name for t in tg.tiers]})")


def build_file(stem: str, mfa_tg: TextGrid, utts: list[Interval], lex: dict, edge: str):
    words_t, phones_t = find_tier(mfa_tg, "words"), find_tier(mfa_tg, "phones")
    xmin, xmax = 0.0, mfa_tg.xmax
    ph = [[iv.start, iv.end, iv.text] for iv in phones_t.intervals if iv.text.strip()]
    words: list[Word] = []
    for iv in words_t.intervals:
        if not iv.text.strip():
            continue
        mine = [p for p in ph if iv.start - EPS <= (p[0] + p[1]) / 2 <= iv.end + EPS]
        if not mine:
            mine = [[iv.start, iv.end, "spn"]]
        # make phones inside a word contiguous and match the word's span
        mine[0][0] = iv.start
        mine[-1][1] = iv.end
        for a, b in zip(mine, mine[1:]):
            b[0] = a[1]
        words.append(Word(iv.text.strip(), mine))
    words.sort(key=lambda w: w.start)

    # ---- syllables per word ----
    for w in words:
        labels = [p[2] for p in w.phones]
        sylls = lex.get(unicodedata.normalize("NFC", w.text.lower()))
        if sylls and [p for s in sylls for p in s] == labels:
            w.sylls = [len(s) for s in sylls]
        else:
            w.sylls, w.fallback = generic_syllables(labels), True

    # ---- utterance of each word ----
    for w in words:
        mid = (w.start + w.end) / 2
        for k, u in enumerate(utts):
            if u.start - EPS <= mid <= u.end + EPS:
                w.utt = k
                break
        else:
            w.utt = min(range(len(utts)), key=lambda k: min(abs(utts[k].start - mid), abs(utts[k].end - mid)))

    # ---- remove silences between words inside an utterance (split at midpoint) ----
    filled, max_gap = 0.0, 0.0
    for a, b in zip(words, words[1:]):
        gap = b.start - a.end
        if a.utt == b.utt and gap > EPS:
            mid = (a.end + b.start) / 2
            a.phones[-1][1] = mid
            b.phones[0][0] = mid
            filled += gap
            max_gap = max(max_gap, gap)

    # ---- leading / trailing silence of the file -> first / last syllable ----
    lead = trail = 0.0
    if words and edge == "absorb":
        lead, trail = words[0].start - xmin, xmax - words[-1].end
        words[0].phones[0][0] = xmin
        words[-1].phones[-1][1] = xmax

    # ---- tiers ----
    W, P, S = Tier("words"), Tier("phones"), Tier("syllables")
    for w in words:
        W.intervals.append(Interval(w.start, w.end, w.text))
        for s, e, lab in w.phones:
            P.intervals.append(Interval(s, e, lab))
        k = 0
        for si, n in enumerate(w.sylls, 1):
            chunk = w.phones[k:k + n]
            k += n
            label = w.text if len(w.sylls) == 1 else f"{w.text}-{si}"
            S.intervals.append(Interval(chunk[0][0], chunk[-1][1], label))

    stats = dict(words=words, filled=filled, max_gap=max_gap, lead=lead, trail=trail)
    return xmin, xmax, [W, P, S], stats


def eval_groups(syll_tier: Tier, xmin: float, xmax: float) -> int:
    """Number of utterance groups boundaries.py's split_utterance will see."""
    ivs = _fill(syll_tier.intervals, xmin, xmax)
    groups, prev_nonempty = 0, False
    for iv in ivs:
        ne = iv.text != ""
        if ne and not prev_nonempty:
            groups += 1
        prev_nonempty = ne
    return groups


def postprocess(args, mfa_out: Path | None = None) -> None:
    corpus, work, out = args.corpus.expanduser(), args.work_dir.expanduser(), args.out.expanduser()
    mfa_out = Path(mfa_out or args.mfa_output or (work / "mfa_output")).expanduser()
    lex = load_syllabified(work)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    pairs = find_pairs(corpus)
    if args.only:
        pairs = [p for p in pairs if p[0].stem in set(args.only)]
    for wav, tgp in pairs:
        stem = wav.stem
        utts = utterances(read_textgrid(tgp), args.tier)
        n_tr = sum(len(norm_text(u.text)) for u in utts)
        cand = list(mfa_out.rglob(f"{stem}.TextGrid"))
        if not cand:
            print(f"  ! {stem}: no MFA output (not aligned)")
            rows.append(dict(file=stem, status="NOT_ALIGNED", utterances=len(utts), transcript_words=n_tr))
            continue
        xmin, xmax, tiers, st = build_file(stem, read_textgrid(cand[0]), utts, lex, args.edge_silence)
        write_textgrid(out / f"{stem}{args.suffix}.TextGrid", xmin, xmax, tiers)
        words = st["words"]
        utts_aligned = len({w.utt for w in words})
        n_spn = sum(1 for w in words if any(p[2] == "spn" for p in w.phones) and w.text != "unk")
        n_fb = sum(1 for w in words if w.fallback)
        groups = eval_groups(tiers[2], xmin, xmax)
        problems = []
        if len(words) != n_tr:
            problems.append(f"words {len(words)}/{n_tr}")
        if utts_aligned != len(utts):
            problems.append(f"utterances {utts_aligned}/{len(utts)}")
        if n_spn:
            problems.append(f"{n_spn} word(s) aligned as spn")
        if n_fb:
            problems.append(f"{n_fb} word(s) syllabified by fallback")
        rows.append(dict(
            file=stem, status="CHECK" if problems else "OK", problems="; ".join(problems),
            utterances=len(utts), utterances_aligned=utts_aligned, eval_groups=groups,
            transcript_words=n_tr, aligned_words=len(words), syllables=len(tiers[2].intervals),
            phones=len(tiers[1].intervals), silence_filled_s=round(st["filled"], 3),
            largest_gap_filled_s=round(st["max_gap"], 3),
            leading_silence_s=round(st["lead"], 3), trailing_silence_s=round(st["trail"], 3)))
        print(f"  {stem}: {len(words)} words, {len(tiers[2].intervals)} syllables, "
              f"{utts_aligned}/{len(utts)} utterances, {groups} eval groups"
              + (f"  CHECK: {'; '.join(problems)}" if problems else ""))
    qc = work / "alignment_qc.csv"
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(qc, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    n_ok = sum(r["status"] == "OK" for r in rows)
    print(f"\nwrote {len(rows) - sum(r['status'] == 'NOT_ALIGNED' for r in rows)} TextGrids to {out}")
    print(f"QC: {n_ok}/{len(rows)} OK -> {qc}")


# ==========================================================================
# check
# ==========================================================================
def check(args) -> None:
    """Verify output files against the corpus utterances."""
    corpus, out = args.corpus.expanduser(), args.out.expanduser()
    bad = 0
    for wav, tgp in find_pairs(corpus):
        if args.only and wav.stem not in set(args.only):
            continue
        p = out / f"{wav.stem}{args.suffix}.TextGrid"
        if not p.exists():
            print(f"  missing: {p.name}")
            bad += 1
            continue
        tg = read_textgrid(p)
        utts = utterances(read_textgrid(tgp), args.tier)
        msgs = []
        names = [t.name for t in tg.tiers]
        if names != ["words", "phones", "syllables"]:
            msgs.append(f"tiers {names}")
        W, P, S = (tg.tier(n) for n in ("words", "phones", "syllables"))
        # boundaries of syllables must be phone boundaries, word boundaries syllable boundaries
        pb = {round(iv.start, 5) for iv in P.intervals} | {round(iv.end, 5) for iv in P.intervals}
        sb = {round(iv.start, 5) for iv in S.intervals} | {round(iv.end, 5) for iv in S.intervals}
        wb = {round(iv.start, 5) for iv in W.intervals} | {round(iv.end, 5) for iv in W.intervals}
        if not sb <= pb:
            msgs.append("syllable boundary not on a phone boundary")
        if not wb <= sb:
            msgs.append("word boundary not on a syllable boundary")
        # no empty interval between two syllables of the same utterance
        ne = [iv for iv in S.intervals if iv.text]
        def utt_of(iv):
            m = (iv.start + iv.end) / 2
            return next((k for k, u in enumerate(utts) if u.start - EPS <= m <= u.end + EPS), -1)
        for a, b in zip(ne, ne[1:]):
            if utt_of(a) == utt_of(b) and b.start - a.end > EPS:
                msgs.append(f"silence inside utterance at {a.end:.3f}-{b.start:.3f}")
                break
        if ne and (ne[0].start > tg.xmin + EPS or ne[-1].end < tg.xmax - EPS) and args.edge_silence == "absorb":
            msgs.append("leading/trailing silence not absorbed")
        status = "OK" if not msgs else "; ".join(msgs)
        bad += bool(msgs)
        print(f"  {wav.stem}: {status}")
    print("all good" if not bad else f"{bad} file(s) with problems")


# ==========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="folder with wav + TextGrid pairs")
        p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder for the GT TextGrids")
        p.add_argument("--work-dir", type=Path, default=DEFAULT_WORK, help="MFA input/output, lexicon, QC")
        p.add_argument("--tier", default=None, help="utterance tier in the corpus TextGrids (default: first tier)")
        p.add_argument("--only", nargs="+", help="only these file stems, e.g. id_3_main_post_test")
        p.add_argument("--suffix", default="", help="added to output file names (default none: same stem as the wav)")
        p.add_argument("--edge-silence", choices=["absorb", "keep"], default="absorb",
                       help="silence before the first / after the last word of the file (default: absorb)")
        p.add_argument("--dict-local", type=Path, default=Path(__file__).resolve().parent / "dict.local",
                       help="extra/corrected pronunciations, syllables separated by ' . '")

    for name in ("run", "prepare", "postprocess", "check"):
        p = sub.add_parser(name)
        common(p)
        if name == "run":
            p.add_argument("--acoustic-model", default=DEFAULT_MODEL,
                           help="MFA model name or .zip path (default: models/dutch_cv.zip next to this script)")
            p.add_argument("--adapt", action="store_true",
                           help="adapt the Dutch model to these children's voices first (mfa adapt)")
            p.add_argument("--beam", type=int, default=100)
            p.add_argument("--retry-beam", type=int, default=400)
            p.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
        if name == "postprocess":
            p.add_argument("--mfa-output", type=Path, default=None,
                           help="MFA output folder (default: <work-dir>/mfa_output)")
    a = ap.parse_args(argv)
    if not a.dict_local.exists():
        a.dict_local = None

    if a.cmd == "prepare":
        prepare(a)
    elif a.cmd == "run":
        prepare(a)
        out = run_mfa(a)
        a.mfa_output = None
        postprocess(a, out)
        check(a)
    elif a.cmd == "postprocess":
        postprocess(a)
        check(a)
    elif a.cmd == "check":
        check(a)


if __name__ == "__main__":
    main()
__AFR_GT_EOF__
cat > afr_add_syllables.py <<'__AFR_GT_EOF__'
#!/usr/bin/env python3
"""
Add a rule-based Afrikaans syllable tier to existing word/phone TextGrids.

Input : a folder of TextGrids with a "words" and a "phones" tier (forced alignments)
Output: <input folder>/gt/<same file name>.TextGrid with tiers  words, phones, syllables
        (words and phones are copied unchanged; silences stay as they are)

    python3 afr_add_syllables.py ~/Skripsie/DATA/babaloon_alignments/alignments/af/test
    python3 afr_add_syllables.py IN_DIR --out OUT_DIR

No extra packages are needed (plain Python 3.9+). afr_g2p.py must be in the same folder.

How the syllables are made (same Afrikaans rules as the dev ground truth):
  * the syllables are cut from each word's *aligned* phones, so every syllable
    boundary is a phone boundary and syllables never cross words
  * every vowel phone (a, aː, ə, oəy, uə, aɪ, ...) is a syllable nucleus;
    the spelling (afr_g2p.py) says how many syllables the word has, which decides
    whether two adjacent vowel phones are one diphthong or two syllables (ge.ëet)
  * prefixes / compound parts end a syllable: op.eet, ge.spring, weg.hard.loop, hier.oor
  * other consonants between two vowels: Maximal Onset with Afrikaans onsets
    (pr br tr dr kr ɡr fr vr pl bl kl ɡl fl vl kn kw tw dw ...), but
    s + consonant is not an onset inside a word (kas.te, vis.lyn) and ŋ is never an onset
  * a word with no vowel phone (hmm, s, spn) is one syllable
  * labels: "word" for one syllable, "word-1", "word-2", ... otherwise
"""
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import afr_g2p  # noqa: E402

EPS = 1e-6
VOWEL_CHARS = set("aeiouyæøœɑɒɔəɛɜɞɪʊʌʏɐɘɵɤɯɨʉ")
LIQUIDS = {"r", "l", "ɾ", "ɹ"}
OBSTRUENTS_BEFORE_LIQUID = {"p", "b", "t", "d", "k", "ɡ", "g", "x", "f", "v", "ʃ"}
OTHER_ONSETS = {("k", "n"), ("k", "w"), ("k", "v"), ("k", "ʋ"), ("t", "w"), ("t", "v"), ("t", "ʋ"),
                ("d", "w"), ("d", "v"), ("d", "ʋ"), ("x", "w"), ("ɡ", "w")}


# --------------------------------------------------------------------------
# TextGrid I/O (long or short Praat text format)
# --------------------------------------------------------------------------
def read_textgrid(path: Path):
    raw = path.read_bytes()
    txt = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    parts = re.split(r'("(?:[^"]|"")*")', txt)
    txt = "".join(p if k % 2 else re.sub(r"\[\d*\]", " ", p) for k, p in enumerate(parts))
    toks = re.findall(r'"((?:[^"]|"")*)"|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)|(<exists>)', txt)
    vals = [("ex", e) if e else ("num", float(n)) if n else ("str", q.replace('""', '"')) for q, n, e in toks]
    i = 0

    def nxt():
        nonlocal i
        i += 1
        return vals[i - 1][1]

    nxt(), nxt()
    xmin, xmax = nxt(), nxt()
    if vals[i][0] == "ex":
        i += 1
    tiers = {}
    order = []
    for _ in range(int(nxt())):
        cls, name = nxt(), nxt()
        nxt(), nxt()
        ivs = []
        for _ in range(int(nxt())):
            if cls == "IntervalTier":
                ivs.append([nxt(), nxt(), nxt()])
            else:
                nxt(), nxt()
        tiers[name] = ivs
        order.append(name)
    return xmin, xmax, tiers, order


def _q(s):
    return '"' + s.replace('"', '""') + '"'


def _fill(ivs, xmin, xmax):
    out, t = [], xmin
    for s, e, lab in sorted(ivs, key=lambda x: x[0]):
        if s > t + EPS:
            out.append([t, s, ""])
        out.append([max(s, t), e, lab])
        t = e
    if xmax > t + EPS:
        out.append([t, xmax, ""])
    return out


def write_textgrid(path: Path, xmin, xmax, tiers):
    L = ['File type = "ooTextFile"', 'Object class = "TextGrid"', "",
         f"xmin = {xmin} ", f"xmax = {xmax} ", "tiers? <exists> ", f"size = {len(tiers)} ", "item []: "]
    for ti, (name, ivs) in enumerate(tiers, 1):
        ivs = _fill(ivs, xmin, xmax)
        L += [f"    item [{ti}]:", '        class = "IntervalTier" ', f"        name = {_q(name)} ",
              f"        xmin = {xmin} ", f"        xmax = {xmax} ", f"        intervals: size = {len(ivs)} "]
        for k, (s, e, lab) in enumerate(ivs, 1):
            L += [f"        intervals [{k}]:", f"            xmin = {s} ",
                  f"            xmax = {e} ", f"            text = {_q(lab)} "]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Spelling plan: number of syllables + prefix boundaries, from afr_g2p rules
# --------------------------------------------------------------------------
_EXC = afr_g2p.parse_syllabified(afr_g2p.BUILTIN_EXCEPTIONS, "builtin")


def spelling_plan(word: str):
    """Return (n_syllables, {gap_index: n_coda_consonants}) for a forced (prefix) boundary."""
    w = afr_g2p.normalise(word)
    if w in _EXC:
        return len(_EXC[w]), {}
    units = afr_g2p._tokenise(w)
    afr_g2p._assign_vowels(units, w)
    afr_g2p._consonant_rules(units)
    nuc = [k for k, u in enumerate(units) if u.kind == "V"]
    forced = {}
    for g, (a, b) in enumerate(zip(nuc, nuc[1:])):
        between = units[a + 1:b]
        if any(u.kind == "|" for u in between):
            bar = max(k for k, u in enumerate(between) if u.kind == "|")
            forced[g] = sum(len(u.phones) for u in between[:bar] if u.kind == "C")
    return max(1, len(nuc)), forced


# --------------------------------------------------------------------------
# Phone-level syllabification
# --------------------------------------------------------------------------
def is_vowel(p: str) -> bool:
    return bool(p) and p[0] in VOWEL_CHARS


def base(p: str) -> str:
    return p.replace("ː", "").replace(":", "").replace("ʰ", "").replace("ʲ", "")


def legal_onset(cons: list[str]) -> bool:
    if not cons:
        return True
    c = [base(p) for p in cons]
    if len(c) == 1:
        return c[0] not in ("ŋ",)
    if len(c) == 2:
        if c[0] in OBSTRUENTS_BEFORE_LIQUID and c[1] in LIQUIDS and (c[0], c[1]) not in (("t", "l"), ("d", "l")):
            return True
        return (c[0], c[1]) in OTHER_ONSETS
    return False   # 3-consonant onsets inside a word are s-clusters (spr, str, skr): not used


def syllabify_phones(word: str, labels: list[str]) -> list[int]:
    """Return the number of phones in each syllable."""
    n = len(labels)
    vow = [k for k, p in enumerate(labels) if is_vowel(p)]
    if not vow:
        return [n]
    runs = []                                   # groups of adjacent vowel phones
    for k in vow:
        if runs and runs[-1][-1] == k - 1:
            runs[-1].append(k)
        else:
            runs.append([k])
    n_sp, forced = spelling_plan(word)
    if len(vow) == n_sp:
        nuclei = [[k] for k in vow]
    elif len(runs) == n_sp:
        nuclei = runs
    else:
        nuclei = runs if abs(len(runs) - n_sp) <= abs(len(vow) - n_sp) else [[k] for k in vow]
        forced = {}                             # can't map spelling boundaries onto these phones
    if len(nuclei) != n_sp:
        forced = {}
    starts = [0]
    for g, (a, b) in enumerate(zip(nuclei, nuclei[1:])):
        lo, hi = a[-1] + 1, b[0]               # consonants labels[lo:hi]
        cons = labels[lo:hi]
        if g in forced:
            split = lo + min(forced[g], len(cons))
        else:
            split = hi
            for k in range(len(cons), 0, -1):
                if legal_onset(cons[len(cons) - k:]):
                    split = hi - k
                    break
        starts.append(split)
    ends = starts[1:] + [n]
    return [e - s for s, e in zip(starts, ends)]


# --------------------------------------------------------------------------
def find(tiers, order, name):
    for t in order:
        if t == name or t.endswith(" - " + name):
            return tiers[t]
    raise KeyError(f"no '{name}' tier (tiers: {order})")


def process(path: Path, out: Path):
    xmin, xmax, tiers, order = read_textgrid(path)
    words = [iv for iv in find(tiers, order, "words") if iv[2].strip()]
    phones = [iv for iv in find(tiers, order, "phones") if iv[2].strip()]
    sylls, warn = [], []
    for ws, we, wl in words:
        wl = wl.strip()
        ph = [p for p in phones if ws - EPS <= (p[0] + p[1]) / 2 <= we + EPS]
        if not ph:
            sylls.append([ws, we, wl])
            warn.append(f"{wl}@{ws:.2f}: no phones")
            continue
        sizes = syllabify_phones(wl, [p[2] for p in ph])
        k = 0
        for i, size in enumerate(sizes, 1):
            chunk = ph[k:k + size]
            k += size
            s = ws if i == 1 else chunk[0][0]           # syllables cover the whole word
            e = we if i == len(sizes) else chunk[-1][1]
            sylls.append([s, e, wl if len(sizes) == 1 else f"{wl}-{i}"])
    write_textgrid(out, xmin, xmax, [("words", find(tiers, order, "words")),
                                     ("phones", find(tiers, order, "phones")),
                                     ("syllables", sylls)])
    return len(words), len(sylls), warn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="folder with word/phone TextGrids (or one TextGrid)")
    ap.add_argument("--out", type=Path, help="output folder (default: <input>/gt)")
    a = ap.parse_args(argv)
    inp = a.input.expanduser()
    files = [inp] if inp.is_file() else sorted(inp.glob("*.TextGrid"))
    out = (a.out or ((inp.parent if inp.is_file() else inp) / "gt")).expanduser()
    if not files:
        sys.exit(f"no .TextGrid files in {inp}")
    tw = ts = 0
    for f in files:
        nw, ns, warn = process(f, out / f.name)
        tw, ts = tw + nw, ts + ns
        print(f"  {f.name}: {nw} words -> {ns} syllables" + (f"   ! {'; '.join(warn)}" if warn else ""))
    print(f"\n{len(files)} files, {tw} words, {ts} syllables -> {out}")


if __name__ == "__main__":
    main()
__AFR_GT_EOF__
if [ ! -f dict.local ]; then   # never overwrite your own corrections
cat > dict.local <<'__AFR_GT_EOF__'
# Extra or corrected pronunciations. These override the built-in rules.
# Format: word <TAB or spaces> phones, with syllables separated by " . "
# Phones must be from the dutch_cv set:
#   a aː b c d eː f h i iː j k l m n oː p r s t u uː v w x y yː z øː ŋ œ ɑ ɔ ɛ ɣ ɥ ɪ ʃ ʋ ʏ   (+ spn = noise)
# Look up a word first:  python afr_g2p.py babaloon
#
# examples (uncomment / edit):
# babaloon    b ɑ . b ɑ . l u n
# ball        b ɑ l
__AFR_GT_EOF__
fi
cat > README.md <<'__AFR_GT_EOF__'
# Afrikaans babaloon: ground-truth phone / word / syllable TextGrids

`afr_gt_align.py` makes one TextGrid per recording in
`~/Skripsie/DATA/babaloon_alignments/alignments/af/dev/Syllable_ground_truth_hendrik_made/`,
with three tiers in this order: `words`, `phones`, `syllables`. This is the same layout as the
reference ground-truth files.

| tier        | how it's made                                                              |
|-------------|----------------------------------------------------------------------------|
| `words`     | MFA word alignment                                                         |
| `phones`    | MFA phone alignment (Dutch CV acoustic model, Epitran phone set)           |
| `syllables` | rule-based Afrikaans syllabification of each word's phones (`word`, or `word-1`, `word-2`, …) |

## Silence handling (for `boundaries.py`)

`split_utterance` in `boundaries.py` treats every run of non-empty syllables as one utterance,
so a pause between two words would split an utterance in two. The output therefore follows
these rules:

* **Pauses between words inside an utterance are removed.** An utterance is one interval
  of the corpus `child` tier. Each pause is split exactly in the middle: the word before
  it ends at the midpoint and the word after it starts there. The phone, word and syllable
  tiers all move together.
* **Silences between utterances are kept** as empty intervals.
* **Silence before the first word and after the last word of the file** is given to the
  first and last syllable.
* Fillers (`uh`, `um`, `hmm`, `huh`), word fragments (`s`, `g`, `ba`, …) and `unk` are
  all syllables. `unk` is aligned as spoken noise (`spn`).

If two utterances follow each other with no pause in between, `boundaries.py` sees them as
one group. That's harmless, because there's no silence inside the group.
(`id_3_main_post_test`: 20 utterances → 19 groups.)

## 1. One-time setup

Put this folder somewhere, for example `~/Skripsie/code/afr_gt`, then:

```bash
cd ~/Skripsie/code/afr_gt
bash setup_afr_mfa.sh        # reuses conda env "aligner" (or creates it), downloads models/dutch_cv.zip
```

## 2. Run it

```bash
conda activate aligner
cd ~/Skripsie/code/afr_gt

# quick test on one file
python afr_gt_align.py run --only id_3_main_post_test

# all 40 files in corpora/afr/dev
python afr_gt_align.py run
```

The default paths are already set to your folders:

| option        | default                                                                      |
|---------------|------------------------------------------------------------------------------|
| `--corpus`    | `~/Skripsie/DATA/babaloon_alignments/corpora/afr/dev`                        |
| `--out`       | `~/Skripsie/DATA/babaloon_alignments/alignments/af/dev/Syllable_ground_truth_hendrik_made` |
| `--work-dir`  | `~/Skripsie/DATA/babaloon_alignments/work_hendrik_gt` (MFA input/output, lexicon, QC) |

The work dir is deliberately **outside** `alignments/`, so `boundaries.py`'s TextGrid glob
only ever sees the ground-truth files.

Other options:

```
--adapt                adapt the Dutch model to these children's voices first (mfa adapt), then align
--beam 100 --retry-beam 400   widen these if MFA can't align some utterances
-j N                   parallel jobs (each child id_X is one speaker)
--suffix _syllables    add a suffix to the output file names (default: same stem as the wav,
                       which is what boundaries.py needs: sp.stem == tp.stem)
--edge-silence keep    leave the file's leading/trailing silence as empty intervals
--only STEM ...        only these files
```

Other commands:

```bash
python afr_gt_align.py prepare       # only build the MFA corpus + lexicon (inspect before aligning)
python afr_gt_align.py postprocess   # rebuild the TextGrids from the existing MFA output (no re-alignment)
python afr_gt_align.py check         # verify the output files
python afr_g2p.py babaloon seuntjie geëet   # show pronunciation + syllables of any word
```

## 3. Check the result

`work_hendrik_gt/alignment_qc.csv` has one row per file. A row is `CHECK` when:

* the number of aligned words differs from the transcript,
* an utterance has no aligned words,
* a word other than `unk` came out as `spn`,
* a word's syllables came from the fallback rule (its aligned phones didn't match the lexicon).

The CSV also lists how much silence was redistributed per file, and the largest gap that was filled.

`check` runs automatically after `run`. It confirms the tier order and that every
syllable boundary is a phone boundary and every word boundary is a syllable boundary.
It also confirms there is no silence inside an utterance and that the edge silences
were absorbed.

Open a few files in Praat together with the wav and listen. The acoustic model was trained on
adult Dutch, so expect some errors on child speech.

## 4. Pronunciations

`afr_g2p.py` converts Afrikaans spelling to `dutch_cv` phones with spelling rules.
The rules cover:

* vowels and diphthongs: `aai ooi oei ui ou ei y ie oe ee aa oo uu eu ê ô`
* open- and closed-syllable `a e o u`
* `g`→x, `w`→v (ʋ after a consonant), `v`→f, `ng`→ŋ
* the `-tjie`/`-djie` diminutive → `ki` (with `n`→ŋ)
* final devoicing of `b`/`d`, and diaeresis (`geëet`)

English words and fillers, such as `ball`, `butterfly`, `happy`, `balloon`, `mouse`, `uh`
and `unk`, are in a built-in table.

The lexicon that was actually used is written to:

* `work_hendrik_gt/lexicon/afrikaans_syllabified.txt` (with syllables and the source of each entry)
* `work_hendrik_gt/lexicon/afrikaans_dutchcv.dict` (the MFA dictionary)

To correct a word, add it to `dict.local` and run again. Separate syllables with ` . `:

```
babaloon    b ɑ . b ɑ . l u n
```

## 5. Syllabification rules (Afrikaans)

* Every vowel or diphthong is one nucleus. A word without a vowel (a fragment like `s`,
  `hmm`) is one syllable.
* A diaeresis starts a new syllable: `ge.eet`, `o.ë`.
* Prefixes and compound parts end a syllable: `op`, `uit`, `af`, `oor`, `aan`, `weg`, `in`,
  `ge`, `be`, `ver`, `on`, `daar`, `hier`, … For example `op.eet`, `weg.spring`, `ge.spring`,
  `hier.oor`.
* Consonants between two vowels follow Maximal Onset. The longest legal Afrikaans onset
  goes to the next syllable (`pr br tr dr kr kl kn kw fl fr gr …`), with two exceptions:
  * `s`+consonant is not an onset inside a word: `kas.te`, `vens.ters`, `vis.lyn`.
    This follows Afrikaans hyphenation.
  * `ŋ` is never an onset: `hoŋ.er`.
* Syllables never cross word boundaries, so every syllable boundary is also an MFA phone boundary.

## Evaluating against these files with `boundaries.py`

The default `textgrid_pattern="dev*/**/*.TextGrid"` expects `dev*` subfolders. Point it
straight at the new folder instead:

```python
evaluate_boundary_metrics(
    segments_dir=".../segments/af/dev",
    textgrid_dir="~/Skripsie/DATA/babaloon_alignments/alignments/af/dev/Syllable_ground_truth_hendrik_made",
    segments_pattern="*.pt", textgrid_pattern="*.TextGrid")
```

## Later: an Afrikaans acoustic model

With more Afrikaans audio (other babaloon splits, or NCHLT Afrikaans), you can replace the
Dutch model. Two options:

* Train an Afrikaans model: `mfa train <corpus> work_hendrik_gt/lexicon/afrikaans_dutchcv.dict afr_model.zip`.
  The same lexicon works, because it is just a phone set.
* Adapt the Dutch model with `--adapt` on all the data.

Then pass the new model with `--acoustic-model afr_model.zip`. Nothing else changes.

Tested with MFA 3.4.3 and `dutch_cv` v2.0.0.

## Adding syllables to existing word/phone TextGrids (e.g. the `test` set)

`afr_add_syllables.py` takes TextGrids that already have `words` and `phones` tiers. It adds a
`syllables` tier using the same Afrikaans rules, cut from the aligned phones. The output files
have the same names and go to `<folder>/gt/`. The words, phones and silences are left unchanged.
It needs only plain `python3` (no conda), with `afr_g2p.py` in the same folder.

```bash
python3 ~/Skripsie/code/afr_gt/afr_add_syllables.py ~/Skripsie/DATA/babaloon_alignments/alignments/af/test
```
__AFR_GT_EOF__
echo "    wrote: $(ls "$INSTALL_DIR" | tr '\n' ' ')"

echo "==> 2/5  Conda"
find_conda_base() {
    if command -v conda >/dev/null 2>&1; then conda info --base 2>/dev/null && return 0; fi
    for d in "$HOME/miniforge3" "$HOME/mambaforge" "$HOME/miniconda3" "$HOME/anaconda3"; do
        if [ -x "$d/bin/conda" ] && [ -f "$d/etc/profile.d/conda.sh" ]; then echo "$d"; return 0; fi
    done
    return 1
}
if CONDA_BASE="$(find_conda_base)"; then
    echo "    found conda at $CONDA_BASE"
else
    echo "    no working conda found -> installing Miniforge in ~/miniforge3"
    TMP_INST="$(mktemp -d)/Miniforge3.sh"
    download "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" "$TMP_INST"
    if [ -e "$HOME/miniforge3" ]; then
        echo "    ~/miniforge3 exists but is incomplete -> repairing it in place (-u)"
        bash "$TMP_INST" -b -u -p "$HOME/miniforge3"
    else
        bash "$TMP_INST" -b -p "$HOME/miniforge3"
    fi
    rm -f "$TMP_INST"
    CONDA_BASE="$HOME/miniforge3"
    "$CONDA_BASE/bin/conda" init bash >/dev/null   # so 'conda activate' works in new terminals
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
echo "    using conda at $CONDA_BASE"

echo "==> 3/5  Conda env '$ENV_NAME' with Montreal Forced Aligner (takes a few minutes the first time)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "    env exists"
else
    conda create -y -n "$ENV_NAME" -c conda-forge "python=3.11" montreal-forced-aligner
fi
conda activate "$ENV_NAME"
if ! command -v mfa >/dev/null 2>&1; then
    conda install -y -c conda-forge montreal-forced-aligner
fi
echo "    MFA version: $(mfa version)"

echo "==> 4/5  Dutch acoustic model"
if [ -s models/dutch_cv.zip ]; then
    echo "    already downloaded"
else
    download "$MODEL_URL" models/dutch_cv.zip
fi

echo "==> 5/5  Test"
python afr_g2p.py babaloon seuntjie geëet
python -c "import afr_gt_align" && echo "    pipeline imports OK"

cat <<EOF

=====================================================================
 Installed in $INSTALL_DIR

 Every new terminal:
     conda activate $ENV_NAME
     cd $INSTALL_DIR

 Test on one file:
     python afr_gt_align.py run --only id_3_main_post_test

 All files in ~/Skripsie/DATA/babaloon_alignments/corpora/afr/dev:
     python afr_gt_align.py run

 Add syllables to existing word/phone TextGrids (e.g. the test set):
     python3 afr_add_syllables.py ~/Skripsie/DATA/babaloon_alignments/alignments/af/test

 Output -> ~/Skripsie/DATA/babaloon_alignments/alignments/af/dev/Syllable_ground_truth_hendrik_made
 QC     -> ~/Skripsie/DATA/babaloon_alignments/work_hendrik_gt/alignment_qc.csv
=====================================================================
EOF
