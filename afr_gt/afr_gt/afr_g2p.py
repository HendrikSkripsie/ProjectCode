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
