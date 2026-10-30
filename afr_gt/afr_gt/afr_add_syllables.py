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
