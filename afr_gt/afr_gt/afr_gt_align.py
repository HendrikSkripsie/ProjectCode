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
