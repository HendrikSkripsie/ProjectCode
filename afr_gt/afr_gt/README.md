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
