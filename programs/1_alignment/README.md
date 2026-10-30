# Forced alignment for Conrad's British English child speech → Zerosyl

`zerosyl_align.py` reads `manifest.csv`, runs the Montreal Forced Aligner (MFA) and writes one
Praat TextGrid per wav with three tiers, in this order:

| tier        | source                                                                 |
|-------------|------------------------------------------------------------------------|
| `words`     | MFA word alignment                                                     |
| `phones`    | MFA phone alignment (IPA, MFA English phone set)                       |
| `syllables` | each word's phones grouped into syllables (Maximal Onset Principle)    |

Silence is an empty interval (`""`), the same layout as the reference TextGrid.

## 1. One-time setup

```bash
cd ~/Skripsie/code            # wherever you put these files
bash setup_mfa.sh             # creates conda env "aligner", downloads the models
```

Models used (all from the official MFA model repository; they share one IPA phone set):

* **Dictionary: `english_uk_mfa`**: British English pronunciations (~46k words). This is the
  British dictionary for this data. The CMU dictionary is American (rhotic, US vowels)
  and uses ARPAbet, which doesn't match the English MFA acoustic model's phone set.
* **Acoustic model: `english_mfa`**: trained on many English varieties, including British English.
* **G2P model: `english_uk_mfa`**: generates British pronunciations for words missing from the dictionary.

## 2. Run it

```bash
conda activate aligner
DATA=~/Skripsie/DATA/Conrad_english/british_english/british_english

# one file
python zerosyl_align.py one $DATA/16kHz_Mono/G0001/G0001S0001.wav \
       --manifest $DATA/manifest.csv --out textgrids/

# a few sessions (good first test)
python zerosyl_align.py batch --audio-root $DATA/16kHz_Mono --manifest $DATA/manifest.csv \
       --sessions G0001 G0002 --out textgrids/

# whole corpus (recommended for final results), adapting the model to children's voices
python zerosyl_align.py batch --audio-root $DATA/16kHz_Mono --manifest $DATA/manifest.csv \
       --out textgrids/ --adapt
```

Output: `textgrids/<session>/<utterance>.TextGrid` and `textgrids/alignment_qc.csv`.

**Use `batch` for real results, even for a single session.** `batch` treats each session folder
(one child) as a speaker and adapts the model to that child's voice (fMLLR). `--adapt` also
retrains the adult model on all 13k child utterances before aligning. `one` can't do
either, so it's best kept for quick checks.

## 3. Pronunciation dictionary and `dict.local`

The first run builds `mfa_work/lexicon/english_uk_children.dict` =
`english_uk_mfa` + your `dict.local` + spelled-out acronyms (KFC, PTA, GP, SMS, PVA)
+ G2P pronunciations for the other ~100 words not in the dictionary (`dunnit`, `onesie`,
`collywobbles`, `crowthorne`, `swinley`, possessives like `mum's`, ...).

Every generated pronunciation is listed in `mfa_work/lexicon/generated_pronunciations.txt`.
To correct one, put the corrected line in a `dict.local` file in the folder you run from (or pass
`--dict-local PATH`). `dict.local` entries replace the dictionary's for that word:

```
dunnit	d ɐ n ɪ t
swinley	s w ɪ n ʎ i
```

Phones **must** be from the MFA English set (the script stops if they aren't), e.g.
`a aj aw b bʲ c cʰ d dʒ dʲ ej f h i iː j k kʰ l m n p pʰ s t tʃ tʰ tʲ v w z ð ŋ ɐ ɑː ɒ ɒː ɔj ə əw ɛ ɛː ɜː ɡ ɪ ɫ ɫ̩ ɲ ɹ ʃ ʉː ʊ ʎ ʒ ʔ θ`
(look up similar words in the generated lexicon to copy their phones).

## 4. Check the alignments

`alignment_qc.csv` has one row per utterance. A row is marked `CHECK` when:

* the number of aligned words ≠ the number of transcript words,
* a word was aligned as spoken noise (`spn`),
* the first word starts or the last word ends more than 0.3 s (`--tolerance`) away from the
  manifest's `start_ms` / `end_ms`.

A row is marked `NOT_ALIGNED` when MFA gave up on the file. Re-run with wider beams
(`--beam 200 --retry-beam 800`). This usually means the child said something other than the prompt.
Open a few `CHECK` files in Praat together with the wav.

## 5. How the syllable tier is made

For each word, every vowel or syllabic consonant (`n̩ ɫ̩ m̩`) is one syllable nucleus. The consonants
between two nuclei go to the next syllable as long as they form a legal English onset (`st`, `pl`,
`stɹ`, ...). The remaining consonants close the previous syllable (Maximal Onset Principle). So
`extra` → `ɛk.stɹə` and `potato` → `pʰə.tʰej.tʰəw`. Syllables never cross word boundaries, so every
syllable boundary is also an MFA phone boundary. The label is the syllable's phones joined together.

Already have MFA TextGrids? `python zerosyl_align.py syllabify mfa_output/ textgrids/` adds the
syllable tier without re-aligning.

## Options

```
one   WAV --manifest M [--out DIR|FILE] [--text "custom transcript"]
batch --audio-root DIR --manifest M [--out DIR] [--sessions G0001 ...] [--limit N]
      [-j JOBS] [--adapt] [--flat] [--copy-audio]
common: [--dict-local F] [--rebuild-lexicon] [--work-dir mfa_work]
        [--beam 100] [--retry-beam 400] [--acoustic-model NAME|ZIP] [--tolerance 0.3]
```
