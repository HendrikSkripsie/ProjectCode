# Audio preprocessing: silence removal + adult removal (Afrikaans child speech)

`preprocess_audio.py` takes recordings with one adult and one child and produces audio that
contains only the child's speech. It runs three steps, and each one can be switched on or off:

| Step | What it does | Tool | Output folder |
|---|---|---|---|
| 1. VAD | finds where *anyone* is speaking (silence, noise and rustling count as non-speech) | Silero VAD | `01_vad/` |
| 2. Diarisation | works out who speaks when, decides which speaker is the child, drops the adult | pyannote (or `simple`) | `02_diarisation/` |
| 3. Render | keeps audio that is speech **and** child, then cuts (default) or mutes the rest | – | `03_final/` |

Every decision is made as time segments on the **original** timeline. Audio is only edited at the
end, so switching a step off never shifts the times of the other steps.

Neither step depends on the language: both look at sound and voices, not words, so Afrikaans
needs no special model.

## Install (CPU only)

```bash
python3 -m venv ~/venvs/preproc && source ~/venvs/preproc/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu   # CPU build, much smaller
pip install silero-vad soundfile librosa scikit-learn pyannote.audio
pip install --no-deps resemblyzer        # only needed for --diariser simple (provides the encoder weights)
```

### HuggingFace token (needed for pyannote only, once)
1. Create a free account at huggingface.co.
2. Open https://huggingface.co/pyannote/speaker-diarization-community-1 and accept the conditions.
3. Settings → Access Tokens → create a token with "read" access.
4. `export HF_TOKEN=hf_xxxxxxxx` (add it to `~/.bashrc` to make it permanent).

The model downloads on the first run and is cached after that. The script switches off
pyannote's usage telemetry (`PYANNOTE_METRICS_ENABLED=0`).

## Run

```bash
source ~/venvs/preproc/bin/activate

# one file, all steps, cut mode
python preprocess_audio.py run recordings/id_3_main_post_test.wav --out preprocessed/

# a whole folder (searched recursively)
python preprocess_audio.py run recordings/ --out preprocessed/

# switch steps off to judge each one on its own
python preprocess_audio.py run recordings/ --out exp_vad_only/  --no-diarise
python preprocess_audio.py run recordings/ --out exp_diar_only/ --no-vad
python preprocess_audio.py run recordings/ --out exp_mute/      --mode mute

# no HuggingFace token / quick comparison
python preprocess_audio.py run recordings/ --out exp_simple/ --diariser simple

# settings from a file (flags still override it)
python preprocess_audio.py run recordings/ --out exp1/ --config example_config.json
```

Diarisation is the slow step on a CPU. Its result is cached in `<file>/.cache/`, so re-running with
different VAD, output or child-selection settings doesn't diarise again. To force a fresh run, use
`--no-reuse`.

## Output, and how to check each step

```
preprocessed/
  summary.csv                     one row per file: seconds kept per step, chosen child, flags
  <file>/
    review.TextGrid               ORIGINAL timeline. Tiers: vad | speaker | role | kept
    run_config.json               exact settings used for this run
    01_vad/
      speech_segments.csv         start,end of every speech segment
      speech_only.wav             the recording with non-speech removed (cut/mute)
      removed_by_vad.wav          everything the VAD threw away (pieces 150 ms apart)
    02_diarisation/
      speaker_turns.csv / .rttm   who speaks when (RTTM is the standard diarisation format)
      speakers.json               talk time and median pitch per speaker, and why the child was chosen
      child_only.wav              the child's turns (no VAD applied)
      adult_only.wav              what was removed as adult
      overlap.wav                 both speaking at once (pyannote only; removed by default)
    03_final/
      <file>.wav                  ← THE RESULT: child speech only
      removed.wav                 everything not in the result
      time_map.csv                result time ↔ original time, for every kept chunk
      <file>.source.TextGrid      result timeline; each chunk labelled with its original time span
```

To check the result:

* **Look:** in Praat, open the *original* wav and `review.TextGrid`, select both and click
  *View & Edit*. You'll see every decision on one screen.
* **Listen:** the `removed_*.wav`, `adult_only.wav` and `removed.wav` files hold only what was cut.
  Any child speech you hear in them is an error.
* **Compare settings:** run into different `--out` folders and compare the `summary.csv` files.

## How the child is chosen

The diariser only gives anonymous labels (`SPEAKER_00`, `SPEAKER_01`). The script then measures,
for each speaker, the **talk time** and the **median pitch (F0)**, using only the regions where
that speaker talks alone.

* `--child-rule auto` (default): if the higher-pitched speaker is also the one who talks most, that's
  the child. If the cues disagree, pitch wins only when the speakers differ by at least 15%
  (`--pitch-margin`); otherwise talk time wins.
* Files are **flagged** in `summary.csv` when the cues disagree, when the pitch difference is under
  15%, or when the diariser finds more or fewer than 2 speakers.
* `--child-rule pitch` or `--child-rule talktime` use a single cue.
* If a file is wrong, list it in an overrides CSV and pass `--overrides overrides.csv`:

  ```
  file,child_speaker
  id_3_main_post_test.wav,SPEAKER_00
  ```

  The speaker labels stay the same as long as the cached diarisation is reused.

**On the test recording** (`id_3_main_post_test`), the two voices differ by only about 7% in pitch
(~196 vs ~210 Hz). Pitch alone isn't a reliable way to tell an adult woman from a 6–7-year-old,
so expect many files to be flagged. Check those first.

## Settings

| Flag | Default | Meaning |
|---|---|---|
| `--no-vad` / `--no-diarise` | both on | switch a step off (it then lets everything through) |
| `--mode` | `cut` | `cut` deletes removed audio; `mute` replaces it with silence (same length as the input) |
| `--fade-ms` | 10 | fade at every edit point, so joins don't click (a click could look like a syllable onset) |
| `--join-gap-ms` | 0 | cut mode: silence inserted between kept chunks |
| `--min-keep-ms` | 100 | drop kept pieces shorter than this |
| `--vad-threshold` | 0.5 | speech probability needed; lower keeps more (and more noise) |
| `--min-speech-ms` | 250 | ignore speech bursts shorter than this |
| `--min-silence-ms` | 300 | only cut pauses longer than this (pauses inside a sentence are kept) |
| `--speech-pad-ms` | 100 | margin kept around each speech segment so word edges aren't clipped |
| `--diariser` | `pyannote` | `pyannote` (best, detects overlap) or `simple` (no token, no overlap detection) |
| `--num-speakers` | 2 | 0 lets the diariser decide |
| `--keep-overlap` | off | keep regions where child and adult talk at the same time |
| `--child-rule` / `--pitch-margin` | `auto` / 0.15 | see above |
| `--overrides` | – | manual child speaker per file |

## Mapping TextGrids back to the original recording

You'll make TextGrids on the output audio (`03_final/<file>.wav`). To view them on the original
recording, map them back:

```bash
python preprocess_audio.py map-textgrid my_alignment.TextGrid \
       preprocessed/<file>/03_final/time_map.csv my_alignment.original.TextGrid --original-duration 225.675
```

In mute mode the timelines are the same, so no mapping is needed.

## Measuring accuracy against ground truth

`evaluate_diarisation.py` compares `preprocess_audio.py`'s output with ground-truth TextGrids that
have an `ADULT` and a `CHILD` tier. Files are matched by name (`id_3_main_post_test.TextGrid` ↔
`preprocessed/id_3_main_post_test/`).

```bash
# 1. preprocess the recordings that have ground truth
python preprocess_audio.py run recordings/ --out preprocessed/
# 2. score them
python evaluate_diarisation.py --gt groundtruths/textgrid/afr --pred preprocessed/ --out evaluation/
```

| Metric | Meaning |
|---|---|
| VAD recall / precision | share of true speech the VAD kept / share of VAD speech that really is speech |
| **DER** | Diarisation Error Rate = (missed + false alarm + confusion) ÷ true speech, using the best match of SPEAKER_xx to ADULT/CHILD. Standard metric; lower is better |
| role error rate | the same, but with the program's own child/adult decision |
| child picked correctly | share of files where the speaker chosen as the child really is the child |
| **child recall / precision / F1** | of the final output: share of the child's speech kept / share of the output that is the child |
| adult / non-speech kept, child lost | the output's errors, in seconds |

Outputs: `per_file.csv` (one row per recording), `overall.json` (totals, summed over all files) and
`error_textgrids/<file>.errors.TextGrid`. Open the error TextGrid with the original wav in Praat to
see exactly where the errors are (tiers `diarisation_errors` and `output_errors`).

`--collar 0.25` ignores 0.25 s around every ground-truth boundary, which is the standard (NIST)
way of reporting DER. Report both: without a collar it's strict about exact boundary placement,
and with a collar it measures mainly whether the right person was found.

Only the annotated part of each recording is scored (0 up to the ground truth's `xmax`). The script
prints a note when the annotation stops well before the end of the audio.

## The `simple` diariser

This backend computes a speaker embedding (Resemblyzer's GE2E encoder) for 1.5 s windows every
0.5 s inside the VAD speech, and clusters them into 2 groups. Each 10 ms frame takes the majority
label of the windows covering it. It needs no token and takes seconds, but it can't detect
overlapping speech and its boundaries are coarser (about 0.5 s). It also always uses Silero VAD
internally to find speech, even when step 1 is switched off.
