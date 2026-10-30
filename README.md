# Syllable counting and diarisation of child audio using machine learning

Code for the final-year project (skripsie) of **Hendrik de Man** (Electrical and Electronic
Engineering, Stellenbosch University, 2026), supervised by Prof Herman Kamper.

The project evaluates **ZeroSyl**, a syllable segmentation method based on the self-supervised
speech model WavLM, on recordings of Afrikaans-speaking children aged four to five. The pipeline
first extracts the child's speech from a recording of a storytelling assessment, then applies
ZeroSyl, and finally turns its output into a syllable count for each recording.

> **The data is not included.** See [Data](#data) below.

---

## Built on ZeroSyl

This code does **not** reimplement ZeroSyl. It runs on top of the official ZeroSyl repository by
Visser, Malan, Slabbert and Kamper:

* Code: <https://github.com/nicolvisser/ZeroSyl>
* Paper: N. Visser, S. Malan, D. Slabbert and H. Kamper, "ZeroSyl: Simple zero-resource syllable
  tokenization for spoken language modeling," in *Proc. Interspeech*, 2026.
  [arXiv:2602.15537](https://arxiv.org/abs/2602.15537)

The boundary evaluation (precision, recall, F1, over segmentation and R value) is the one from the
ZeroSyl repository, so results can be compared directly with the published LibriSpeech results.

To use the tools in `ZeroSyl_final_programs/`, first clone and install ZeroSyl, then copy those
files into the root of the ZeroSyl repository (see `ZeroSyl_final_programs/README.md`):

```bash
git clone https://github.com/nicolvisser/ZeroSyl.git
cd ZeroSyl
# install ZeroSyl as described in its own README, then:
cp /path/to/this/repo/ZeroSyl_final_programs/* .
```

---

## Repository layout (in pipeline order)

| Step | Folder | What it does | Report |
|---|---|---|---|
| 1. Extract the child's speech | `programs/2_preprocessing/` | `preprocess_audio.py`: voice activity detection (Silero VAD), speaker diarisation (pyannote) and pitch based child selection (pYIN). Keeps only the child's speech. `evaluate_diarisation.py`: DER, child recall, precision and F1 against reference annotations. | Sections 3.1.1, 3.5.1; Chapter 4 |
| 2a. Reference syllables, Afrikaans (Babaloon) | `afr_gt/` | Forced alignment with the Montreal Forced Aligner (Dutch CV acoustic model) and rule based Afrikaans syllabification, giving `words`, `phones` and `syllables` TextGrid tiers. | Section 3.2 |
| 2b. Reference syllables, English child (Nexdata) | `programs/1_alignment/` | The same for English child speech: MFA alignment (`english_mfa`, `english_uk_mfa`) and syllabification with the maximal onset principle. | Section 3.2 |
| 2c. Prepare the English child data | `programs/3_concatenation/` | Joins each child's short recordings into one file, stitches their TextGrids together, and shortens long silences. | Section 3.3.4 |
| 3. Evaluate ZeroSyl | `ZeroSyl_final_programs/` | Fixed interval baselines (`fixed_stupid_baseline.py`, `stupid_baseline.py`), the prominence threshold sweep (`prominence_sweep.py`) and the syllable count estimator (`syllable_estimator.py`). The `run_*.sh` scripts reproduce the Babaloon results. | Chapters 5 and 6 |
| 3b. Manual annotations | `Boundary Eval/` | Notebooks that score ZeroSyl against a manually annotated TextGrid of a single recording and show the boundaries next to each other. | Section 5.2 |
| 4. Plots | `ZeroSyl_final_programs/` | `syllable_count_plot.py`: predicted against reference syllable counts with Pearson's r. `devfit_plots.py`: the count estimator fitted on the development set and applied to the test set (R², MAE, RMSE). | Sections 5.4 and 5.5 |

Each folder has its own `README.md` with setup instructions and example commands.

---

## Data

No audio or annotations are included in this repository.

| Dataset | Content | Availability |
|---|---|---|
| **Babaloon** | Afrikaans child speech (ages 4 to 5), storytelling assessments | **Not public.** These are research recordings of children and cannot be shared. |
| Nexdata | English child speech (ages 6 to 7), read speech | Commercial dataset (Nexdata). |
| LibriSpeech | English adult read speech | Public: <https://www.openslr.org/12> |
| FLEURS | Afrikaans adult read speech | Public, via Google / HuggingFace. |

The scripts expect the data in a `data/` folder; the paths are set at the top of each script
and in the `run_*.sh` files.

---

## Setup

The project used three separate conda environments, because the tools need conflicting
package versions:

| Environment | Used for | Setup |
|---|---|---|
| `zerosyl` | ZeroSyl and everything in `ZeroSyl_final_programs/` | ZeroSyl's own instructions, plus `pip install matplotlib` |
| `aligner` | Montreal Forced Aligner (`afr_gt/`, `programs/1_alignment/`) | `bash programs/1_alignment/setup_mfa.sh` or `bash afr_gt/install_afr_gt.sh` |
| `preproc` | Child speech extraction (`programs/2_preprocessing/`) | See `programs/2_preprocessing/README.md` |

Everything runs on a CPU; a GPU only makes it faster.

### HuggingFace token (needed for speaker diarisation)

The pyannote diarisation model is free but gated, so it needs a HuggingFace access token.
**Never write the token into a script or commit it to Git.** Set it as an environment variable
instead:

1. Create a free account at <https://huggingface.co>.
2. Open <https://huggingface.co/pyannote/speaker-diarization-community-1> and accept the conditions.
3. Go to Settings → Access Tokens (<https://hf.co/settings/tokens>) and create a token with
   **read** access.
4. Set it in your terminal:

   ```bash
   export HF_TOKEN=your_token_here
   ```

   To keep it for future terminals, add that line to `~/.bashrc` and run `source ~/.bashrc`.

`preprocess_audio.py` reads `HF_TOKEN` automatically. It can also be passed once with
`--hf-token`, but the environment variable is safer because it does not end up in your
shell history or in output files.

---

## Acknowledgements

* **ZeroSyl** and its evaluation code: Visser, Malan, Slabbert and Kamper (see above).
* Simon Malan, for explaining how the boundary evaluation code works.
* Prof Herman Kamper, for supervision, the oscillator syllabifier implementation and the Babaloon data.
* Danél Slabbert and Conrad Joubert, for providing datasets.
* Tools: [pyannote.audio](https://github.com/pyannote/pyannote-audio),
  [Silero VAD](https://github.com/snakers4/silero-vad),
  [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io),
  [Praat](https://www.praat.org).
