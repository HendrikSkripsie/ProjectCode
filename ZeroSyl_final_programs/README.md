# ZeroSyl evaluation tools

Extra programs for analysing ZeroSyl syllable-boundary predictions:
stupid baselines, syllable-count plots, a prominence (threshold) sweep, and a tool that
estimates the number of syllables in a file from ZeroSyl output alone.

None of these programs change ZeroSyl itself (`zerosyl/`) or `reproduce_results.sh`.
They only read ZeroSyl's output (`.pt` files) and your alignments (`.TextGrid` files).

---

## Contents

| File | Type | Purpose |
|---|---|---|
| [`stupid_baseline.py`](#stupid_baselinepy) | program | Baseline: boundaries at ZeroSyl's average rate |
| [`fixed_stupid_baseline.py`](#fixed_stupid_baselinepy) | program | Baseline: a boundary every 50 / 100 / 200 / 300 / 400 / 500 ms |
| [`syllable_count_plot.py`](#syllable_count_plotpy) | program | Ground truth vs predicted syllables per file + Pearson's r |
| [`prominence_sweep.py`](#prominence_sweeppy) | program | Re-runs ZeroSyl for prominence 0.00 → 1.00 and scores every run |
| [`syllable_estimator.py`](#syllable_estimatorpy) | program | Estimates syllables per file without ground truth (5 estimators) |
| [`devfit_plots.py`](#devfit_plotspy) | program | Dev regression line applied to test, with R² and MSE |
| `run_count_plots.sh` | runner | Count plots for Babaloon, ConradConcat, mergedFleurs, LibriSpeech |
| `run_fixed_baseline_babaloon.sh` | runner | Fixed-rate baselines on Babaloon + their evaluation |
| `run_prominence_sweep_babaloon.sh` | runner | Prominence sweep on Babaloon |
| `run_estimator_babaloon.sh` | runner | Syllable estimator (dev → test) + 300 ms test baseline |
| `run_devfit_plots_babaloon.sh` | runner | Dev-fit plots for ZeroSyl and the 300 ms baseline |

---

## Setup

1. Copy **all files** into the root of the ZeroSyl repository:

   ```
   ~/ZeroSyl-master/
   ├── zerosyl/                 (ZeroSyl package, unchanged)
   ├── data/                    (waveforms + alignments)
   ├── output/                  (ZeroSyl output + everything these tools create)
   ├── reproduce_results.sh
   ├── stupid_baseline.py       ← these files
   ├── fixed_stupid_baseline.py
   ├── ...
   └── run_devfit_plots_babaloon.sh
   ```

2. Always run from that folder, inside the ZeroSyl environment:

   ```bash
   conda activate zerosyl
   cd ~/ZeroSyl-master
   ```

3. Requirements: everything ZeroSyl already needs (`torch`, `numpy`, `scipy`, `tgt`,
   `faiss`, `torchcodec`, `rich`) plus `matplotlib`. No GPU is needed; everything runs on CPU.

---

## Recommended workflow

```bash
# 0. Encode + evaluate a dataset with ZeroSyl (your existing reproduce_results.sh blocks)
bash reproduce_results.sh

# 1. Baselines
bash run_count_plots.sh baselines            # average-rate stupid baseline, all datasets
bash run_fixed_baseline_babaloon.sh          # fixed-rate baselines on Babaloon dev

# 2. How well do predicted counts follow the true counts?
bash run_count_plots.sh                      # scatter plots + Pearson's r

# 3. Does the boundary threshold matter?
bash run_prominence_sweep_babaloon.sh test   # quick check on one file first
bash run_prominence_sweep_babaloon.sh        # full Babaloon dev sweep

# 4. Can ZeroSyl estimate syllable counts on unseen data? (needs Babaloon test encoded)
bash run_estimator_babaloon.sh all
bash run_devfit_plots_babaloon.sh
```

---

## Shared conventions

These apply to every program, so results are comparable between them.

| Thing | Convention |
|---|---|
| ZeroSyl output | `.pt` file per audio file: tensor `(N, 3)` = `[start_frame, end_frame, id]` |
| Frame rate | 50 Hz (one frame = 20 ms) |
| Ground-truth syllables | non-empty intervals in the `syllables` tier of the TextGrid |
| Predicted syllables | ZeroSyl segments **excluding silence**. Silence is the largest id (9115 for ZeroSylCollapsed), detected automatically |
| Baseline syllables | every segment (all baseline segments share dummy id 0, so nothing is removed) |
| File matching | ground truth and prediction are paired by file name (`id_3_main_post_test.TextGrid` ↔ `id_3_main_post_test.pt`) |
| Boundary metrics | always computed with ZeroSyl's own `zerosyl/eval/boundaries.py` (tolerance 0.05 s, no constant shift) |

Baseline outputs use exactly the same folder layout and file format as ZeroSyl output,
so **every ZeroSyl evaluation command also works on them**:

```bash
zerosyl evaluate boundaries <baseline folder> <alignments folder> \
    --segments-pattern "*.pt" --textgrid-pattern "*.TextGrid"
```

---

## Programs

### `stupid_baseline.py`

A deliberately naive baseline: it measures ZeroSyl's **average** segment rate over a whole
dataset and places boundaries at that constant rate in every file. Same number of boundaries
as ZeroSyl, but none of them follow the speech.

```bash
python stupid_baseline.py output/segments/ZeroSylCollapsed/Babaloon       # one dataset
python stupid_baseline.py output/segments/ZeroSylCollapsed --all          # every dataset
```

| Option | Meaning |
|---|---|
| `--all` | treat the input as a model folder and process every dataset in it |
| `--scale 1.5` | longer interval (fewer boundaries); `< 1` gives more |
| `--textgrid` | also write TextGrids of the dummy boundaries (to view in Praat) |
| `--pattern` | which `.pt` files to use (default `**/*.pt`, i.e. all splits) |

**Output:** `output/segments/StupidBaseline/ZeroSylCollapsed/<dataset>/...`
plus `<dataset>_baseline_info.json` (measured rate and chosen interval).

> The rate is averaged over every split in the dataset folder (e.g. dev and test together).

---

### `fixed_stupid_baseline.py`

A baseline with a **fixed** interval, independent of ZeroSyl. ZeroSyl's output is only used
to know how long each file is.

```bash
python fixed_stupid_baseline.py output/segments/ZeroSylCollapsed/Babaloon                  # all six rates
python fixed_stupid_baseline.py output/segments/ZeroSylCollapsed/Babaloon --rates 300      # one rate
python fixed_stupid_baseline.py output/segments/ZeroSylCollapsed/Babaloon --rates 300 --pattern "test/*.pt"
```

**Output:** `output/segments/FixedStupidBaseline/<N>ms/<dataset>/<split>/...`

> 50 ms is 2.5 frames, so its segments alternate between 40 and 60 ms (average exactly 50 ms).
> Very short intervals give recall ≈ 100 % but a very negative R-value; that is expected.

**Runner:** `run_fixed_baseline_babaloon.sh` creates all six rates for Babaloon dev, then runs
`zerosyl evaluate boundaries` and `syllable_count_plot.py` for each.

```bash
bash run_fixed_baseline_babaloon.sh          # create + evaluate
bash run_fixed_baseline_babaloon.sh eval     # evaluate only
```

Results: `output/results/FixedStupidBaseline_Babaloon.txt`,
plots: `output/plots/Babaloon_FixedStupidBaseline_<N>ms_syllable_counts.png`.

---

### `syllable_count_plot.py`

Scatter plot with one dot per file: **x = ground-truth syllables**, **y = predicted syllables**,
with the y = x line, a best-fit line and **Pearson's r** under the graph.

```bash
python syllable_count_plot.py <alignments folder> <prediction folder>
python syllable_count_plot.py data/babaloonAlignments/dev/ output/segments/ZeroSylCollapsed/Babaloon/dev/
```

| Option | Meaning |
|---|---|
| `--out file.png` | where to save (default `output/plots/<model>_<dataset>_<split>_syllable_counts.png`) |
| `--keep-silence` | count ZeroSyl's silence segments as syllables too |
| `--sil-id N` | set the silence id manually |
| `--tier` | TextGrid tier to count (default `syllables`) |

Files present in only one folder are listed as a warning and left out.

**Runner:** `run_count_plots.sh` covers the four project datasets
(`babaloon`, `conrad`, `fleurs`, `librispeech`) for ZeroSyl and the stupid baseline.

```bash
bash run_count_plots.sh baselines         # create the stupid baselines (once)
bash run_count_plots.sh                   # all four datasets
bash run_count_plots.sh conrad fleurs     # selected datasets
```

Plots: `output/plots/<dataset>_ZeroSyl_syllable_counts.png` and `..._StupidBaseline_...`;
all Pearson's r values: `output/results/pearson_summary.txt`.

> The LibriSpeech paths at the top of `run_count_plots.sh` (`GT_librispeech`, `ZS_librispeech`)
> are a best guess. Edit them if that dataset reports "no files found".

---

### `prominence_sweep.py`

ZeroSyl places a boundary at every peak of its (smoothed) feature-norm curve whose
**prominence** is at least 0.45. This program re-runs ZeroSyl for prominence
0.00, 0.05, …, 1.00 and evaluates every run.

WavLM (the slow part) runs **once per file**. Only the steps after it are repeated per
prominence, so the whole sweep costs about as much as one normal `zerosyl encode`.
Before saving anything, it checks that its output at 0.45 is identical to `zerosyl encode`
and stops if not.

```bash
python prominence_sweep.py data/babaloonWaveforms/dev/ data/babaloonAlignments/dev/ --name Babaloon
```

| Option | Meaning |
|---|---|
| `--name` | label for output folders (required) |
| `--start / --stop / --step` | prominence range (default 0.0 / 1.0 / 0.05) |
| `--discrete` | also save discrete outputs for `zerosyl evaluate clustering` |
| `--skip-encode` / `--skip-evaluate` | run only one half |
| `--overwrite` | re-encode files already done (default: an interrupted run resumes) |
| `--threads N` | CPU threads for WavLM |
| `--segments-pattern`, `--textgrid-pattern` | as in `zerosyl evaluate boundaries` (use `"**/*.pt"` for nested folders) |
| `--constant-shift`, `--tolerance` | evaluation settings (default 0.0 / 0.05) |

**Output**

| Path | Content |
|---|---|
| `output/segments/ProminenceSweep/<name>/prom_X.XX/` | ZeroSyl output for each prominence |
| `output/results/ProminenceSweep_<name>.txt` / `.csv` | Precision, Recall, F1, OS, R-value, token scores and segments/s per prominence |
| `output/plots/ProminenceSweep_<name>.png` | scores vs prominence |

> **Sanity check:** the 0.45 row must equal your normal Babaloon dev result.

**Runner**

```bash
bash run_prominence_sweep_babaloon.sh test   # devSMALL, quick check
bash run_prominence_sweep_babaloon.sh        # Babaloon dev
bash run_prominence_sweep_babaloon.sh eval   # re-evaluate only
```

---

### `syllable_estimator.py`

Estimates how many syllables a file contains **from ZeroSyl output alone**. A correction
is learned on dev (where ground truth exists) and applied to new data.

| Step | Command | Needs ground truth? |
|---|---|---|
| fit | `python syllable_estimator.py fit <gt dev> <pred dev> --name Babaloon` | yes (dev) |
| predict | `python syllable_estimator.py predict <pred folder> --name Babaloon` | **no** |
| evaluate | `python syllable_estimator.py evaluate <gt test> <pred test> --name Babaloon` | yes (test, only for scoring) |

**Estimators** (all learned on dev only):

| Name | Formula | Idea |
|---|---|---|
| `raw` | ZeroSyl count | no correction |
| `inverse` | (ZS − c) / m | your dev plot line, flipped |
| `direct` | a·ZS + b | least-squares fit of ground truth on ZeroSyl (**default**) |
| `rate` | (a·ZS/s + b) · seconds | correction of syllables per second |
| `duration` | dev syll/s · seconds | ignores ZeroSyl, uses only file length (reference) |

`predict --estimator rate` selects a different estimator.

**Output**

| Path | Content |
|---|---|
| `output/calibration/<name>.json` | learned correction |
| `output/estimates/<name>_<split>.csv` | estimated syllables per file and per second |
| `output/results/SyllableEstimator_<name>_<split>.txt` | MAE (+95 % CI), RMSE, bias, MAPE, total error, r, per-second error; and whether each estimator beats `duration` |
| `..._perfile.csv`, `output/plots/SyllableEstimator_<name>_<split>.png` | per-file numbers and plots |

> The `fit` step prints the dev line `ZS = m·GT + c`; it should match your
> `syllable_count_plot.py` dev plot (≈ 1.06x + 11.8 for Babaloon).

**Runner**

```bash
bash run_estimator_babaloon.sh             # fit on dev, predict + evaluate on test
bash run_estimator_babaloon.sh baseline    # 300 ms fixed baseline on test + evaluation
bash run_estimator_babaloon.sh all         # both
```

---

### `devfit_plots.py`

The figures for the generalisation question, with **ground truth on the y axis** and the
**predicted count on the x axis**.

- **Dev plot:** dots = dev files; orange line = least-squares fit `GT = a·predicted + b`.
  This line *is* the estimator: read a count on the x axis and go up to the line.
- **Test plot:** dots = test files (ground truth "hidden" in real use); orange line = the
  **same** dev line, not refitted; faint grey line = the best line this test set could
  ideally have. A dot's vertical distance from the orange line is its estimation error.

```bash
# dev only
python devfit_plots.py <gt dev> <pred dev> --label ZeroSyl

# dev + test
python devfit_plots.py <gt dev> <pred dev> --test <gt test> <pred test> --label ZeroSyl
```

| Option | Meaning |
|---|---|
| `--label` | system name for titles and file names (e.g. `ZeroSyl`, `300ms`) |
| `--dataset` | dataset name in file names (default `Babaloon`) |
| `--outdir` | output folder (default `output/plots/DevFit/`) |

**Reported numbers**

| Number | Meaning |
|---|---|
| **R² (dev line on test)** | share of the test files' variation explained by the dev line. 1 = perfect, 0 = no better than guessing the test average, < 0 = worse. **Main generalisation number.** |
| R² / MSE ideal test line | the best any straight line could do on these test files |
| R² / MSE dev line on dev | how well the line fits the files it was learned on |
| MSE, RMSE, MAE | squared / root-squared / absolute error in syllables per file |
| Pearson's r | strength of the linear relationship (not changed by the correction) |

**Runner:** `run_devfit_plots_babaloon.sh` (also creates the 300 ms baseline for dev and test)

```bash
bash run_devfit_plots_babaloon.sh
```

| Output in `output/plots/DevFit/` | Shows |
|---|---|
| `Babaloon_dev_ZeroSyl_devfit.png` | ZeroSyl dev fit |
| `Babaloon_test_ZeroSyl_devfit.png` + `.csv` | ZeroSyl dev line on test |
| `Babaloon_dev_300ms_devfit.png` | 300 ms dev fit |
| `Babaloon_test_300ms_devfit.png` + `.csv` | 300 ms dev line on test |

---

## Where everything is saved

```
output/
├── segments/
│   ├── ZeroSylCollapsed/...               ZeroSyl (from zerosyl encode)
│   ├── StupidBaseline/...                 stupid_baseline.py
│   ├── FixedStupidBaseline/<N>ms/...      fixed_stupid_baseline.py
│   ├── ProminenceSweep/<name>/prom_X.XX/  prominence_sweep.py
│   └── ProminenceSweepDiscrete/...        prominence_sweep.py --discrete
├── calibration/<name>.json                syllable_estimator.py fit
├── estimates/<name>_<split>.csv           syllable_estimator.py predict
├── results/                               tables (.txt / .csv) of every evaluation
└── plots/
    ├── *_syllable_counts.png              syllable_count_plot.py
    ├── ProminenceSweep_<name>.png         prominence_sweep.py
    ├── SyllableEstimator_*.png            syllable_estimator.py
    └── DevFit/                            devfit_plots.py
```

---

## Metrics glossary

| Metric | Meaning |
|---|---|
| Precision | share of predicted boundaries within 50 ms of a true boundary |
| Recall | share of true boundaries that were found |
| F1 | balance of precision and recall |
| OS | over-segmentation: + = too many boundaries, − = too few |
| R-value | combines recall and over-segmentation; heavily penalises over-segmentation (can be very negative) |
| Token P / R / F1 | same, but a whole syllable counts only if *both* its boundaries are correct |
| Pearson's r | how linearly predicted counts follow true counts (−1 … 1) |
| MAE | average absolute error in syllables per file |
| MSE / RMSE | average squared error / its square root (in syllables) |
| R² | share of variation explained; on test it measures generalisation |
| bias | average signed error: + overcounts, − undercounts |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `No .pt files found` / `No .TextGrid files found` | check the folder path; run from `~/ZeroSyl-master` |
| `AssertionError` in `zerosyl evaluate boundaries` | ground-truth and prediction folders don't contain the same files |
| warning "file counts don't line up" | some files exist in only one folder; they are skipped |
| 300 ms or test plots missing | encode Babaloon test first (test block in `reproduce_results.sh`) |
| prominence sweep interrupted | run the same command again; finished files are skipped |
| `ModuleNotFoundError: zerosyl` | activate the `zerosyl` environment and run from `~/ZeroSyl-master` |
