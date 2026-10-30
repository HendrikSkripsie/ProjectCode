#!/usr/bin/env bash
# Syllable-count scatter plots (ground truth vs predicted) + Pearson's r
# for ZeroSyl and the stupid baseline, on all four project datasets.
#
# Run from ~/ZeroSyl-master:
#
#   bash run_count_plots.sh baselines        # 1. make the stupid baselines (once)
#   bash run_count_plots.sh                  # 2. plot all four datasets
#   bash run_count_plots.sh conrad           #    or just one
#   bash run_count_plots.sh babaloon fleurs  #    or a few
#
# Dataset names: babaloon  conrad  fleurs  librispeech
#
# Plots: output/plots/<dataset>_ZeroSyl_syllable_counts.png
#        output/plots/<dataset>_StupidBaseline_syllable_counts.png
# Pearson's r for everything is also collected in output/results/pearson_summary.txt

cd "$(dirname "$0")"
mkdir -p output/plots output/results
SUMMARY=output/results/pearson_summary.txt

# ---------------------------------------------------------------------------
# dataset paths:   name           ground truth                      ZeroSyl output
# ---------------------------------------------------------------------------
GT_babaloon=data/babaloonAlignments/dev/;       ZS_babaloon=output/segments/ZeroSylCollapsed/Babaloon/dev/
GT_conrad=data/ConradConcatAlignments/;         ZS_conrad=output/segments/ZeroSylCollapsed/ConradConcat/
GT_fleurs=data/mergedFleursAlignments/dev/;     ZS_fleurs=output/segments/ZeroSylCollapsed/mergedFleurs/dev/
GT_librispeech=data/alignments/LibriSpeech/;    ZS_librispeech=output/segments/ZeroSylCollapsed/LibriSpeech/

# stupid baseline lives at the same path with ZeroSylCollapsed -> StupidBaseline/ZeroSylCollapsed
baseline_of() { echo "${1/ZeroSylCollapsed/StupidBaseline/ZeroSylCollapsed}"; }

make_baselines() {
    # stupid_baseline.py takes the dataset folder (one level above dev/)
    python stupid_baseline.py output/segments/ZeroSylCollapsed/Babaloon
    python stupid_baseline.py output/segments/ZeroSylCollapsed/ConradConcat
    python stupid_baseline.py output/segments/ZeroSylCollapsed/mergedFleurs
    python stupid_baseline.py output/segments/ZeroSylCollapsed/LibriSpeech
}

plot_one() {  # plot_one <label> <ground truth> <prediction dir> <out name>
    echo "================ $1 ================"
    python syllable_count_plot.py "$2" "$3" --out "output/plots/$4_syllable_counts.png" \
        | tee /dev/stderr | grep "Pearson" | sed "s/^/$1: /" >> "$SUMMARY"
}

plot_dataset() {  # plot_dataset <name>
    local gt_var="GT_$1" zs_var="ZS_$1"
    local gt="${!gt_var}" zs="${!zs_var}"
    if [ -z "$gt" ]; then
        echo "Unknown dataset '$1' (use: babaloon conrad fleurs librispeech)"; return
    fi
    plot_one "$1 ZeroSyl" "$gt" "$zs" "$1_ZeroSyl"

    local sb; sb=$(baseline_of "$zs")
    if [ -d "$sb" ]; then
        plot_one "$1 StupidBaseline" "$gt" "$sb" "$1_StupidBaseline"
    else
        echo "(no stupid baseline at $sb - run: bash run_count_plots.sh baselines)"
    fi
    echo
}

if [ "$1" = "baselines" ]; then
    make_baselines
    exit
fi

[ $# -eq 0 ] && set -- babaloon conrad fleurs librispeech
: > "$SUMMARY"
for d in "$@"; do
    plot_dataset "$d"
done

echo "================ Pearson's r summary ================"
cat "$SUMMARY"
