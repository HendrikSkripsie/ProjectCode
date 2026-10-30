#!/usr/bin/env bash
# Fixed-rate stupid baseline on Babaloon: generate, then evaluate every rate.
# Run from ~/ZeroSyl-master (inside the zerosyl conda env):
#
#   bash run_fixed_baseline_babaloon.sh            # generate + evaluate all rates
#   bash run_fixed_baseline_babaloon.sh eval       # evaluate only (already generated)
#
# Boundary results are printed and also saved to
#   output/results/FixedStupidBaseline_Babaloon.txt
# Count plots are saved to output/plots/Babaloon_FixedStupidBaseline_<N>ms_syllable_counts.png

cd "$(dirname "$0")"

RATES="50 100 200 300 400 500"
RESULTS=output/results/FixedStupidBaseline_Babaloon.txt
mkdir -p output/results output/plots

# ---- 1. generate the baselines ----
if [ "$1" != "eval" ]; then
    python fixed_stupid_baseline.py output/segments/ZeroSylCollapsed/Babaloon --rates $RATES
fi

# ---- 2. evaluate each rate ----
: > "$RESULTS"
for ms in $RATES; do
    echo "================ Babaloon fixed baseline: ${ms} ms ================" | tee -a "$RESULTS"

    # boundary metrics (same command as for ZeroSyl, different segments folder)
    zerosyl evaluate boundaries \
        output/segments/FixedStupidBaseline/${ms}ms/Babaloon/dev/ \
        data/babaloonAlignments/dev/ \
        --segments-pattern "*.pt" \
        --textgrid-pattern "*.TextGrid" 2>&1 | grep -v -e "INFO:" -e "constant shift" -e "Calculating" \
        | tee -a "$RESULTS"

    # syllable count scatter plot + Pearson's r
    python syllable_count_plot.py \
        data/babaloonAlignments/dev/ \
        output/segments/FixedStupidBaseline/${ms}ms/Babaloon/dev/ \
        --out output/plots/Babaloon_FixedStupidBaseline_${ms}ms_syllable_counts.png \
        | grep -e "Pearson" -e "Saved" | tee -a "$RESULTS"
    echo | tee -a "$RESULTS"
done

echo "All results saved to $RESULTS"
