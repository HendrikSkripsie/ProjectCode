#!/usr/bin/env bash
# Dev-fit plots for Babaloon, ground truth on the y axis, predicted count on the x axis.
# All images are saved in output/plots/DevFit/
# Run from ~/ZeroSyl-master (zerosyl env):   bash run_devfit_plots_babaloon.sh

cd "$(dirname "$0")"

# make sure the 300 ms baseline exists for dev AND test (re-running gives identical files)
python fixed_stupid_baseline.py output/segments/ZeroSylCollapsed/Babaloon --rates 300

echo
echo "================ ZeroSyl: dev plot + test plot ================"
python devfit_plots.py \
    data/babaloonAlignments/dev/ output/segments/ZeroSylCollapsed/Babaloon/dev/ \
    --test data/babaloonAlignments/test/ output/segments/ZeroSylCollapsed/Babaloon/test/ \
    --label ZeroSyl

echo
echo "================ 300 ms baseline: dev plot + test plot ================"
python devfit_plots.py \
    data/babaloonAlignments/dev/ output/segments/FixedStupidBaseline/300ms/Babaloon/dev/ \
    --test data/babaloonAlignments/test/ output/segments/FixedStupidBaseline/300ms/Babaloon/test/ \
    --label 300ms
