#!/usr/bin/env bash
# Syllable-count estimator on Babaloon (fit on dev, test on test)
# + 300 ms fixed stupid baseline on the test set.
#
# Run from ~/ZeroSyl-master inside the zerosyl conda env, AFTER the Babaloon test
# set has been encoded (output/segments/ZeroSylCollapsed/Babaloon/test/ must exist):
#
#   bash run_estimator_babaloon.sh             # estimator: fit + predict + evaluate
#   bash run_estimator_babaloon.sh baseline    # 300 ms fixed baseline on test + its evaluation
#   bash run_estimator_babaloon.sh all         # both

cd "$(dirname "$0")"

estimator() {
    echo "================ 1. fit on dev ================"
    python syllable_estimator.py fit \
        data/babaloonAlignments/dev/ \
        output/segments/ZeroSylCollapsed/Babaloon/dev/ \
        --name Babaloon

    echo "================ 2. predict on test (no ground truth used) ================"
    python syllable_estimator.py predict \
        output/segments/ZeroSylCollapsed/Babaloon/test/ \
        --name Babaloon

    echo "================ 3. evaluate on test ================"
    python syllable_estimator.py evaluate \
        data/babaloonAlignments/test/ \
        output/segments/ZeroSylCollapsed/Babaloon/test/ \
        --name Babaloon
}

baseline() {
    echo "================ 300 ms fixed baseline on Babaloon test ================"
    python fixed_stupid_baseline.py output/segments/ZeroSylCollapsed/Babaloon \
        --rates 300 --pattern "test/*.pt"

    echo "---- boundary metrics ----"
    zerosyl evaluate boundaries \
        output/segments/FixedStupidBaseline/300ms/Babaloon/test/ \
        data/babaloonAlignments/test/ \
        --segments-pattern "*.pt" \
        --textgrid-pattern "*.TextGrid"

    echo "---- syllable count plot + Pearson's r ----"
    python syllable_count_plot.py \
        data/babaloonAlignments/test/ \
        output/segments/FixedStupidBaseline/300ms/Babaloon/test/ \
        --out output/plots/Babaloon_test_FixedStupidBaseline_300ms_syllable_counts.png
}

case "$1" in
    baseline) baseline ;;
    all)      estimator; baseline ;;
    *)        estimator ;;
esac
