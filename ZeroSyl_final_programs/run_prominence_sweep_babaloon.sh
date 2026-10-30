#!/usr/bin/env bash
# ZeroSyl prominence sweep (0.00 -> 1.00 in steps of 0.05) on Babaloon.
# Run from ~/ZeroSyl-master inside the zerosyl conda env:
#
#   bash run_prominence_sweep_babaloon.sh test   # quick check on devSMALL (one file)
#   bash run_prominence_sweep_babaloon.sh        # full Babaloon dev sweep
#   bash run_prominence_sweep_babaloon.sh eval   # re-evaluate dev only (already encoded)
#
# Results: output/results/ProminenceSweep_Babaloon.txt / .csv
#          output/plots/ProminenceSweep_Babaloon.png

cd "$(dirname "$0")"

case "$1" in
  test)
    python prominence_sweep.py \
        data/babaloonWaveforms/devSMALL/ data/babaloonAlignments/devSMALL/ \
        --name BabaloonSMALL
    ;;
  eval)
    python prominence_sweep.py \
        data/babaloonWaveforms/dev/ data/babaloonAlignments/dev/ \
        --name Babaloon --skip-encode
    ;;
  *)
    python prominence_sweep.py \
        data/babaloonWaveforms/dev/ data/babaloonAlignments/dev/ \
        --name Babaloon
    ;;
esac
