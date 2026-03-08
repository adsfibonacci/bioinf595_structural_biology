#!/usr/bin/env bash
# run_backrub_sweep.sh
# Runs sample_backrub.py across a grid of mc_kt x ntrials values.
# Edit the variables below to match your paths.

set -euo pipefail

NATIVE_PDB="data/MC4R.cif"
INPUT_PDB="intermediates/relaxes/MC4R_relaxed_3.pdb"    # lowest-energy relaxed
OUTPUT_DIR="intermediates/backrub"
NSAMPLES=20

# Parameter grid
MC_KT_VALUES=(0.4 0.7 1.0 1.5 10)
NTRIALS_VALUES=(1000 5000 10000)

mkdir -p "$OUTPUT_DIR"

for kt in "${MC_KT_VALUES[@]}"; do
  for nt in "${NTRIALS_VALUES[@]}"; do
    echo "========================================"
    echo "  mc_kt=${kt}  ntrials=${nt}"
    echo "========================================"
    python src/sample_backrub.py \
      --native_pdb  "$NATIVE_PDB" \
      --input_pdb   "$INPUT_PDB"  \
      --output_dir  "$OUTPUT_DIR" \
      --ntrials     "$nt"         \
      --mc_kt       "$kt"         \
      --nsamples    "$NSAMPLES"
  done
done

echo "All sweeps complete."
