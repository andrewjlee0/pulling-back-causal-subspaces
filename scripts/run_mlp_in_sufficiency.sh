#!/bin/bash
# Runner for Appendix G: causal sufficiency at mlp_in, for every model.
#
# Paper section: Appendix G (Causal sufficiency at mlp_in), referenced from Section 3.
#
# Claim: Within the estimated causal subspace of mlp_in the KL divergence to a target output can be brought far lower
# than within the same number of random directions orthogonal to it.
#
# Experiment: experiments/sufficiency/run_mlp_in_sufficiency.py for every group of models.
#
# Saved Outputs: results/mlp_in_sufficiency/<variant>/mlp_in_sufficiency_v<d_vocab>_<L>l_h<H>.npz, read by
# notebooks/03_mlp_in_sufficiency.ipynb.
#
# How the script works: It loops over the variants (override with VARIANTS="..."), the vocabulary sizes including
# the BOS token (DVOCABS), the numbers of layers (LAYERS), and the numbers of heads (HEADS), and reads the models
# from results/train/<variant>/. Usage: bash scripts/run_mlp_in_sufficiency.sh
set -eu
cd "$(dirname "$0")/.."
VARIANTS="${VARIANTS:-free pinned}"
DVOCABS="${DVOCABS:-3 4 5}"
LAYERS="${LAYERS:-1 2 3 4 5}"
HEADS="${HEADS:-1 2 4 8}"
for VARIANT in $VARIANTS; do
    for DV in $DVOCABS; do
        for L in $LAYERS; do
            for H in $HEADS; do
                python experiments/sufficiency/run_mlp_in_sufficiency.py "$DV" "$L" "$H" "results/train/$VARIANT" \
                    "results/mlp_in_sufficiency/$VARIANT"
            done
        done
    done
done
