#!/bin/bash
# Runner for the models of the paper: the evaluation contexts, then every free and pinned model.
#
# Paper section: Section 3 (Experiments, Models) and Appendix D (Architecture, Training, and Pinned readout).
#
# Claim: The paper trains 2,640 transformers on 22 processes, with one to five layers, 1, 2, 4, or 8 attention heads,
# three training seeds, and a free and a pinned variant.
#
# Experiment: experiments/training/make_contexts.py once, then experiments/training/train_models.py for every
# vocabulary size, number of layers, and number of heads, once for the free variant and once for the pinned variant
# (PULLBACK_FROZEN_READOUT=1 and PULLBACK_RMS_PENALTY=0.3).
#
# Saved Outputs: results/contexts/<process>.npy, and one .pt file and one .json file per model in results/train/free/
# and results/train/pinned/. The other runner scripts read these.
#
# How the script works: It loops over the variants (override with VARIANTS="..."), the vocabulary sizes including
# the BOS token (DVOCABS), the numbers of layers (LAYERS), and the numbers of heads (HEADS). Each call trains all
# models of one group together on one GPU and skips the models that are already trained.
# Usage: bash scripts/run_training.sh
set -eu
cd "$(dirname "$0")/.."
VARIANTS="${VARIANTS:-free pinned}"
DVOCABS="${DVOCABS:-3 4 5}"
LAYERS="${LAYERS:-1 2 3 4 5}"
HEADS="${HEADS:-1 2 4 8}"
python experiments/training/make_contexts.py
for VARIANT in $VARIANTS; do
    for DV in $DVOCABS; do
        for L in $LAYERS; do
            for H in $HEADS; do
                if [ "$VARIANT" = "pinned" ]; then
                    # The pinned variant: a fixed readout and the penalty that holds rms(resid_post) near one.
                    PULLBACK_FROZEN_READOUT=1 PULLBACK_RMS_PENALTY=0.3 \
                        python experiments/training/train_models.py "$DV" "$L" "$H" results/train/pinned
                else
                    python experiments/training/train_models.py "$DV" "$L" "$H" results/train/free
                fi
            done
        done
    done
done
