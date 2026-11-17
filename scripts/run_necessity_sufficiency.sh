#!/bin/bash
# Runner for Section 3: causal necessity at the four hooks, the models' loss, and causal sufficiency at resid_final
# and resid_post, for every model.
#
# Paper section: Section 3 (Causal necessity and sufficiency), and the estimated causal subspace of Appendix C.
#
# Claim: Negating the orthogonal component leaves the output unchanged at the logits, resid_final, and resid_post,
# negating the causal component changes it by 14.2, 13.5, and 13.5 nats, and substituting the causal component that
# produces a target output makes the model's output equal the target.
#
# Experiment: For every group of models, experiments/necessity/run_negation.py, experiments/necessity/run_loss.py,
# experiments/necessity/run_mlp_in_negation.py, and experiments/sufficiency/run_steering.py.
#
# Saved Outputs: results/negation/<variant>/negation_v<d_vocab>_<L>l_h<H>.npz, and the same pattern for loss,
# mlp_in_negation, and steering, read by notebooks/02_necessity_sufficiency.ipynb.
#
# How the script works: It loops over the variants (override with VARIANTS="..."), the vocabulary sizes including
# the BOS token (DVOCABS), the numbers of layers (LAYERS), and the numbers of heads (HEADS), and reads the models
# from results/train/<variant>/. Usage: bash scripts/run_necessity_sufficiency.sh
set -eu
cd "$(dirname "$0")/.."
VARIANTS="${VARIANTS:-free pinned}"
DVOCABS="${DVOCABS:-3 4 5}"
LAYERS="${LAYERS:-1 2 3 4 5}"
HEADS="${HEADS:-1 2 4 8}"
for VARIANT in $VARIANTS; do
    CKPT="results/train/$VARIANT"
    for DV in $DVOCABS; do
        for L in $LAYERS; do
            for H in $HEADS; do
                python experiments/necessity/run_negation.py "$DV" "$L" "$H" "$CKPT" "results/negation/$VARIANT"
                python experiments/necessity/run_loss.py "$DV" "$L" "$H" "$CKPT" "results/loss/$VARIANT"
                python experiments/necessity/run_mlp_in_negation.py "$DV" "$L" "$H" "$CKPT" "results/mlp_in_negation/$VARIANT"
                python experiments/sufficiency/run_steering.py "$DV" "$L" "$H" "$CKPT" "results/steering/$VARIANT"
            done
        done
    done
done
