#!/bin/bash
# Runner for Appendix H: gauge usage, the model coordinates of Figure 2, Boundless DAS, and the rescaling of the
# orthogonal component of resid_post, for every model.
#
# Paper section: Appendix H (Gauge usage and comparison to Boundless DAS) with Figure 2, referenced from Section 3
# (Boundless DAS and gauge freedoms).
#
# Claim: Freely trained models use the gauge freedoms mildly, Boundless DAS recovers the derived subspaces, and the
# additional directions that it learns at resid_post act only through the length of the orthogonal component.
#
# Experiment: For every group of models, experiments/gauge/run_gauge_usage.py,
# experiments/boundless_das/run_boundless_das.py, and experiments/boundless_das/run_length_rescaling.py. For the
# group of Figure 2 (d_vocab 4, four layers, four heads), experiments/gauge/run_gauge_geometry.py.
#
# Saved Outputs: results/gauge_usage/<variant>/gauge_usage_v<d_vocab>_<L>l_h<H>.npz, the same pattern for
# boundless_das and length_rescaling, and results/gauge_geometry/<variant>/gauge_geometry_v4_4l_h4.npz, read by
# notebooks/04_gauge_boundless_das.ipynb.
#
# How the script works: It loops over the variants (override with VARIANTS="..."), the vocabulary sizes including
# the BOS token (DVOCABS), the numbers of layers (LAYERS), and the numbers of heads (HEADS), and reads the models
# from results/train/<variant>/. Usage: bash scripts/run_gauge_boundless_das.sh
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
                python experiments/gauge/run_gauge_usage.py "$DV" "$L" "$H" "$CKPT" "results/gauge_usage/$VARIANT"
                python experiments/boundless_das/run_boundless_das.py "$DV" "$L" "$H" "$CKPT" "results/boundless_das/$VARIANT"
                python experiments/boundless_das/run_length_rescaling.py "$DV" "$L" "$H" "$CKPT" "results/length_rescaling/$VARIANT"
            done
        done
    done
    # The group of Figure 2.
    python experiments/gauge/run_gauge_geometry.py 4 4 4 "$CKPT" "results/gauge_geometry/$VARIANT"
done
