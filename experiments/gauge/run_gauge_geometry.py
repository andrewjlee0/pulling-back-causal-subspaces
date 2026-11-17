"""The model coordinates of Figure 2: each model's logits, and the coordinates of resid_final and resid_post in their
derived causal subspaces.

Paper section: Appendix H (Gauge usage) with Figure 2.

Claim: In the words of the caption of Figure 2, "To the right, we show the gauges realized in the derived causal
subspaces (2D for this dataset) by a free and a pinned model with four layers and four heads."

Experiment: For every model of a group, the logits, resid_final, and resid_post of the final layer are recorded at
every token position of the 10 evaluation contexts. resid_final and resid_post are expressed in orthonormal bases of
their derived causal subspaces, built from the unembedding columns of the V process symbols, so the coordinates have V
− 1 dimensions.

Saved Outputs: gauge_geometry_v<d_vocab>_<L>l_h<H>.npz in the output folder, with tags, coords_fin2 and coords_post2
(the coordinates of resid_final and resid_post, of shape [models, 10, 128, V − 1]), and logits (of shape [models, 10,
128, V + 1]).

How the code works: The script builds the two bases (src/pullback.py with full=False), records resid_post and
resid_final in one forward pass, and projects them onto the bases. Figure 2 uses the group with d_vocab 4, four
layers, and four heads, which holds the models of the nonergodic_c2a06 process. Usage: python
experiments/gauge/run_gauge_geometry.py <d_vocab> <n_layers> <n_heads> [ckpt_dir] [out_dir], where d_vocab is V + 1.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src import pullback as PB
from src.evaluation import context_tokens, load_group


def main(d_vocab, n_layers, n_heads, ckpt_dir, out_dir):
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    V = d_vocab - 1
    stack, tags, _ = load_group(d_vocab, n_layers, n_heads, ckpt_dir, dev)
    tokens = context_tokens(tags, V, dev)
    print(f"group v{d_vocab} {n_layers}l h{n_heads}: {stack.M} models on {dev}", flush=True)

    # Orthonormal bases of the derived subspaces of resid_post and resid_final, from the columns of the V process
    # symbols.
    Qpost2 = PB.window_basis(stack.params, V=V, full=False)
    Qfin2 = PB.resid_final_basis(stack.params, V=V, full=False)

    # Record resid_post and resid_final, and project them onto the bases.
    rec = {}
    def rec_post(x):
        rec["post"] = x
        return x
    def rec_fin(x):
        rec["fin"] = x
        return x
    with torch.no_grad():
        logits = stack.forward(tokens, edits={"resid_post": rec_post,
                                              "resid_final": rec_fin})
        coords_post2 = torch.einsum("mbpd,mdk->mbpk", rec["post"], Qpost2)
        coords_fin2 = torch.einsum("mbpd,mdk->mbpk", rec["fin"], Qfin2)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"gauge_geometry_v{d_vocab}_{n_layers}l_h{n_heads}.npz")
    np.savez(out, tags=np.array(tags),
             coords_post2=coords_post2.cpu().numpy().astype(np.float32),
             coords_fin2=coords_fin2.cpu().numpy().astype(np.float32),
             logits=logits.cpu().numpy().astype(np.float32))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    dv, nl, nh = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    ckpt = sys.argv[4] if len(sys.argv) > 4 else "results/train/free"
    outd = sys.argv[5] if len(sys.argv) > 5 else "results/gauge_geometry/free"
    main(dv, nl, nh, ckpt, outd)
