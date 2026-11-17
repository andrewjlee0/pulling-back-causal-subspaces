"""Causal necessity at the derived hooks: negating the causal component or the orthogonal component at the logits, at
resid_final, and at resid_post.

Paper section: Section 3 (Causal necessity and sufficiency), which tests Sections 2.1 to 2.3 and Proposition 1 of
Appendix B.

Claim: In the paper's words, "Across all 2,640 models, negating the orthogonal component leaves the output unchanged
at the three derived hooks (average KL ∼10^−11 nats at the logits, resid_final, and resid_post), whereas negating the
causal component gives a substantially larger KL of 14.2, 13.5, and 13.5 nats, respectively."

Experiment: For every model of a group and each of the 10 evaluation contexts, the activation at one hook is split at
every token position into its component in the derived causal subspace and its orthogonal component. One of the two is
multiplied by −1 and the forward pass continues. The causal subspace is the orthogonal complement of the all-ones
vector at the logits, the span of the centered unembedding columns at resid_final, and the span of the g_i of Equation
(3) at resid_post. The subspaces are built from all V + 1 columns of the unembedding, which include the column of the
BOS token.

Saved Outputs: negation_v<d_vocab>_<L>l_h<H>.npz in the output folder, with tags (one per model) and, per model, the
KL divergence from the unedited output to the edited output, averaged over contexts and token positions, under the
keys logits_causal_negate, logits_comp_negate, resid_final_causal_negate, resid_final_comp_negate,
resid_post_causal_negate, and resid_post_comp_negate. Here causal marks the negation of the causal component and comp
marks the negation of the orthogonal component.

How the code works: The script loads the group and its contexts, builds an orthonormal basis of each derived subspace
(src/pullback.py), and runs one unedited forward pass. Each intervention is then one forward pass with an edit
function at its hook. At the logits the causal component is the centered logit vector, so its negation is 2·mean −
logits and the negation of the orthogonal component is logits − 2·mean. At resid_final and resid_post the causal
component is the projection onto the subspace. Negation keeps the length of the activation, so the final RMSNorm
rescales the edited resid_post exactly as it rescaled the original. Usage: python
experiments/necessity/run_negation.py <d_vocab> <n_layers> <n_heads> [ckpt_dir] [out_dir], where d_vocab is V + 1.
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
    Mn = stack.M
    print(f"group v{d_vocab} {n_layers}l h{n_heads}: {Mn} models on {dev}", flush=True)

    # Orthonormal bases of the derived causal subspaces of resid_post and resid_final, per model.
    Qpost = PB.window_basis(stack.params)
    Qfin = PB.resid_final_basis(stack.params)

    results = {}
    with torch.no_grad():
        # The unedited outputs, and a helper that runs one intervention and stores its KL divergence per model.
        clean = stack.forward(tokens)

        def run(name, edits):
            out = stack.forward(tokens, edits=edits)
            results[name] = PB.kl(clean, out).cpu().numpy()
            print(f"  {name:34s} median KL {np.median(results[name]):.2e}", flush=True)

        # Logits and resid_final: negate the causal component, then negate the orthogonal component.
        run("logits_causal_negate",
            {"logits": lambda lg: 2 * lg.mean(-1, keepdim=True) - lg})
        run("logits_comp_negate",
            {"logits": lambda lg: lg - 2 * lg.mean(-1, keepdim=True)})
        run("resid_final_causal_negate",
            {"resid_final": lambda x: x - 2 * PB.project(x, Qfin)})
        run("resid_final_comp_negate",
            {"resid_final": lambda x: 2 * PB.project(x, Qfin) - x})

        # resid_post: the same two negations around the span of the g_i.
        def negate(x, side):
            pj = PB.project(x, Qpost)
            return x - 2 * pj if side == "causal" else 2 * pj - x
        run("resid_post_causal_negate",
            {"resid_post": lambda x: negate(x, "causal")})
        run("resid_post_comp_negate",
            {"resid_post": lambda x: negate(x, "comp")})

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"negation_v{d_vocab}_{n_layers}l_h{n_heads}.npz")
    np.savez(out, tags=np.array(tags), **results)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    dv, nl, nh = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    ckpt = sys.argv[4] if len(sys.argv) > 4 else "results/train/free"
    outd = sys.argv[5] if len(sys.argv) > 5 else "results/negation/free"
    main(dv, nl, nh, ckpt, outd)
