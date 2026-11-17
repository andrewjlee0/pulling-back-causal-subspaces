"""The orthogonal component of resid_post rescaled to three times its length, with its direction kept.

Paper section: Appendix H (Comparison with Boundless DAS), with Proposition 1 of Appendix B.

Claim: In the paper's words, "Conversely, rescaling the orthogonal complement to three times its length while keeping
its direction raises the KL to a median of 3.3 × 10^−2 nats, because the length change moves the denominator of
Equation (3) and rescales the activation-dependent part of every centered logit."

Experiment: For every model of a group and each of the 10 evaluation contexts, the component of resid_post orthogonal
to the span of the g_i is multiplied by three at every token position, and the causal component is left unchanged. The
subspace is built from all V + 1 columns of the unembedding.

Saved Outputs: length_rescaling_v<d_vocab>_<L>l_h<H>.npz in the output folder, with tags, scale (the factor, 3), and
resid_post_comp_scale (per model, the KL divergence from the unedited output to the edited output, averaged over
contexts and token positions).

How the code works: The intervention is one forward pass with an edit function at resid_post. QSCALE sets the factor.
Usage: python experiments/boundless_das/run_length_rescaling.py <d_vocab> <n_layers> <n_heads> [ckpt_dir] [out_dir],
where d_vocab is V + 1.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src import pullback as PB
from src.evaluation import context_tokens, load_group

# The factor that multiplies the orthogonal component.
SCALE = float(os.environ.get("QSCALE", "3.0"))


def main(d_vocab, n_layers, n_heads, ckpt_dir, out_dir):
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    V = d_vocab - 1
    stack, tags, _ = load_group(d_vocab, n_layers, n_heads, ckpt_dir, dev)
    tokens = context_tokens(tags, V, dev)
    print(f"group v{d_vocab} {n_layers}l h{n_heads}: {stack.M} models on {dev}, scale {SCALE:g}", flush=True)

    Qpost = PB.window_basis(stack.params)

    results = {}
    with torch.no_grad():
        clean = stack.forward(tokens)

        def run(name, edits):
            out = stack.forward(tokens, edits=edits)
            results[name] = PB.kl(clean, out).cpu().numpy()
            print(f"  {name:28s} median KL {np.median(results[name]):.2e}", flush=True)

        # Multiply the orthogonal component by the factor and keep the causal component.
        def comp_scale(x):
            pj = PB.project(x, Qpost)
            return pj + SCALE * (x - pj)

        run("resid_post_comp_scale", {"resid_post": comp_scale})

        results["scale"] = np.array(SCALE)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"length_rescaling_v{d_vocab}_{n_layers}l_h{n_heads}.npz")
    np.savez(out, tags=np.array(tags), **results)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    dv, nl, nh = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    ckpt = sys.argv[4] if len(sys.argv) > 4 else "results/train/free"
    outd = sys.argv[5] if len(sys.argv) > 5 else "results/length_rescaling/free"
    main(dv, nl, nh, ckpt, outd)
