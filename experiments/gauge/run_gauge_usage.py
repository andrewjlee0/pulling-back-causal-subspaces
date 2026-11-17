"""Gauge usage: how strongly trained models use the invertible linear map and the per-token-position scaling that the
derivation leaves free.

Paper section: Appendix H (Gauge usage), referenced from Section 3 (Boundless DAS and gauge freedoms).

Claim: In the paper's words, "Freely trained models use both freedoms mildly. Across models, the mean singular-value
ratio is 1.25, and the mean log-RMS variance is 0.08. The pinned construction effectively fixes both freedoms,
yielding 1.000 and approximately 10^−5, respectively."

Experiment: For every model, κ(G) is the ratio of the largest to the smallest nonzero singular value of the matrix G
whose columns are the g_i of the V process symbols, and the log-RMS variance is the variance of log rms(h_post) over
the token positions of the 10 evaluation contexts.

Saved Outputs: gauge_usage_v<d_vocab>_<L>l_h<H>.npz in the output folder, with tags, shear_cond_w (κ(G), per model),
and var_log_rms (the variance of log rms(h_post), per model).

How the code works: The script records resid_post of the final layer in one forward pass and takes its root-mean-
square at each of the first 127 positions of every context, with the epsilon of the RMSNorm. For κ(G) it centers the
unembedding columns of the V process symbols, rescales them entrywise by the weight of the final RMSNorm, and
expresses them in an orthonormal basis of the V − 1 dimensional space orthogonal to the all-ones vector, which leaves
V − 1 nonzero singular values. Usage: python experiments/gauge/run_gauge_usage.py <d_vocab> <n_layers> <n_heads>
[ckpt_dir] [out_dir], where d_vocab is V + 1.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.evaluation import context_tokens, load_group


def main(d_vocab, n_layers, n_heads, ckpt_dir, out_dir):
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    V = d_vocab - 1
    stack, tags, _ = load_group(d_vocab, n_layers, n_heads, ckpt_dir, dev)
    tokens = context_tokens(tags, V, dev)
    Mn = stack.M
    N = tokens.shape[1] * (tokens.shape[2] - 1)
    print(f"group v{d_vocab} {n_layers}l h{n_heads}: {Mn} models on {dev}", flush=True)

    # Record resid_post of the final layer and its root-mean-square at every token position.
    rec = {}
    def record(x):
        rec["post"] = x
        return x
    with torch.no_grad():
        clean = stack.forward(tokens, edits={"resid_post": record})
    post = rec["post"].cpu().float()[:, :, :-1, :]
    rmsp = post.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()

    # An orthonormal basis of the space orthogonal to the all-ones vector in R^V.
    Cm_ = np.eye(V) - np.ones((V, V)) / V
    Hh, _ = np.linalg.qr(Cm_[:, :V - 1])
    Ht = torch.tensor(Hh, dtype=torch.float32)
    cparams = {k: v.cpu().float() for k, v in stack.params.items()}
    gam_f = cparams["ln_final.w"]

    # κ(G) from the singular values of the g_i of the V process symbols, and the variance of log rms(h_post).
    WUc = cparams["unembed.W_U"][:, :, :V]
    Ucen = WUc - WUc.mean(-1, keepdim=True)
    Gmap = torch.einsum("md,mdv,vk->mdk", gam_f, Ucen, Ht)
    sv = torch.linalg.svdvals(Gmap)
    shear_cond_w = (sv[:, 0] / sv[:, -1].clamp_min(1e-12)).numpy()
    var_log_rms = rmsp.reshape(Mn, N).log().var(1).numpy()

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"gauge_usage_v{d_vocab}_{n_layers}l_h{n_heads}.npz")
    np.savez(out, tags=np.array(tags), shear_cond_w=shear_cond_w, var_log_rms=var_log_rms)

    print(f"  SHEAR-W cond {np.median(shear_cond_w):.4f} | "
          f"Var(log rms) {np.median(var_log_rms):.4f}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    dv, nl, nh = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    ckpt = sys.argv[4] if len(sys.argv) > 4 else "results/train/free"
    outd = sys.argv[5] if len(sys.argv) > 5 else "results/gauge_usage/free"
    main(dv, nl, nh, ckpt, outd)
