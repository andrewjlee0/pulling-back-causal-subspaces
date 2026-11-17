"""Causal necessity at mlp_in: the estimated causal subspace from the sensitivity matrix, and the effect of negating its
component.

Paper section: Section 2.4 (Estimated causal subspace), Section 3 (Causal necessity and sufficiency), and Appendix C
with Equations (5) and (7).

Claim: In the paper's words, "At mlp_in, whose subspace is estimated rather than derived, negating the orthogonal
component yields a small but nonzero KL of 0.041 nats (about 5% of the models' average per-token loss of 0.78 nats),
compared with 2.1 nats for the causal component", and "k99 ranges from 3 to 31 across processes (median 7)".

Experiment: For every model, the sensitivity matrix M_sens is accumulated over the token positions of the 10
evaluation contexts, and the estimated causal subspace is the span of its top k99 eigenvectors, where k99 is the
smallest number of eigenvectors that hold 99 percent of the eigenvalue sum. Two interventions on the final layer's
mlp_in are compared. The first, causal_negate, multiplies the component in the estimated subspace by −1. The second,
comp_matched, moves the activation against the direction of its orthogonal component by twice the length of its causal
component, so that both interventions displace the activation by the same distance. After either intervention the
activation is rescaled so that, divided entrywise by the weight of the MLP's RMSNorm, it has a root-mean-square of
one, as every output of that RMSNorm has.

Saved Outputs: mlp_in_negation_v<d_vocab>_<L>l_h<H>.npz in the output folder, with tags, k99 (per model), and, per
model, the KL divergence from the unedited output to the edited output, averaged over contexts and token positions,
under the keys causal_negate_kl and comp_matched_kl.

How the code works: The script records mlp_in of the final layer in one forward pass. For each vocabulary entry i it
forms, at every token position, the vector v_i = W_in (GELU'(z) ⊙ W_out g_i) of Equation (5), and it averages the sum
over i of v_i v_i^T over the first 127 positions of the 10 contexts. The g_i are built from all V + 1 columns of the
unembedding, which include the column of the BOS token, and the positive scalar 1/rms(h_post) of Equation (5) is not
included in v_i. The eigenvectors of this matrix are sorted by eigenvalue, and the first k99 span the estimated
subspace. Each edited activation is put back on the shell of the MLP's RMSNorm by shell_renorm and substituted at
mlp_in in a full forward pass. Usage: python experiments/necessity/run_mlp_in_negation.py <d_vocab> <n_layers>
<n_heads> <ckpt_dir> <out_dir>, where d_vocab is V + 1.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src import pullback as PB
from src.evaluation import context_tokens, load_group


# Rescale an activation so that, divided by the weight of the MLP's RMSNorm, it has a root-mean-square of one.
def shell_renorm(x, gam2):
    u = x / gam2[:, None, None, :].clamp_min(1e-12)
    u = u / u.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-12)
    return gam2[:, None, None, :] * u


LABELS = ("causal_negate", "comp_matched")


def main(d_vocab, n_layers, n_heads, ckpt_dir, out_dir):
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    V = d_vocab - 1
    stack, tags, _ = load_group(d_vocab, n_layers, n_heads, ckpt_dir, dev)
    tokens = context_tokens(tags, V, dev)
    Mn = stack.M
    B_, P_ = tokens.shape[1], tokens.shape[2]
    N = B_ * (P_ - 1)
    print(f"group v{d_vocab} {n_layers}l h{n_heads}: {Mn} models on {dev}", flush=True)

    # Record mlp_in of the final layer in the unedited forward pass.
    rec = {}
    def rec_mlp(x):
        rec["mlp_in"] = x
        return x
    with torch.no_grad():
        clean = stack.forward(tokens, edits={"mlp_in": rec_mlp})
    xm = rec["mlp_in"]
    # The weights of the final MLP and of the readout. G holds the g_i, and C holds W_out g_i for every vocabulary
    # entry.
    pre = f"blocks.{n_layers-1}."
    W_in = stack.params[pre + "mlp.W_in"]
    b_in = stack.params[pre + "mlp.b_in"]
    W_out = stack.params[pre + "mlp.W_out"]
    gam_f = stack.params["ln_final.w"]
    gam2 = stack.params[pre + "ln2.w"]
    WU = stack.params["unembed.W_U"]
    Ucen = WU - WU.mean(-1, keepdim=True)
    G = gam_f[:, :, None] * Ucen
    C = torch.einsum("mfd,mdv->mfv", W_out, G)

    with torch.no_grad():
        # The sensitivity matrix: GELU'(z) at every position, the vectors v_i, and the average of their outer
        # products. Its eigenvectors are sorted from the largest eigenvalue to the smallest.
        z = torch.einsum("mbpd,mdf->mbpf", xm, W_in) + b_in[:, None, None, :]
        sp = torch.distributions.Normal(0, 1).cdf(z) + z * torch.exp(
            torch.distributions.Normal(0, 1).log_prob(z))
        Ms = torch.zeros(Mn, xm.shape[-1], xm.shape[-1], device=dev)
        for v in range(C.shape[-1]):
            Dv = torch.einsum("mdf,mbpf->mbpd", W_in, sp * C[:, None, None, :, v])
            Dv = Dv[:, :, :-1, :].reshape(Mn, N, -1)
            Ms += torch.einsum("mnd,mne->mde", Dv, Dv) / N
        evals, evecs = torch.linalg.eigh(Ms.cpu())
        evals = evals.flip(-1); evecs = evecs.flip(-1)

    xm_t = xm[:, :, :-1, :].cpu().float()
    edits = {k: torch.zeros_like(xm_t) for k in LABELS}
    res = {"tags": np.array(tags), "k99": np.zeros(Mn)}
    with torch.no_grad():
        for m in range(Mn):
            # k99, the estimated subspace, and the two edited activations of each model.
            mass = torch.cumsum(evals[m], 0) / evals[m].sum().clamp_min(1e-30)
            k = int((mass < 0.99).sum().item()) + 1
            res["k99"][m] = k
            Q = evecs[m][:, :k].float()
            X = xm_t[m].reshape(N, -1)
            cvec = X @ Q
            cX = cvec @ Q.T
            rv = X - cX
            rhat = rv / rv.norm(dim=1, keepdim=True).clamp_min(1e-12)
            edits["causal_negate"][m] = (X - 2 * cX).reshape(B_, P_ - 1, -1)
            edits["comp_matched"][m] = (X - 2 * cvec.norm(dim=1, keepdim=True) * rhat
                                        ).reshape(B_, P_ - 1, -1)
        # Substitute each edited activation at mlp_in, after the rescaling, and measure the KL divergence to the
        # unedited output.
        for label in LABELS:
            xin = edits[label]
            full = torch.cat([xin, xm[:, :, -1:, :].cpu().float()], 2).to(dev)
            full = shell_renorm(full, gam2)
            def sub(x, fx=full):
                return fx.to(x.dtype)
            out = stack.forward(tokens, edits={"mlp_in": sub})
            res[f"{label}_kl"] = PB.kl(clean, out).cpu().numpy()

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"mlp_in_negation_v{d_vocab}_{n_layers}l_h{n_heads}.npz")
    np.savez(out, **res)
    print(f"wrote {out} | " + " | ".join(f"{k} {np.mean(res[k + '_kl']):.4g}" for k in LABELS),
          flush=True)


if __name__ == "__main__":
    dv, nl, nh = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    main(dv, nl, nh, sys.argv[4], sys.argv[5])
