"""Causal sufficiency at resid_final and resid_post: substituting the causal component that produces a target output.

Paper section: Section 3 (Causal necessity and sufficiency) with footnote 3, which tests Equations (2) and (3).

Claim: In the paper's words, "At resid_final and resid_post, we solve Equation (2) or Equation (3) for the causal
component that produces the target and substitute it into the activation, rescaling the orthogonal component at
resid_post to preserve the overall length", and "Across all models, the average KL between the target output and the
model's output after the substitution is ∼10^−10 nats at both resid_final and resid_post."

Experiment: At every token position of the 10 evaluation contexts a target distribution over the V process symbols is
drawn from a flat Dirichlet distribution, 0.01 is added to every probability, and the result is renormalized. The same
targets are used for every model, and the probability of the BOS token is kept at each model's own value. At
resid_final the causal component is replaced by the solution of Equation (2) for the target's centered logits. At
resid_post the causal component is replaced by the solution of Equation (3), and the orthogonal component is rescaled
so that the activation keeps its length. If that causal component would be longer than 95 percent of the activation's
length, the target's centered logits are scaled down until it fits.

Saved Outputs: steering_v<d_vocab>_<L>l_h<H>.npz in the output folder, with tags and, per model, target_rf_kl (the KL
divergence from the target to the model's output after the substitution at resid_final), reach_post_kl (the same at
resid_post, where the target is the scaled-down one wherever scaling was needed), and reach_temp_frac (the fraction of
token positions whose target was scaled down). The KL divergences are averaged over contexts and token positions.

How the code works: The causal subspaces are built from all V + 1 columns of the unembedding (src/pullback.py). The
target's centered logits, minus the centered unembedding bias, are what the causal component must produce. At
resid_final the pseudoinverse of the map from subspace coordinates to centered logits gives the correction to the
causal component. At resid_post the same pseudoinverse, for the g_i, gives the coordinates u of the required causal
component per unit of rms(h_post). The causal component is set to rms(h_post) times the scaling factor times u, the
orthogonal component is rescaled so that the length is unchanged, and the output that Equation (3) assigns to this
activation, which is the target or its scaled-down version, is compared with the model's output. Usage: python
experiments/sufficiency/run_steering.py <d_vocab> <n_layers> <n_heads> [ckpt_dir] [out_dir], where d_vocab is V + 1.
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

    # Orthonormal bases of the derived subspaces, and the centered unembedding bias.
    Qpost = PB.window_basis(stack.params)
    Qfin = PB.resid_final_basis(stack.params)
    bU = stack.params["unembed.b_U"]
    bUc = (bU - bU.mean(-1, keepdim=True))[:, None, None, :]

    results = {}
    with torch.no_grad():
        clean = stack.forward(tokens)

        # The targets: flat Dirichlet draws over the V process symbols with 0.01 added, shared across models, with the
        # BOS probability of each model kept. tstar holds the centered logits that the causal component must produce.
        gt = torch.Generator(device="cpu").manual_seed(123)
        B_, P_ = tokens.shape[1], tokens.shape[2]
        e = -torch.rand((B_, P_, V), generator=gt).log()
        pstar_real = e / e.sum(-1, keepdim=True)
        pstar_real = (pstar_real + 0.01).clamp_min(0.01)
        pstar_real = pstar_real / pstar_real.sum(-1, keepdim=True)
        pstar_real = pstar_real.to(dev)
        p_clean = clean.softmax(-1)
        p_bos = p_clean[..., V:V + 1]
        pstar = torch.cat([pstar_real[None] * (1 - p_bos), p_bos], -1)
        tstar = pstar.clamp_min(1e-12).log()
        tstar = tstar - tstar.mean(-1, keepdim=True) - bUc

        # resid_final: solve Equation (2) for the causal component and substitute it.
        WU = stack.params["unembed.W_U"]
        Ucen = WU - WU.mean(-1, keepdim=True)
        Bf = torch.einsum("mdk,mdv->mvk", Qfin, Ucen)
        Bf_pinv = torch.linalg.pinv(Bf.cpu()).to(dev)
        def target_rf(x):
            t_now = torch.einsum("mbpd,mdv->mbpv", x, Ucen)
            q = torch.einsum("mkv,mbpv->mbpk", Bf_pinv, tstar - (t_now - t_now.mean(-1, keepdim=True)))
            return x + torch.einsum("mbpk,mdk->mbpd", q, Qfin)
        out = stack.forward(tokens, edits={"resid_final": target_rf})
        results["target_rf_kl"] = PB.kl(pstar.log(), out).cpu().numpy()

        # resid_post: solve Equation (3) for the causal component, scale the target down where its causal component
        # would not fit in 95 percent of the activation's length, and rescale the orthogonal component to keep the
        # length.
        gam = stack.params["ln_final.w"]
        G = gam[:, :, None] * Ucen
        Bp = torch.einsum("mdk,mdv->mvk", Qpost, G)
        Bp_pinv = torch.linalg.pinv(Bp.cpu()).to(dev)

        reach_pred = [None, None]
        def reach_post(x):
            pj = PB.project(x, Qpost)
            comp = x - pj
            xn = x.norm(dim=-1, keepdim=True)
            cn = comp.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            u = torch.einsum("mkv,mbpv->mbpk", Bp_pinv, tstar)
            un = u.norm(dim=-1, keepdim=True)
            s_orig = (x.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()
            temp = (0.95 * xn / (s_orig * un)).clamp_max(1.0)
            reach_pred[1] = (temp[..., 0] < 1).float().mean((-1, -2))
            pprime = s_orig * temp * torch.einsum("mbpk,mdk->mbpd", u, Qpost)
            pn2 = pprime.pow(2).sum(-1, keepdim=True)
            qscale = ((xn.pow(2) - pn2).clamp_min(0).sqrt() / cn)
            xprime = pprime + qscale * comp
            s2 = xprime.pow(2).mean(-1, keepdim=True) + 1e-5
            cpred = torch.einsum("mbpd,mdv->mbpv", xprime, G) / s2.sqrt()
            reach_pred[0] = (cpred - cpred.mean(-1, keepdim=True) + bUc).log_softmax(-1)
            return xprime
        out = stack.forward(tokens, edits={"resid_post": reach_post})
        results["reach_post_kl"] = PB.kl(reach_pred[0], out).cpu().numpy()
        results["reach_temp_frac"] = reach_pred[1].cpu().numpy()

        print(f"  resid_final KL {np.median(results['target_rf_kl']):.2e} | "
              f"resid_post KL {np.median(results['reach_post_kl']):.2e}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"steering_v{d_vocab}_{n_layers}l_h{n_heads}.npz")
    np.savez(out, tags=np.array(tags), **results)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    dv, nl, nh = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    ckpt = sys.argv[4] if len(sys.argv) > 4 else "results/train/free"
    outd = sys.argv[5] if len(sys.argv) > 5 else "results/steering/free"
    main(dv, nl, nh, ckpt, outd)
