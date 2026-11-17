"""Boundless DAS at resid_final and resid_post, compared with the derived causal subspaces.

Paper section: Appendix H (Comparison with Boundless DAS), referenced from Section 3 (Boundless DAS and gauge
freedoms).

Claim: In the paper's words, at resid_final "the learned subspace has directions that align with the derived subspace
with a median largest-principal-angle cosine of 0.98, and has a median of zero additional directions beyond our
derived subspace", and at resid_post, in freely trained models, "Boundless DAS recovers the derived subspace (median
largest-principal-angle cosine of 0.96), but also learns a median of two additional directions per model".
Interchanging the additional directions "changes the output by a median of 6.5 × 10^−3 nats in freely trained models",
and after rescaling the orthogonal component to restore the original value of rms(h_post), "The output change falls to
∼10^−11 nats".

Experiment: For every model, 4,000 token positions are drawn from the first 127 positions of the 10 evaluation
contexts, and their resid_post activations, and the resid_final activations computed from them, are the data. At each
of the two hooks Boundless DAS learns an orthonormal frame of up to eight directions and a boundary that sets how many
of them are used. It is trained for 500 steps to minimize the interchange KL divergence: the coordinates of a base
activation along the used directions are replaced by those of a source activation, and the output should equal the
source's output over the V process symbols. A penalty of 3e-5 per used direction is added. After training, a used
direction is removed when its removal raises the interchange KL divergence by less than 2e-5. The learned frame is
then compared with the derived subspace, which is built from the unembedding columns of the V process symbols. The
additional directions are the part of the learned frame outside the derived subspace. Their effect is measured by
interchanging only them, once as they are and once with the component orthogonal to the derived subspace rescaled so
that the activation keeps the base's length.

Saved Outputs: boundless_das_v<d_vocab>_<L>l_h<H>.npz in the output folder, with tags and, per model and for each site
in post and final, <site>_bdim (the number of learned directions), <site>_kl_bdas and <site>_kl_derived (the
interchange KL divergence of the learned frame and of the derived subspace), <site>_content_svmin (the smallest
singular value of the derived basis times the learned frame, which is the cosine of the largest principal angle),
<site>_content_dim (the number of those singular values above 0.7), <site>_n_extra (the number of additional
directions), and <site>_extra_kl_raw (the KL divergence from the base's output after interchanging only the additional
directions). For resid_post the file also holds post_extra_kl (the same quantity), post_extra_kl_normfix (the same
after the rescaling that restores the length), and post_extra_r2 (the squared correlation between the coordinate along
the first additional direction and log rms(h_post)).

How the code works: The frame is the QR factor of a learned matrix, and the boundary enters through a sigmoid mask
over the ordered directions whose temperature is annealed from 1 to 0.05. swap_kl estimates an interchange KL
divergence from random pairs of base and source activations. The additional directions of a model are the left
singular vectors, with singular value above 0.3, of the learned frame minus its projection onto the derived subspace.
The random draws of this script are not seeded, so its numbers vary slightly between runs. BDAS_ITERS sets the number
of training steps. Usage: python experiments/boundless_das/run_boundless_das.py <d_vocab> <n_layers> <n_heads>
[ckpt_dir] [out_dir], where d_vocab is V + 1.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src import pullback as PB
from src.evaluation import context_tokens, load_group

# Training steps, pairs per step, sampled token positions, the largest number of directions, and the penalty per
# direction.
ITERS = int(__import__("os").environ.get("BDAS_ITERS", 500))
BS = 256
NSUB = 4000
KMAX = 8
LAM = 3e-5


def main(d_vocab, n_layers, n_heads, ckpt_dir, out_dir):
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    V = d_vocab - 1
    stack, tags, _ = load_group(d_vocab, n_layers, n_heads, ckpt_dir, dev)
    tokens = context_tokens(tags, V, dev)
    Mn = stack.M
    # The data: resid_post at 4,000 sampled token positions, and resid_final computed from it.
    rec = {}
    def cap_post(x):
        rec["post"] = x
        return x
    with torch.no_grad():
        stack.forward(tokens, edits={"resid_post": cap_post})
    post = rec["post"][:, :, :-1, :].reshape(Mn, -1, 128).float()
    g = torch.Generator(device="cpu").manual_seed(0)
    idx = torch.randperm(post.shape[1], generator=g)[:NSUB]
    Hp = post[:, idx].to(dev)
    gam = stack.params["ln_final.w"].detach().float()
    WU = stack.params["unembed.W_U"].detach().float()
    bU = stack.params["unembed.b_U"].detach().float()
    with torch.no_grad():
        Hf = (Hp * torch.rsqrt(Hp.pow(2).mean(-1, keepdim=True) + 1e-5)
              * gam[:, None, :])
    cparams = {kk: v.cpu().float() for kk, v in stack.params.items()}
    # The derived causal subspaces of the two hooks.
    Qd = {"post": PB.window_basis(cparams, V=V, full=False).to(dev).float(),
          "final": PB.resid_final_basis(cparams, V=V, full=False).to(dev).float()}

    # The model's log-probabilities over the V process symbols, as a function of the activation at a hook.
    def make_readout(site):
        if site == "post":
            def ro(h):
                ln = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-5) \
                     * gam[:, None, :]
                return torch.log_softmax(
                    (torch.einsum("mnd,mdv->mnv", ln, WU) + bU[:, None, :])[..., :V], -1)
        else:
            def ro(h):
                return torch.log_softmax(
                    (torch.einsum("mnd,mdv->mnv", h, WU) + bU[:, None, :])[..., :V], -1)
        return ro

    # The interchange: move the base activation to the source's coordinates along the masked directions of the frame.
    def masked_swap(hb, hs, W, mask):
        co = torch.einsum("mnd,mdk->mnk", hs - hb, W) * mask[:, None, :]
        return hb + torch.einsum("mnk,mdk->mnd", co, W)

    # The interchange KL divergence over random pairs, toward the source's output or, with vs_base, toward the base's
    # output. With normfix the component orthogonal to the derived subspace is rescaled to restore the base's length.
    def swap_kl(H, W, mask, ro, iters=20, normfix=False, vs_base=False):
        with torch.no_grad():
            tot = torch.zeros(Mn, device=dev)
            n = H.shape[1]
            for _ in range(iters):
                bi = torch.randint(0, n, (BS,), device=dev)
                si = torch.randint(0, n, (BS,), device=dev)
                hb, hs = H[:, bi], H[:, si]
                hp = masked_swap(hb, hs, W, mask)
                if normfix:
                    Q = Qd["post"]
                    pj = torch.einsum("mnd,mdk,mek->mne", hp, Q, Q)
                    comp = hp - pj
                    want = hb.pow(2).sum(-1, keepdim=True) - pj.pow(2).sum(-1, keepdim=True)
                    cn2 = comp.pow(2).sum(-1, keepdim=True).clamp_min(1e-12)
                    hp = pj + comp * (want.clamp_min(0) / cn2).sqrt()
                lp = ro(hp)
                tgt = ro(hb) if vs_base else ro(hs)
                tot += (tgt.exp() * (tgt - lp)).sum(-1).mean(-1)
            return (tot / iters).cpu().numpy()

    res = {}
    for site, H in [("post", Hp), ("final", Hf)]:
        ro = make_readout(site)
        # Train the frame and the boundary of every model.
        Mp = torch.randn(Mn, 128, KMAX, device=dev)
        Mp.requires_grad_(True)
        bnd = torch.full((Mn,), float(V), device=dev, requires_grad=True)
        opt = torch.optim.Adam([{"params": [Mp], "lr": 0.05},
                                {"params": [bnd], "lr": 0.02}])
        n = H.shape[1]
        for it in range(ITERS):
            tau = 1.0 * (0.05 / 1.0) ** (it / (ITERS - 1))
            W, _ = torch.linalg.qr(Mp)
            ar = torch.arange(KMAX, device=dev)[None, :]
            mask = torch.sigmoid((bnd[:, None] - ar - 0.5) / tau)
            bi = torch.randint(0, n, (BS,), device=dev)
            si = torch.randint(0, n, (BS,), device=dev)
            hb, hs = H[:, bi], H[:, si]
            hp = masked_swap(hb, hs, W, mask)
            lp = ro(hp)
            with torch.no_grad():
                tgt = ro(hs)
            loss = (tgt.exp() * (tgt - lp)).sum(-1).mean(-1) + LAM * bnd
            opt.zero_grad(set_to_none=True)
            loss.sum().backward()
            opt.step()
        with torch.no_grad():
            W, _ = torch.linalg.qr(Mp)
            ar = torch.arange(KMAX, device=dev)[None, :]
            # The directions in use after training, and the removal of directions that the interchange does not need.
            mask_h = (torch.sigmoid((bnd[:, None] - ar - 0.5) / 0.05) > 0.5).float()
            TOL = 2e-5
            for _round in range(KMAX):
                base_kl = swap_kl(H, W, mask_h, ro, iters=8)
                best_gain, best_j = None, None
                for j in range(KMAX):
                    col_active = mask_h[:, j] > 0.5
                    if not bool(col_active.any()):
                        continue
                    m2 = mask_h.clone(); m2[:, j] = 0.0
                    kl2 = swap_kl(H, W, m2, ro, iters=8)
                    gain = np.where(col_active.cpu().numpy(), kl2 - base_kl, np.inf)
                    med = np.median(gain[np.isfinite(gain)]) if np.isfinite(gain).any() else np.inf
                    if best_gain is None or med < best_gain:
                        best_gain, best_j, best_active = med, j, col_active
                if best_gain is None or best_gain >= TOL:
                    break
                mask_h[:, best_j] = 0.0
            res[f"{site}_bdim"] = mask_h.sum(-1).cpu().numpy()
            res[f"{site}_kl_bdas"] = swap_kl(H, W, mask_h, ro)
            # Compare the learned frame with the derived subspace, and collect the additional directions of every
            # model.
            ones = torch.ones(Mn, V - 1, device=dev)
            res[f"{site}_kl_derived"] = swap_kl(H, Qd[site], ones, ro)
            Q = Qd[site]
            n_extra, svmins, cdims, Elist = [], [], [], []
            for m in range(Mn):
                Wa = W[m][:, mask_h[m] > 0.5]
                if Wa.shape[1] == 0:
                    n_extra.append(0.0); svmins.append(0.0); cdims.append(0.0); Elist.append(None); continue
                sv = torch.linalg.svdvals((Q[m].T @ Wa).cpu())
                svmins.append(float(sv.min()) if len(sv) else 0.0)
                cdims.append(float((sv > 0.7).sum()))
                R = Wa - Q[m] @ (Q[m].T @ Wa)
                U8, S8, _ = torch.linalg.svd((R.cpu()), full_matrices=False)
                keep = S8 > 0.3
                E = U8[:, keep].to(dev)
                n_extra.append(float(keep.sum()))
                Elist.append(E if E.shape[1] else None)
            res[f"{site}_n_extra"] = np.array(n_extra)
            res[f"{site}_content_svmin"] = np.array(svmins)
            res[f"{site}_content_dim"] = np.array(cdims)
            # The effect of interchanging only the additional directions.
            kmaxE_all = max((0 if E is None else E.shape[1]) for E in Elist)
            if kmaxE_all:
                Epad0 = torch.zeros(Mn, 128, kmaxE_all, device=dev)
                emask0 = torch.zeros(Mn, kmaxE_all, device=dev)
                for m, E in enumerate(Elist):
                    if E is not None:
                        Epad0[m, :, :E.shape[1]] = E
                        emask0[m, :E.shape[1]] = 1.0
                res[f"{site}_extra_kl_raw"] = swap_kl(H, Epad0, emask0, ro, vs_base=True)
            else:
                res[f"{site}_extra_kl_raw"] = np.zeros(Mn)
            # At resid_post: the same interchange with the length restored, and the relation of the first additional
            # direction to log rms(h_post).
            if site == "post":
                kmaxE = max((0 if E is None else E.shape[1]) for E in Elist)
                if kmaxE:
                    Epad = torch.zeros(Mn, 128, kmaxE, device=dev)
                    emask = torch.zeros(Mn, kmaxE, device=dev)
                    for m, E in enumerate(Elist):
                        if E is not None:
                            Epad[m, :, :E.shape[1]] = E
                            emask[m, :E.shape[1]] = 1.0
                    res["post_extra_kl"] = swap_kl(H, Epad, emask, ro, vs_base=True)
                    res["post_extra_kl_normfix"] = swap_kl(H, Epad, emask, ro, normfix=True, vs_base=True)
                    co = torch.einsum("mnd,mdk->mnk", H, Epad)
                    r2s = []
                    for m, E in enumerate(Elist):
                        if E is None:
                            r2s.append(np.nan); continue
                        c = co[m][:, 0]
                        lr = 0.5 * torch.log(H[m].pow(2).mean(-1) + 1e-12)
                        cc = torch.corrcoef(torch.stack([c, lr]))[0, 1]
                        r2s.append(float(cc ** 2))
                    res["post_extra_r2"] = np.array(r2s)
                else:
                    res["post_extra_kl"] = np.zeros(Mn)
                    res["post_extra_kl_normfix"] = np.zeros(Mn)
                    res["post_extra_r2"] = np.full(Mn, np.nan)

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"boundless_das_v{d_vocab}_{n_layers}l_h{n_heads}.npz")
    np.savez(out, tags=np.array(tags), **res)
    for site in ("post", "final"):
        print(f"  {site}: bdim med {np.median(res[f'{site}_bdim']):.1f} | "
              f"KL bdas {np.median(res[f'{site}_kl_bdas']):.2e} "
              f"derived {np.median(res[f'{site}_kl_derived']):.2e} | "
              f"extra {np.median(res[f'{site}_n_extra']):.1f}", flush=True)
    if "post_extra_kl" in res:
        ok = ~np.isnan(res["post_extra_r2"])
        print(f"  post extra: swap KL {np.nanmedian(res['post_extra_kl']):.2e} "
              f"normfix {np.nanmedian(res['post_extra_kl_normfix']):.2e} "
              f"r2(len) {np.nanmedian(res['post_extra_r2']):.2f} "
              f"(models with extra: {ok.sum()}/{len(ok)})", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    dv, nl, nh = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    ckpt = sys.argv[4] if len(sys.argv) > 4 else "results/train/free"
    outd = sys.argv[5] if len(sys.argv) > 5 else "results/boundless_das/free"
    main(dv, nl, nh, ckpt, outd)
