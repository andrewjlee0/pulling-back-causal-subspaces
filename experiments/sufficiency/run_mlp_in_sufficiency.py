"""Causal sufficiency at mlp_in: minimizing the KL divergence to a target output within the estimated causal subspace,
and within random directions orthogonal to it.

Paper section: Appendix G (Causal sufficiency at mlp_in), referenced from Section 3 (Causal necessity and
sufficiency).

Claim: In the paper's words, "In free models, the median KL across token positions is 6.4 × 10^−4 nats within the
estimated subspace and 4.6 × 10^−2 nats within the control directions", and "In pinned models, the median KL is 1.1 ×
10^−7 nats, and the KL remains above 10^−3 nats at 4% of positions."

Experiment: The targets start as the targets of the resid_post sufficiency test: flat Dirichlet draws over the V
process symbols with 0.01 added, the BOS probability of each model kept, and the centered logits scaled down where
resid_post could not produce them. Three minimizations of the KL divergence from the target to the model's output then
run with Adam from a zero perturbation of mlp_in, each over a different set of directions. The first, full, uses all
directions of mlp_in, and the output it reaches replaces the target, so that every target can be produced from mlp_in.
The second, sub, uses only the k99 directions of the estimated causal subspace. The third, ctrl, uses the same number
of random directions orthogonal to that subspace. A minimization stops when the KL divergence, averaged over all token
positions of all models of the group, improves by less than 0.1 percent over 200 steps, or after 16,000 steps. At
every step the perturbed activation is rescaled so that, divided entrywise by the weight of the MLP's RMSNorm, it has
the length of the original activation.

Saved Outputs: mlp_in_sufficiency_v<d_vocab>_<L>l_h<H>.npz in the output folder, with tags, k99 and k_ctrl (the
numbers of subspace and control directions per model), temp (the factor by which each target's centered logits were
scaled down, per model and token position), kl_full_dirichlet (the KL divergence from the Dirichlet target to the
output reached over all directions), base_kl (the KL divergence from the final target to the unedited output),
kl_full, kl_sub, and kl_ctrl (the KL divergence from the final target to the output after each minimization, per model
and token position), polish_steps_full, polish_steps_sub, and polish_steps_ctrl (the number of Adam steps taken), and
the per-model means and medians full_kl_mean, full_kl_median, sub_kl_mean, sub_kl_median, ctrl_kl_mean,
ctrl_kl_median, base_kl_mean, and base_kl_median.

How the code works: The script records mlp_in and the residual path around the MLP in one forward pass, and recomputes
resid_post and the logits from mlp_in in closed form, which it checks against the model's logits. The sensitivity
matrix, its eigenvectors, and k99 are computed as in experiments/necessity/run_mlp_in_negation.py, here over all 128
positions of each context. The control directions are random directions made orthogonal to the estimated subspace,
drawn with a fixed seed. polish runs Adam with learning rate 0.02 in windows of 200 steps and stops on the criterion
above. The settings SUFF_POLISH_MAX, SUFF_POLISH_WIN, SUFF_TOL, and SUFF_POLISH_LR can be set as environment
variables. Usage: python experiments/sufficiency/run_mlp_in_sufficiency.py <d_vocab> <n_layers> <n_heads> <ckpt_dir>
<out_dir>, where d_vocab is V + 1.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src import pullback as PB
from src.evaluation import context_tokens, load_group


# GELU in its exact form, as in the models.
def gelu(z):
    return 0.5 * z * (1.0 + torch.erf(z / np.sqrt(2.0)))


def main(d_vocab, n_layers, n_heads, ckpt_dir, out_dir):
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available() else "cpu")
    V = d_vocab - 1
    stack, tags, _ = load_group(d_vocab, n_layers, n_heads, ckpt_dir, dev)
    tokens = context_tokens(tags, V, dev)
    Mn = stack.M
    B_, P_ = tokens.shape[1], tokens.shape[2]
    N = B_ * P_
    print(f"mlp_in sufficiency v{d_vocab} {n_layers}l h{n_heads}: {Mn} models on {dev}", flush=True)

    # Record mlp_in and the residual path around the MLP (resid_mid_direct) in the unedited forward pass.
    rec = {}
    def rec_mlp(x):
        rec["mlp_in"] = x
        return x
    def rec_mid(x):
        rec["mid"] = x
        return x
    with torch.no_grad():
        clean = stack.forward(tokens, edits={"mlp_in": rec_mlp,
                                             "resid_mid_direct": rec_mid})
    xm = rec["mlp_in"].detach().float()
    mid = rec["mid"].detach().float()
    d_model = xm.shape[-1]

    pre = f"blocks.{n_layers-1}."
    W_in = stack.params[pre + "mlp.W_in"].float()
    b_in = stack.params[pre + "mlp.b_in"].float()
    W_out = stack.params[pre + "mlp.W_out"].float()
    b_out = stack.params[pre + "mlp.b_out"].float()
    gam_f = stack.params["ln_final.w"].float()
    WU = stack.params["unembed.W_U"].float()
    bU = stack.params["unembed.b_U"].float()

    # resid_post and the logits as functions of mlp_in, with the residual path held at its recorded value.
    def resid_post(x):
        return mid + torch.einsum("mnf,mfd->mnd",
                                  gelu(torch.einsum("mnd,mdf->mnf", x, W_in)
                                       + b_in[:, None, :]), W_out) + b_out[:, None, :]

    def out_logits(x):
        h = resid_post(x)
        rms = h.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()
        return torch.einsum("mnd,mdv->mnv", (h / rms) * gam_f[:, None, :], WU) \
            + bU[:, None, :]

    mid = mid.reshape(Mn, N, d_model)
    X = xm.reshape(Mn, N, d_model)
    # A perturbation of mlp_in along the columns of Qb, rescaled to the length that the original activation has after
    # division by the weight of the MLP's RMSNorm.
    gam_m = stack.params[pre + "ln2.w"].float()[:, None, :]
    shell_r = (X / gam_m).norm(dim=-1, keepdim=True)
    def xof(delta, Qb):
        xp = X + torch.einsum("mnk,mdk->mnd", delta, Qb)
        xp = xp * (shell_r / (xp / gam_m).norm(dim=-1, keepdim=True).clamp_min(1e-12))
        return xp
    clean_flat = clean.reshape(Mn, N, -1).float()
    with torch.no_grad():
        # Check that the closed-form readout reproduces the model's output.
        ident = (out_logits(X).log_softmax(-1) - clean_flat.log_softmax(-1)).abs().max()
    identity_err = float(ident)
    assert identity_err < 1e-2, f"readout mismatch {identity_err}"

    with torch.no_grad():
        # The sensitivity matrix and its eigenvectors, sorted from the largest eigenvalue to the smallest.
        Ucen = WU - WU.mean(-1, keepdim=True)
        G = gam_f[:, :, None] * Ucen
        C = torch.einsum("mfd,mdv->mfv", W_out, G)
        z = torch.einsum("mnd,mdf->mnf", X, W_in) + b_in[:, None, :]
        sp = torch.distributions.Normal(0, 1).cdf(z) + z * torch.exp(
            torch.distributions.Normal(0, 1).log_prob(z))
        Ms = torch.zeros(Mn, d_model, d_model, device=dev)
        for v in range(C.shape[-1]):
            Dv = torch.einsum("mdf,mnf->mnd", W_in, sp * C[:, None, :, v])
            Ms += torch.einsum("mnd,mne->mde", Dv, Dv) / N
        evals, evecs = torch.linalg.eigh(Ms.cpu())
        evals = evals.flip(-1); evecs = evecs.flip(-1)

    # k99 per model, the estimated subspace, and the same number of random directions orthogonal to it.
    k99 = np.zeros(Mn, dtype=int)
    for m in range(Mn):
        lam = evals[m]
        mass = torch.cumsum(lam, 0) / lam.sum().clamp_min(1e-30)
        k99[m] = int((mass < 0.99).sum().item()) + 1
    ks = k99
    kc = np.minimum(ks, d_model - ks)
    K = int(ks.max()); Kc = int(kc.max())
    Qpad = torch.zeros(Mn, d_model, K)
    Qcpad = torch.zeros(Mn, d_model, Kc)
    gq = torch.Generator().manual_seed(500)
    for m in range(Mn):
        Q = evecs[m][:, :ks[m]].float()
        Qpad[m, :, :ks[m]] = Q
        Gr = torch.randn(d_model, kc[m], generator=gq)
        Gr = Gr - Q @ (Q.T @ Gr)
        Qc, _ = torch.linalg.qr(Gr)
        Qcpad[m, :, :kc[m]] = Qc
    Qpad = Qpad.to(dev); Qcpad = Qcpad.to(dev)
    Qfull = torch.eye(d_model, device=dev)[None].expand(Mn, d_model, d_model).contiguous()

    with torch.no_grad():
        # The targets of the resid_post sufficiency test: Dirichlet draws, then the scaling of footnote 3 of the
        # paper.
        gt = torch.Generator(device="cpu").manual_seed(123)
        e = -torch.rand((B_, P_, V), generator=gt).log()
        pstar_real = e / e.sum(-1, keepdim=True)
        pstar_real = (pstar_real + 0.01).clamp_min(0.01)
        pstar_real = pstar_real / pstar_real.sum(-1, keepdim=True)
        pstar_real = pstar_real.to(dev)
        p_clean = clean.softmax(-1).float()
        p_bos = p_clean[..., V:V + 1]
        pstar = torch.cat([pstar_real[None] * (1 - p_bos), p_bos], -1).reshape(Mn, N, -1)
        bUc = (bU - bU.mean(-1, keepdim=True))[:, None, :]
        tstar = pstar.clamp_min(1e-12).log()
        tstar = tstar - tstar.mean(-1, keepdim=True) - bUc
        h0 = resid_post(X)
        Qpost = PB.window_basis(stack.params).float()
        Bp = torch.einsum("mdk,mdv->mvk", Qpost, G)
        Bp_pinv = torch.linalg.pinv(Bp.cpu()).to(dev)
        u = torch.einsum("mkv,mnv->mnk", Bp_pinv, tstar)
        un = u.norm(dim=-1, keepdim=True)
        xn = h0.norm(dim=-1, keepdim=True)
        s_orig = (h0.pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()
        temp = (0.95 * xn / (s_orig * un).clamp_min(1e-12)).clamp_max(1.0)
        pstar = (temp * tstar + bUc).softmax(-1)
        base = (pstar * (pstar.clamp_min(1e-12).log()
                         - clean_flat.log_softmax(-1))).sum(-1)

    # The KL divergence from the target to the output produced by an mlp_in activation, per model and token position.
    def kl_of(x):
        lg = out_logits(x)
        return (pstar * (pstar.clamp_min(1e-12).log() - lg.log_softmax(-1))).sum(-1)

    # Adam on the perturbation coordinates, in windows of 200 steps, until the mean KL divergence improves by less
    # than 0.1 percent over a window or the step limit is reached.
    def polish(delta0, Qb):
        max_steps = int(os.environ.get("SUFF_POLISH_MAX", 16000))
        window = int(os.environ.get("SUFF_POLISH_WIN", 200))
        tol = float(os.environ.get("SUFF_TOL", 1e-3))
        delta = delta0.clone().requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=float(os.environ.get("SUFF_POLISH_LR", 0.02)))
        def mean_kl():
            with torch.no_grad():
                return float(kl_of(xof(delta.detach(), Qb)).mean())
        last = mean_kl()
        steps = 0
        trace, kl_mid = [last], None
        while steps < max_steps:
            for _ in range(window):
                opt.zero_grad()
                loss = kl_of(xof(delta, Qb)).mean()
                loss.backward()
                opt.step()
            steps += window
            cur = mean_kl()
            trace.append(cur)
            if steps == max_steps // 2:
                with torch.no_grad():
                    kl_mid = kl_of(xof(delta.detach(), Qb))
            if last - cur < tol * max(last, 1e-12):
                last = cur
                break
            last = cur
        with torch.no_grad():
            d = delta.detach()
            kl_end = kl_of(xof(d, Qb))
            if kl_mid is None:
                kl_mid = kl_end
            return kl_end, d, steps, np.array(trace), kl_mid

    res = {"tags": np.array(tags), "k99": k99.astype(float), "k_ctrl": kc.astype(float)}
    with torch.no_grad():
        res["base_kl"] = base.cpu().numpy()
        res["temp"] = temp[..., 0].cpu().numpy()
    # The three minimizations. After the first, the output reached over all directions becomes the target.
    order = (("full", Qfull), ("sub", Qpad), ("ctrl", Qcpad))
    for name, Qb in order:
        d_n = torch.zeros(Mn, N, Qb.shape[-1], device=dev)
        kl_p, d_p, steps, trace, kl_mid = polish(d_n, Qb)
        if name == "full":
            with torch.no_grad():
                res["kl_full_dirichlet"] = kl_p.cpu().numpy()
                pstar = out_logits(xof(d_p, Qb)).softmax(-1)
                base = (pstar * (pstar.clamp_min(1e-12).log()
                                 - clean_flat.log_softmax(-1))).sum(-1)
                res["base_kl"] = base.cpu().numpy()
                kl_p = kl_of(xof(d_p, Qb))

        a = kl_p.cpu().numpy()
        res[f"kl_{name}"] = a
        res[f"polish_steps_{name}"] = np.full(Mn, float(steps))
        res[f"{name}_kl_mean"] = a.mean(1)
        res[f"{name}_kl_median"] = np.median(a, 1)
        print(f"  {name:4s}: median {np.median(a):.2e} mean {a.mean():.2e} "
              f"frac>1e-3 {np.mean(a > 1e-3):.3f} polish {steps}", flush=True)
    res["base_kl_mean"] = res["base_kl"].mean(1)
    res["base_kl_median"] = np.median(res["base_kl"], 1)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"mlp_in_sufficiency_v{d_vocab}_{n_layers}l_h{n_heads}.npz")
    np.savez(out, **res)
    print(f"  k99 median {np.median(k99):.0f} | scaled-down targets {np.mean(res['temp'] < 1):.3f} | wrote {out}",
          flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
