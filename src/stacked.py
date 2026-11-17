"""Many transformers of the same architecture evaluated and trained as one batched computation, with intervention sites
in the final layer and in the readout.

Paper section: Section 2 with Figure 1a (the hooks), and Appendix D (Architecture, Training, and Pinned readout).

Claim: In the paper's words, Figure 1a shows "The transformer architecture's final layer and readout, which converts
the residual stream into next-token output probabilities", and for the pinned models "we fix the unembedding to an
orthonormal simplex frame with zero bias" and "We fix the final RMSNorm's γ = 1".

Experiment: This module runs no experiment. Every experiment runs its models through StackedModels.forward and passes
functions that replace the activation at a hook.

Saved Outputs: None. This module writes no files.

How the code works: StackedModels stacks the weights of M models along a leading axis and runs them together. forward
mirrors the TransformerLens forward pass (RMSNorm with epsilon 1e-5, rotary attention, and GELU) and returns the
logits of every model. Its edits argument maps a site to a function that receives the activation at that site and
returns its replacement. The sites of the final layer are resid_mid, mlp_in, resid_mid_direct, and resid_post, where
resid_mid_direct is resid_mid on the residual path that skips the MLP, so an edit there leaves the MLP's input
unchanged. The sites of the readout are resid_final, after the final RMSNorm, and logits. loss returns each model's
mean next-token loss. _tl_state returns the initial weights of one model from its seed. With frozen_readout it builds
the pinned readout, which is an unembedding whose V + 1 columns form a centered simplex frame in a fixed random
orthonormal basis, a zero bias, and a final RMSNorm weight of one. export_model returns the weights of one model in
the TransformerLens format, with the pinned model's learnable scalar multiplied into the unembedding. GpuHmmSampler
and GpuGhmmSampler sample training sequences on the GPU, the first by rolling the hidden state of an HMM and the
second by exact filtering.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .models import make_model


# Initial weights of one model. The pinned readout is the same for every model.
def _tl_state(n_layers, V, n_heads, seed, frozen_readout=False):
    m = make_model(n_layers, V, n_heads=n_heads, seed=seed)
    sd = {k: v.clone() for k, v in m.state_dict().items()
          if not k.endswith("attn.mask") and "IGNORE" not in k and "rotary" not in k}
    if frozen_readout:
        dv = sd["unembed.W_U"].shape[1]
        d = sd["unembed.W_U"].shape[0]
        g = torch.Generator().manual_seed(1234)
        Q, _ = torch.linalg.qr(torch.randn(d, dv, generator=g))
        C = torch.eye(dv) - torch.ones(dv, dv) / dv
        sd["unembed.W_U"] = (Q @ C) * (0.8 / d ** 0.5) * dv ** 0.5
        sd["unembed.b_U"] = torch.zeros(dv)
        sd["ln_final.w"] = torch.ones(d)
    return sd


class StackedModels:
    def __init__(self, n_layers, V, n_heads, seeds_states, device="cuda"):
        self.n_layers, self.V, self.n_heads = n_layers, V, n_heads
        self.M = len(seeds_states)
        self.device = device
        self.d_head = 128 // n_heads
        self.keys = list(seeds_states[0].keys())
        self.params = {}
        for k in self.keys:
            t = torch.stack([sd[k] for sd in seeds_states]).to(device)
            t.requires_grad_(True)
            self.params[k] = t
        # Rotary tables and the causal mask, shared by all models.
        n_ctx = 128
        pos = torch.arange(n_ctx, dtype=torch.float32)
        dim = torch.arange(self.d_head // 2, dtype=torch.float32)
        freq = 10000 ** (dim / (self.d_head / 2))
        freq = torch.cat([freq, freq])
        angles = pos[:, None] / freq[None, :]
        self.rsin = torch.sin(angles).to(device)
        self.rcos = torch.cos(angles).to(device)
        mask = torch.triu(torch.ones(n_ctx, n_ctx, dtype=torch.bool), diagonal=1)
        self.causal_neg = torch.where(mask, torch.tensor(float("-inf")), 0.0).to(device)

    def _rot(self, x):
        n = self.d_head // 2
        xf = torch.cat([-x[..., n:], x[..., :n]], dim=-1)
        P = x.shape[2]
        cos = self.rcos[:P][None, None, :, None, :]
        sin = self.rsin[:P][None, None, :, None, :]
        return x * cos + xf * sin

    # The forward pass of all models. An edit function, when given for a site, replaces the activation there.
    def forward(self, tokens, edits=None):
        p = self.params
        edits = edits or {}
        Mi = torch.arange(self.M, device=tokens.device)[:, None, None]
        x = p["embed.W_E"][Mi, tokens]
        eps = 1e-5
        for l in range(self.n_layers):
            pre = f"blocks.{l}."
            s = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
            xn = x / s * p[pre + "ln1.w"][:, None, None, :]
            q = torch.einsum("mbpd,mhdk->mbphk", xn, p[pre + "attn.W_Q"]) + \
                p[pre + "attn.b_Q"][:, None, None, :, :]
            k = torch.einsum("mbpd,mhdk->mbphk", xn, p[pre + "attn.W_K"]) + \
                p[pre + "attn.b_K"][:, None, None, :, :]
            v = torch.einsum("mbpd,mhdk->mbphk", xn, p[pre + "attn.W_V"]) + \
                p[pre + "attn.b_V"][:, None, None, :, :]
            q, k = self._rot(q), self._rot(k)
            Mb, Bb, P_, H, dh = q.shape
            qf = q.permute(0, 1, 3, 2, 4).reshape(Mb * Bb, H, P_, dh)
            kf = k.permute(0, 1, 3, 2, 4).reshape(Mb * Bb, H, P_, dh)
            vf = v.permute(0, 1, 3, 2, 4).reshape(Mb * Bb, H, P_, dh)
            zf = F.scaled_dot_product_attention(qf, kf, vf, is_causal=True,
                                                scale=1.0 / math.sqrt(self.d_head))
            z = zf.reshape(Mb, Bb, H, P_, dh).permute(0, 1, 3, 2, 4)
            attn_out = torch.einsum("mbphk,mhkd->mbpd", z, p[pre + "attn.W_O"]) + \
                p[pre + "attn.b_O"][:, None, None, :]
            x = x + attn_out
            last = (l == self.n_layers - 1)
            # resid_mid of the final layer, before the residual stream splits into the MLP branch and the path around
            # it.
            if last and "resid_mid" in edits:
                x = edits["resid_mid"](x)
            s = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
            xn = x / s * p[pre + "ln2.w"][:, None, None, :]
            # mlp_in of the final layer, after the MLP's own RMSNorm.
            if last and "mlp_in" in edits:
                xn = edits["mlp_in"](xn)
            h = torch.einsum("mbpd,mdf->mbpf", xn, p[pre + "mlp.W_in"]) + \
                p[pre + "mlp.b_in"][:, None, None, :]
            h = F.gelu(h)
            mlp_out = torch.einsum("mbpf,mfd->mbpd", h, p[pre + "mlp.W_out"]) + \
                p[pre + "mlp.b_out"][:, None, None, :]
            # resid_mid on the path around the MLP. The MLP has already read its input.
            if last and "resid_mid_direct" in edits:
                x = edits["resid_mid_direct"](x)
            x = x + mlp_out
            # resid_post of the final layer.
            if last and "resid_post" in edits:
                x = edits["resid_post"](x)
        s = (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
        xn = x / s * p["ln_final.w"][:, None, None, :]
        # resid_final, after the final RMSNorm.
        if "resid_final" in edits:
            xn = edits["resid_final"](xn)
        WU = p["unembed.W_U"]
        # The learnable scalar of a pinned model multiplies its fixed unembedding.
        if "unembed.scale" in p:
            WU = WU * p["unembed.scale"][:, None, None].exp()
        logits = torch.einsum("mbpd,mdv->mbpv", xn, WU) + \
            p["unembed.b_U"][:, None, None, :]
        if "logits" in edits:
            logits = edits["logits"](logits)
        return logits

    def loss(self, tokens):
        logits = self.forward(tokens)
        Mn, B, P, dv = logits.shape
        lg = logits[:, :, :-1, :].reshape(Mn, -1, dv)
        tg = tokens[:, :, 1:].reshape(Mn, -1)
        return F.cross_entropy(lg.transpose(1, 2), tg, reduction="none").mean(-1)

    def export_model(self, m):
        sd = {k: self.params[k][m].detach().cpu().clone() for k in self.keys
              if k != "unembed.scale"}
        if "unembed.scale" in self.params:
            sd["unembed.W_U"] = sd["unembed.W_U"] * \
                self.params["unembed.scale"][m].detach().cpu().exp()
        return sd


# Samplers of training sequences on the GPU.
class GpuHmmSampler:
    def __init__(self, M, pi, device, seed):
        V, S, _ = M.shape
        self.V, self.S = V, S
        flat = np.cumsum(M.transpose(1, 0, 2).reshape(S, V * S), axis=1)
        self.flat = torch.tensor(flat, dtype=torch.float64, device=device)
        p0 = np.clip(pi, 0, None); p0 = p0 / p0.sum()
        self.p0 = torch.tensor(p0, dtype=torch.float64, device=device)
        self.g = torch.Generator(device=device); self.g.manual_seed(seed)
        self.device = device

    def sample(self, n, L):
        s = torch.multinomial(self.p0.expand(n, -1), 1, generator=self.g).squeeze(1)
        out = torch.zeros((n, L), dtype=torch.long, device=self.device)
        u = torch.rand((n, L), generator=self.g, device=self.device, dtype=torch.float64)
        VS = self.V * self.S
        for t in range(L):
            k = (u[:, t:t + 1] > self.flat[s]).sum(1)
            k = torch.clamp(k, max=VS - 1)
            out[:, t] = k // self.S
            s = k % self.S
        return out


class GpuGhmmSampler:
    def __init__(self, M, pi, eta, device, seed):
        self.M = torch.tensor(M, dtype=torch.float64, device=device)
        self.Meta = torch.tensor(np.einsum("zij,j->zi", M, eta), dtype=torch.float64,
                                 device=device)
        self.pi = torch.tensor(pi, dtype=torch.float64, device=device)
        self.eta = torch.tensor(eta, dtype=torch.float64, device=device)
        self.V = M.shape[0]
        self.g = torch.Generator(device=device); self.g.manual_seed(seed)
        self.device = device

    def sample(self, n, L):
        b = self.pi.expand(n, -1).clone()
        out = torch.zeros((n, L), dtype=torch.long, device=self.device)
        for t in range(L):
            pz = torch.einsum("ni,zi->nz", b, self.Meta).clamp_min(0)
            pz = pz / pz.sum(1, keepdim=True)
            z = torch.multinomial(pz, 1, generator=self.g).squeeze(1)
            out[:, t] = z
            bn = torch.einsum("ni,nij->nj", b, self.M[z])
            b = bn / (bn @ self.eta).clamp_min(1e-300)[:, None]
        return out
