"""Training: every model of one group (the same vocabulary size, number of layers, and number of heads) is trained
together on one GPU.

Paper section: Section 3 (Experiments, Models) and Appendix D (Architecture, Training, and Pinned readout).

Claim: In the paper's words, "Each training step uses a batch of 1024 sequences of length 128. We train with AdamW
(weight decay 0.01) and a cosine learning-rate decay from 5 × 10^−4 to 5 × 10^−5 for at most 60,000 steps. Every 2,000
steps we evaluate the held-out loss, and we stop early when it fails to improve on its best value by more than 10^−5
for four consecutive evaluations." For the pinned models, "We add an additional loss term 0.3 · E[log^2 rms(h_post)],
which holds rms(h_post) near one at every token position".

Experiment: For one vocabulary size, number of layers, and number of heads, the script trains one model per process
and seed on sequences sampled from that process. Each sequence is a BOS token followed by 127 symbols. Every 2,000
steps each model's loss is measured on 8,192 held-out sequences of its process, and the weights with the lowest held-
out loss are kept. With PULLBACK_FROZEN_READOUT=1 and PULLBACK_RMS_PENALTY=0.3 the script trains the pinned variant,
whose unembedding, unembedding bias, and final RMSNorm weight are fixed, and which learns one scalar that multiplies
the unembedding.

Saved Outputs: In the output folder, one <process>_<L>l_h<H>_s<seed>.pt file per model with the weights of its best
evaluation in the TransformerLens format, and one .json file per model with the keys name, n_layers, seed, n_heads,
opt_nll (the loss of the exact predictor on the held-out sequences), history (the held-out loss at every evaluation),
best_val, and best_gap (best_val minus opt_nll).

How the code works: The defaults are the paper's settings. The models are stacked along a leading axis
(src/stacked.py). The loss is the sum of the models' own mean losses, so each model receives the gradient of its own
loss, and AdamW acts elementwise, so each model trains as it would alone. Micro-batches over the model axis bound the
memory. Each process has one sampler on the GPU, and each model receives its own sequences. A model stops when its
held-out loss has failed to improve on its best value by more than 1e-5 for four evaluations in a row, and its best
weights are written then. Models whose .json file already exists are skipped. Environment variables override the
settings: PULLBACK_STEPS, PULLBACK_BATCH, PULLBACK_LRMIN, PULLBACK_SEEDS (as lo:hi), PULLBACK_PROCS,
PULLBACK_EARLYSTOP, PULLBACK_PATIENCE, PULLBACK_MIN_DELTA, PULLBACK_FROZEN_READOUT, and PULLBACK_RMS_PENALTY, and for
memory and speed PULLBACK_CHUNK, PULLBACK_MICRO_MULT, PULLBACK_ML_CAP, PULLBACK_BF16, PULLBACK_COMPACT, and
PULLBACK_COMPILE. Usage: python experiments/training/train_models.py <d_vocab> <n_layers> <n_heads> [outdir]
[seed_lo:seed_hi], where d_vocab is V + 1.
"""
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src import processes as P
from src.models import L
from src.stacked import GpuGhmmSampler, GpuHmmSampler, StackedModels, _tl_state

# The paper's settings: at most 60,000 steps, a cosine learning rate from 5e-4 to 5e-5, weight decay 0.01, batches of
# 1,024 sequences, an evaluation on 8,192 held-out sequences every 2,000 steps, and three seeds.
STEPS = int(os.environ.get("PULLBACK_STEPS", "60000"))
LR0, WD = 5e-4, 0.01
LRMIN = float(os.environ.get("PULLBACK_LRMIN", "5e-5"))
BATCH = int(os.environ.get("PULLBACK_BATCH", "1024"))
CHUNK = int(os.environ.get("PULLBACK_CHUNK", "4"))
VAL_N = 8192
VAL_EVERY = 2000
SEEDS = list(range(3))
if os.environ.get("PULLBACK_SEEDS"):
    _lo, _hi = os.environ["PULLBACK_SEEDS"].split(":")
    SEEDS = list(range(int(_lo), int(_hi)))
PROCS = [x for x in os.environ.get("PULLBACK_PROCS", "").split(",") if x]
# The number of models per micro-batch, by number of layers.
MICRO = {1: 96, 2: 72, 3: 48, 4: 36, 5: 30}


# A view of some of the stacked models, used for micro-batches.
class SliceView:
    def __init__(self, base, sl):
        self.n_layers, self.V, self.n_heads = base.n_layers, base.V, base.n_heads
        self.d_head, self.device = base.d_head, base.device
        self.rsin, self.rcos, self.causal_neg = base.rsin, base.rcos, base.causal_neg
        self.keys = base.keys
        if isinstance(sl, list) and sl == list(range(sl[0], sl[-1] + 1)):
            sl = slice(sl[0], sl[-1] + 1)
        self.M = (sl.stop - sl.start) if isinstance(sl, slice) else len(sl)
        self.params = {k: v[sl] for k, v in base.params.items()}
    _rot = StackedModels._rot
    forward = StackedModels.forward


# The pinned variant: a fixed readout and the penalty on log^2 rms(resid_post).
FROZEN = os.environ.get("PULLBACK_FROZEN_READOUT", "0") == "1"
RMS_PEN = float(os.environ.get("PULLBACK_RMS_PENALTY", "0"))
FROZEN_KEYS = ("unembed.W_U", "unembed.b_U", "ln_final.w")
# Early stopping: four evaluations in a row without an improvement of more than 1e-5 on the best held-out loss.
EARLY = os.environ.get("PULLBACK_EARLYSTOP", "1") == "1"
BF16 = os.environ.get("PULLBACK_BF16", "0") == "1"
PATIENCE = int(os.environ.get("PULLBACK_PATIENCE", "4"))
MIN_DELTA = float(os.environ.get("PULLBACK_MIN_DELTA", "1e-5"))
COMPACT = os.environ.get("PULLBACK_COMPACT", "0") == "1"


def main(d_vocab, n_layers, n_heads, outdir, seeds=None):
    V = d_vocab - 1
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    os.makedirs(outdir, exist_ok=True)

    # One job per process of this vocabulary size and seed. Finished models are skipped.
    specs = [s for s in P.all_sweep_processes() if s["V"] == V]
    if PROCS:
        specs = [s for s in specs if s["name"] in PROCS]
    jobs = []
    use_seeds = SEEDS if seeds is None else seeds
    for spec in specs:
        for seed in use_seeds:
            tag = f"{spec['name']}_{n_layers}l_h{n_heads}_s{seed}"
            if not os.path.exists(os.path.join(outdir, f"{tag}.json")):
                jobs.append(dict(spec=spec, seed=seed, tag=tag))
    if not jobs:
        print("group complete, nothing to do", flush=True)
        return
    Mn = len(jobs)
    mm = float(os.environ.get("PULLBACK_MICRO_MULT", "1"))
    micro = max(1, int(MICRO[n_layers] * 128 * (2 if BF16 else 1) * mm) // BATCH)
    if BATCH >= 512:
        micro = max(1, min(micro, int(os.environ.get("PULLBACK_ML_CAP", "26")) // n_layers))
    print(f"group V={V} {n_layers}l h{n_heads}: {Mn} models (micro {micro})", flush=True)

    # Initial weights of every model, stacked.
    stack = StackedModels(n_layers, V, n_heads,
                          [_tl_state(n_layers, V, n_heads, j["seed"],
                                     frozen_readout=FROZEN) for j in jobs],
                          device=dev)
    # Pinned models do not train their readout. One scalar per model multiplies the fixed unembedding.
    if FROZEN:
        for k in FROZEN_KEYS:
            stack.params[k].requires_grad_(False)
        sc = torch.zeros(Mn, device=dev)
        sc.requires_grad_(True)
        stack.params["unembed.scale"] = sc
        stack.keys.append("unembed.scale")
        print("frozen readout: gamma_final=1, W_U=simplex frame, b_U=0; "
              "global scalar learnable", flush=True)

    # One sampler and one held-out set per process, and the loss of the exact predictor on the held-out set.
    by_proc = {}
    for i, j in enumerate(jobs):
        by_proc.setdefault(j["spec"]["name"], []).append(i)
    samplers, val_t, opt_nll = {}, {}, {}
    for name, idxs in by_proc.items():
        spec = jobs[idxs[0]]["spec"]
        M, pi, eta = spec["M"], spec["pi"], spec["eta"]
        is_hmm = bool(np.all(M >= -1e-12))
        sd = 99000 + (abs(hash(name)) % 10000)
        samplers[name] = (GpuHmmSampler(M, pi, dev, seed=sd) if is_hmm
                          else GpuGhmmSampler(M, pi, eta, dev, seed=sd))
        rng = np.random.default_rng(555)
        val = (P.sample_fast(M, VAL_N, L, rng, pi=pi) if is_hmm
               else P.sample(M, VAL_N, L, rng, eta=eta, pi=pi))
        opt_nll[name] = float(P.plugin_nll(M, val, eta=eta, pi=pi))
        val_t[name] = torch.tensor(np.concatenate([np.full((VAL_N, 1), V), val], 1),
                                   device=dev)
        print(f"  {name}: {len(idxs)} seeds, opt_nll {opt_nll[name]:.6f}", flush=True)

    # Training sequences for the next steps: a BOS token followed by 127 sampled symbols, separate for every model.
    def gen_chunk():
        toks = torch.empty((Mn, CHUNK * BATCH, L + 1), dtype=torch.long, device=dev)
        toks[:, :, 0] = V
        for name, idxs in by_proc.items():
            seqs = samplers[name].sample(len(idxs) * CHUNK * BATCH, L)
            toks[idxs, :, 1:] = seqs.view(len(idxs), CHUNK * BATCH, L)
        return [toks[:, c * BATCH:(c + 1) * BATCH] for c in range(CHUNK)]

    # Each model's mean next-token loss, plus the penalty on log^2 rms(resid_post) for the pinned variant.
    def ce_per_model(view, tk):
        aux = {}
        if RMS_PEN > 0:
            def cap(x):
                aux["logrms2"] = x.pow(2).mean(-1).add(1e-9).sqrt().log().pow(2)
                return x
            lg = view.forward(tk, edits={"resid_post": cap})
        else:
            lg = view.forward(tk)
        lgf = lg[:, :, :-1, :].reshape(view.M, -1, d_vocab)
        tg = tk[:, :, 1:].reshape(view.M, -1)
        ce = F.cross_entropy(lgf.transpose(1, 2), tg, reduction="none").mean(-1)
        if RMS_PEN > 0:
            ce = ce + RMS_PEN * aux["logrms2"].mean((-1, -2))
        return ce

    @torch.no_grad()
    # Each model's loss on the held-out sequences of its process.
    def val_losses():
        out = torch.zeros(Mn)
        for name, idxs in by_proc.items():
            vt = val_t[name]
            tot = torch.zeros(len(idxs))
            for s0 in range(0, VAL_N, 256):
                bb = vt[s0:s0 + 256]
                for m0 in range(0, len(idxs), micro):
                    sl = idxs[m0:m0 + micro]
                    view = SliceView(stack, sl)
                    tk = bb[None].repeat(len(sl), 1, 1)
                    with torch.autocast("cuda", torch.bfloat16, enabled=BF16):
                        lg = view.forward(tk)
                    lg = lg.float()
                    lgf = lg[:, :, :-1, :].reshape(len(sl), -1, d_vocab)
                    tg = tk[:, :, 1:].reshape(len(sl), -1)
                    tot[m0:m0 + micro] += F.cross_entropy(
                        lgf.transpose(1, 2), tg, reduction="none").sum(-1).cpu()
            out[idxs] = tot / (VAL_N * L)
        return out

    fused_ok = dev == "cuda"
    trainable = [v for v in stack.params.values() if v.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=LR0, weight_decay=WD,
                            fused=fused_ok)
    if dev == "cuda" and os.environ.get("PULLBACK_COMPILE", "1") == "1":
        mode = os.environ.get("PULLBACK_COMPILE_MODE", "default")
        if RMS_PEN > 0 and mode == "reduce-overhead":
            mode = "default"
        try:
            if not getattr(StackedModels, "_compiled", False):
                StackedModels.forward = torch.compile(
                    StackedModels.forward, dynamic=False,
                    mode=(mode if mode != "default" else None))
                SliceView.forward = StackedModels.forward
                StackedModels._compiled = True
            print(f"torch.compile enabled ({mode})", flush=True)
        except Exception as e:
            print(f"torch.compile unavailable: {e}", flush=True)
    best = np.full(Mn, np.inf)
    best_state = [None] * Mn
    history = [[] for _ in range(Mn)]
    stale = np.zeros(Mn, dtype=int)
    done_mask = torch.ones(Mn, device=dev)
    n_final = 0

    # Write a model's best weights and its training record.
    def finalize(m):
        j = jobs[m]
        tag, name = j["tag"], j["spec"]["name"]
        torch.save(best_state[m], os.path.join(outdir, f"{tag}.pt"))
        with open(os.path.join(outdir, f"{tag}.json"), "w") as f:
            json.dump({"name": name, "n_layers": n_layers, "seed": j["seed"],
                       "n_heads": n_heads, "opt_nll": opt_nll[name],
                       "history": history[m], "best_val": float(best[m]),
                       "best_gap": float(best[m] - opt_nll[name])}, f)

    # Remove finished models from the stack and carry the optimizer state of the others.
    def compact(keep):
        nonlocal jobs, Mn, best, best_state, history, stale, done_mask, by_proc, opt
        old = stack.params
        key_order = [k for k in stack.keys if old[k].requires_grad]
        newp = {}
        for k, v in old.items():
            nv = v.detach()[keep].clone()
            nv.requires_grad_(v.requires_grad)
            newp[k] = nv
        lr_now = opt.param_groups[0]["lr"]
        newopt = torch.optim.AdamW([newp[k] for k in key_order], lr=lr_now,
                                   weight_decay=WD, fused=fused_ok)
        for k in key_order:
            st = opt.state.get(old[k])
            if st:
                newopt.state[newp[k]] = {
                    "step": st["step"].clone() if torch.is_tensor(st["step"]) else st["step"],
                    "exp_avg": st["exp_avg"][keep].clone(),
                    "exp_avg_sq": st["exp_avg_sq"][keep].clone()}
        stack.params = newp
        opt = newopt
        jobs = [jobs[i] for i in keep]
        Mn = len(jobs)
        best = best[keep]
        best_state = [best_state[i] for i in keep]
        history = [history[i] for i in keep]
        stale = stale[keep]
        done_mask = torch.ones(Mn, device=dev)
        by_proc = {}
        for i, j in enumerate(jobs):
            by_proc.setdefault(j["spec"]["name"], []).append(i)

    chunk, ci = None, CHUNK

    # The training loop.
    for step in range(STEPS):
        lr = LRMIN + 0.5 * (LR0 - LRMIN) * (1 + math.cos(math.pi * step / STEPS))
        for g in opt.param_groups:
            g["lr"] = lr
        if ci >= CHUNK:
            chunk, ci = gen_chunk(), 0
        tk = chunk[ci]; ci += 1
        opt.zero_grad(set_to_none=True)
        for m0 in range(0, Mn, micro):
            sl = list(range(m0, min(m0 + micro, Mn)))
            view = SliceView(stack, sl)
            with torch.autocast("cuda", torch.bfloat16, enabled=BF16):
                loss = (ce_per_model(view, tk[sl]) * done_mask[sl]).sum()
            loss.backward()
        opt.step()
        # Evaluation: keep each model's best weights, and stop the models that have converged.
        if step % VAL_EVERY == 0 or step == STEPS - 1:
            vl = val_losses().numpy()
            prev_best = best.copy()
            for m in range(Mn):
                history[m].append({"step": step, "val": float(vl[m])})
                if vl[m] < best[m]:
                    best[m] = vl[m]
                    best_state[m] = stack.export_model(m)
            gaps = [vl[m] - opt_nll[jobs[m]["spec"]["name"]] for m in range(Mn)]
            print(f"step {step}: {Mn} active, median gap {np.median(gaps):+.6f} "
                  f"max {np.max(gaps):+.6f}", flush=True)
            if EARLY and step > 0:
                for m in range(Mn):
                    imp = prev_best[m] - vl[m]
                    stale[m] = 0 if imp > MIN_DELTA else stale[m] + 1
                conv = [m for m in range(Mn) if stale[m] >= PATIENCE
                        and done_mask[m].item() > 0]
                for m in conv:
                    finalize(m)
                    done_mask[m] = 0.0
                n_now = int((done_mask == 0).sum())
                if n_now == Mn:
                    print(f"all models converged at step {step}", flush=True)
                    jobs = []
                    break
                if COMPACT and n_now >= max(4, Mn // 4):
                    keep = [m for m in range(Mn) if done_mask[m].item() > 0]
                    compact(keep)
                    ci = CHUNK
                    print(f"compacted to {Mn} active models", flush=True)

    for m in range(len(jobs)):
        if done_mask[m].item() > 0:
            finalize(m)
    print("group done", flush=True)


if __name__ == "__main__":
    d_vocab, n_layers, n_heads = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    outdir = sys.argv[4] if len(sys.argv) > 4 else "results/train/free"
    seeds = None
    if len(sys.argv) > 5:
        lo, hi = sys.argv[5].split(":")
        seeds = list(range(int(lo), int(hi)))
    main(d_vocab, n_layers, n_heads, outdir, seeds)
