"""Loading the trained models of one group, and the evaluation contexts on which they are tested.

Paper section: Section 3 (Experiments, Models).

Claim: In the paper's words, "For each process, we train every combination of L ∈ {1, ..., 5} layers and H ∈ {1, 2, 4,
8} attention heads, with three training seeds each".

Experiment: This module runs no experiment. Every experiment script first calls load_group and context_tokens.

Saved Outputs: None. This module writes no files.

How the code works: A group is every trained model with the same vocabulary size, number of layers, and number of
heads. load_group finds the checkpoints of a group in a folder, stacks them into one StackedModels with gradients
turned off, and returns it with the tag of each model (process name, layers, heads, and seed). context_tokens returns,
for each model, the 10 evaluation contexts of its process from results/contexts, each as a BOS token followed by 127
symbols.
"""
import glob
import os
import re

import numpy as np
import torch

from src import processes as PR
from src.stacked import StackedModels


# All checkpoints of the group, stacked. d_vocab is V + 1, the V process symbols and the BOS token.
def load_group(d_vocab, n_layers, n_heads, ckpt_dir, device):
    V = d_vocab - 1
    names = [s["name"] for s in PR.all_sweep_processes() if s["V"] == V]
    tags, states = [], []
    for name in names:
        for f in sorted(glob.glob(os.path.join(ckpt_dir, f"{name}_{n_layers}l_h{n_heads}_s*.pt"))):
            tags.append(re.sub(r"\.pt$", "", os.path.basename(f)))
            states.append(torch.load(f, map_location="cpu", weights_only=False))
    assert states, "no checkpoints found for this group"
    stack = StackedModels(n_layers, V, n_heads, states, device=device)
    for k in stack.params:
        stack.params[k].requires_grad_(False)
    return stack, tags, names


# The 10 evaluation contexts of each model's process, with the BOS token (index V) in front.
def context_tokens(tags, V, device):
    ctx_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "contexts")
    toks = []
    for t in tags:
        name = re.sub(r"_\d+l_h\d+_s\d+$", "", t)
        seqs = np.load(os.path.join(ctx_dir, f"{name}.npy"))
        toks.append(np.concatenate([np.full((len(seqs), 1), V), seqs], 1))
    return torch.tensor(np.stack(toks), device=device)
