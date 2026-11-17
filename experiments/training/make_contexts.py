"""The evaluation contexts: 10 fixed sequences per process, on which every experiment is run.

Paper section: Section 3 (Experiments).

Claim: The paper measures every intervention "At each hook and every token position". The token positions are those of
the contexts written by this script.

Experiment: For each of the 22 processes, 10 sequences of 127 symbols are sampled from the process with a fixed seed.

Saved Outputs: results/contexts/<process>.npy for each process, an integer array of shape [10, 127].

How the code works: For each process the random generator is seeded with 424242. HMMs are sampled by rolling the
hidden state and GHMMs by exact filtering. Usage: python experiments/training/make_contexts.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from src import processes as P
from src.models import L

# The seed and the number of contexts per process.
CTX_SEED = 424242
N_CTX_SEQS = 10

outdir = os.path.join(ROOT, "results", "contexts")
os.makedirs(outdir, exist_ok=True)
# Sample and save the contexts of every process.
for spec in P.all_sweep_processes():
    rng = np.random.default_rng(CTX_SEED)
    M, pi, eta = spec["M"], spec["pi"], spec["eta"]
    if np.all(M >= -1e-12):
        seqs = P.sample_fast(M, N_CTX_SEQS, L, rng, pi=pi)
    else:
        seqs = P.sample(M, N_CTX_SEQS, L, rng, eta=eta, pi=pi)
    np.save(os.path.join(outdir, f"{spec['name']}.npy"), seqs)
    print(spec["name"], seqs.shape, "first tokens:", seqs[0, :8].tolist())
