"""The next-token loss of every model on the evaluation contexts.

Paper section: Section 3 (Causal necessity and sufficiency).

Claim: In the paper's words, the KL of 0.041 nats at mlp_in is "about 5% of the models' average per-token loss of 0.78
nats".

Experiment: For every model of a group, the cross-entropy between the model's output and the next token is computed at
every token position of the 10 evaluation contexts and averaged.

Saved Outputs: loss_v<d_vocab>_<L>l_h<H>.npz in the output folder, with tags, loss (the mean loss per token of each
model, in nats), and n_pos (the number of predicted tokens per model).

How the code works: The script runs one forward pass, takes the log-probability that each position assigns to the
token that follows it, and averages over the 10 contexts and the 127 predicted positions. Usage: python
experiments/necessity/run_loss.py <d_vocab> <n_layers> <n_heads> <ckpt_dir> <out_dir>, where d_vocab is V + 1.
"""
import os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from src.evaluation import context_tokens, load_group


def main(d_vocab, n_layers, n_heads, ckpt_dir, out_dir):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    V = d_vocab - 1
    stack, tags, _ = load_group(d_vocab, n_layers, n_heads, ckpt_dir, dev)
    tokens = context_tokens(tags, V, dev)
    with torch.no_grad():
        # Log-probabilities at every position, and the negative log-probability of the token that follows.
        logits = stack.forward(tokens).float()
        lp = logits.log_softmax(-1)
        tgt = tokens[:, :, 1:]
        nll = -lp[:, :, :-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        loss = nll.mean((1, 2)).cpu().numpy()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"loss_v{d_vocab}_{n_layers}l_h{n_heads}.npz")
    np.savez(out, tags=np.array(tags), loss=loss, n_pos=np.full(len(tags), nll.shape[1] * nll.shape[2]))
    print(f"  loss v{d_vocab} {n_layers}l h{n_heads}: mean {loss.mean():.4f} median {np.median(loss):.4f} n={len(tags)}", flush=True)

if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
