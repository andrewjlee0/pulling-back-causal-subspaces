"""The transformer architecture of the paper, and the loading of a single trained model.

Paper section: Appendix D (Architecture).

Claim: In the paper's words, "All models are pre-norm transformers with residual stream dimension d_model = 128, MLP
hidden dimension d_mlp = 512, GELU as the MLP non-linear activation function, RMSNorm as the normalization function,
rotary position embeddings, and a context length of 128 tokens. Each attention head has dimension d_model/H, where H
is the number of heads."

Experiment: This module runs no experiment. The training script takes each model's initial weights from make_model.

Saved Outputs: None. This module writes no files.

How the code works: make_model builds a TransformerLens HookedTransformer with these settings. The vocabulary has V +
1 entries, which are the V symbols of the process and one BOS token with index V, and every sequence is the BOS token
followed by 127 symbols. load_checkpoint loads the weights of one trained model into this architecture.
"""
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

# The context length, and the number of symbols that follow the BOS token.
N_CTX = 128
L = N_CTX - 1


def make_model(n_layers, V, n_heads=4, seed=0, device="cpu"):
    cfg = HookedTransformerConfig(
        d_model=128, n_heads=n_heads, d_head=128 // n_heads, d_mlp=512,
        n_layers=n_layers, n_ctx=N_CTX, d_vocab=V + 1,
        act_fn="gelu", normalization_type="RMS",
        positional_embedding_type="rotary",
        gated_mlp=False, device=device, seed=seed,
    )
    return HookedTransformer(cfg)


def load_checkpoint(path, n_layers, V, n_heads=4, device="cpu"):
    m = make_model(n_layers, V, n_heads=n_heads, device=device)
    sd = torch.load(path, map_location=device, weights_only=False)
    sd = {k: v for k, v in sd.items() if not k.endswith("attn.mask") and "IGNORE" not in k}
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m
