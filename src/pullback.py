"""The derived causal subspaces of resid_final and resid_post, the projection onto them, and the KL divergence between
two outputs.

Paper section: Sections 2.2 and 2.3 with Equations (2) and (3), and Appendix B with Lemma 3 and Proposition 1.

Claim: In the paper's words, at resid_final "the causal subspace is the span of the centered unembedding columns,
span{u_i − ū}", and at resid_post we "designate span{g_i} as the causal linear subspace of resid_post", where "g_i = γ
⊙ (u_i − ū) is centered unembedding column i rescaled entry-wise by γ".

Experiment: This module runs no experiment. The experiment scripts take their subspaces and their KL divergence here.

Saved Outputs: None. This module writes no files.

How the code works: resid_final_basis returns, for every model, an orthonormal basis of the span of the centered
unembedding columns, and window_basis returns an orthonormal basis of the span of the g_i. With full=True the columns
of all V + 1 vocabulary entries are used, which include the column of the BOS token, and the subspace has V
dimensions. With full=False only the columns of the V process symbols are used, and the subspace has V − 1 dimensions.
project returns the component of an activation inside a subspace. kl returns, per model, the KL divergence from the
output of the first set of logits to the output of the second, over the V + 1 vocabulary entries and averaged over
contexts and token positions.
"""
import torch


# The causal subspace of resid_post: the span of the g_i of Equation (3).
def window_basis(params, V=None, full=True):
    WU = params["unembed.W_U"]
    gam = params["ln_final.w"]
    U = WU if full else WU[:, :, :V]
    Uc = U - U.mean(-1, keepdim=True)
    G = gam[:, :, None] * Uc
    Q, _ = torch.linalg.qr(G[:, :, :-1].cpu())
    return Q.to(WU.device)


# The causal subspace of resid_final: the span of the centered unembedding columns of Equation (2).
def resid_final_basis(params, V=None, full=True):
    WU = params["unembed.W_U"]
    U = WU if full else WU[:, :, :V]
    Uc = U - U.mean(-1, keepdim=True)
    Q, _ = torch.linalg.qr(Uc[:, :, :-1].cpu())
    return Q.to(WU.device)


def project(x, Q):
    c = torch.einsum("mbpd,mdk->mbpk", x, Q)
    return torch.einsum("mbpk,mdk->mbpd", c, Q)


# KL(p || q) between the outputs of two sets of logits, per model.
def kl(logits_p, logits_q):
    lp = logits_p.log_softmax(-1)
    lq = logits_q.log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1).mean((-1, -2))
