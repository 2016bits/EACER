import torch
import torch.nn.functional as F


def contrastive_retrieval_loss(
    claim_repr: torch.Tensor,            # (B, D), unit-norm
    evidence_repr: torch.Tensor,         # (M, D), unit-norm; M = B + sum(num_neg_per_sample)
    rep_index: torch.Tensor,             # (M,) int — for each evidence, the claim index it belongs to
    num_negatives_per_sample: torch.Tensor,   # (B,)
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE over (in-batch positives + per-sample hard negatives).

    For sample i the positive evidence is the first occurrence of rep_index == i.
    All other evidences (every position in evidence_repr except this positive) act
    as negatives for sample i; this naturally includes hard negatives for sample i
    and in-batch evidences from other samples (positives + hard negs of others).
    """
    B = claim_repr.size(0)
    M = evidence_repr.size(0)
    device = claim_repr.device

    sim = claim_repr @ evidence_repr.t() / temperature                    # (B, M)

    # Locate positive index for each sample: first index where rep_index == i.
    offsets = torch.zeros(B, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(num_negatives_per_sample[:-1] + 1, dim=0)
    targets = offsets                                                     # (B,)

    return F.cross_entropy(sim, targets)
