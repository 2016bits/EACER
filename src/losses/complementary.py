import torch
import torch.nn.functional as F


def complementary_alignment_loss(
    claim_comp_repr: torch.Tensor,        # (B, D)
    weighted_visual: torch.Tensor,        # (M, D) — already unit-norm
    rep_index: torch.Tensor,              # (M,)
    num_negatives_per_sample: torch.Tensor,   # (B,)
    temperature: float = 0.07,
) -> torch.Tensor:
    """Aligns q_comp with the weighted visual representation of the positive evidence.

    Positives = visual of own positive evidence; negatives = every other evidence
    in the batch (their visual representations).
    """
    B = claim_comp_repr.size(0)
    device = claim_comp_repr.device

    sim = claim_comp_repr @ weighted_visual.t() / temperature             # (B, M)

    offsets = torch.zeros(B, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(num_negatives_per_sample[:-1] + 1, dim=0)
    targets = offsets

    return F.cross_entropy(sim, targets)
