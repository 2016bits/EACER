import torch
import torch.nn.functional as F


def complementary_alignment_loss(
    claim_comp_repr: torch.Tensor,        # (B, D)
    weighted_visual: torch.Tensor,        # (M, D) — already unit-norm
    rep_index: torch.Tensor,              # (M,)
    num_negatives_per_sample: torch.Tensor,   # (B,)
    temperature: float = 0.07,
    claim_int_id: torch.Tensor = None,    # (B,) int — same id means same underlying claim
) -> torch.Tensor:
    """Aligns q_comp with the weighted visual representation of the positive evidence.

    Same self-pollution fix as ``contrastive_retrieval_loss``: when
    ``claim_int_id`` is supplied, mask out visual reps of other samples that
    share sample i's underlying claim.
    """
    B = claim_comp_repr.size(0)
    M = weighted_visual.size(0)
    device = claim_comp_repr.device

    sim = claim_comp_repr @ weighted_visual.t() / temperature             # (B, M)

    offsets = torch.zeros(B, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(num_negatives_per_sample[:-1] + 1, dim=0)
    targets = offsets

    if claim_int_id is not None:
        cid_per_evi = claim_int_id[rep_index]                             # (M,)
        same_claim = claim_int_id.unsqueeze(1) == cid_per_evi.unsqueeze(0)
        own_sample = (
            torch.arange(B, device=device).unsqueeze(1) == rep_index.unsqueeze(0)
        )
        # See contrastive.py for the reasoning: only drop other-sample evidences
        # that share i's underlying claim.
        mask = same_claim & ~own_sample
        sim = sim.masked_fill(mask, float("-inf"))

    return F.cross_entropy(sim, targets)
