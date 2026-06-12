import torch
import torch.nn.functional as F


def contrastive_retrieval_loss(
    claim_repr: torch.Tensor,            # (B, D), unit-norm
    evidence_repr: torch.Tensor,         # (M, D), unit-norm; M = B + sum(num_neg_per_sample)
    rep_index: torch.Tensor,             # (M,) int — for each evidence, the claim index it belongs to
    num_negatives_per_sample: torch.Tensor,   # (B,)
    temperature: float = 0.07,
    claim_int_id: torch.Tensor = None,   # (B,) int — same id means same underlying claim
) -> torch.Tensor:
    """InfoNCE over (in-batch positives + per-sample hard negatives).

    For sample i the positive evidence is the first occurrence of rep_index == i.
    All other evidences (every position in evidence_repr except this positive) act
    as negatives for sample i; this naturally includes hard negatives for sample i
    and in-batch evidences from other samples (positives + hard negs of others).

    When ``claim_int_id`` is provided we mask out evidences whose underlying
    claim equals sample i's claim (but aren't sample i's own positive): MR2 has
    ~8 positives per claim and the dataset splits them into separate samples,
    so other-positives-of-same-claim accidentally became negatives. Masking
    them removes that self-pollution.
    """
    B = claim_repr.size(0)
    M = evidence_repr.size(0)
    device = claim_repr.device

    sim = claim_repr @ evidence_repr.t() / temperature                    # (B, M)

    # Locate positive index for each sample: first index where rep_index == i.
    offsets = torch.zeros(B, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(num_negatives_per_sample[:-1] + 1, dim=0)
    targets = offsets                                                     # (B,)

    if claim_int_id is not None:
        # For each evidence j, the sample it belongs to is rep_index[j], whose
        # underlying claim is claim_int_id[rep_index[j]].
        cid_per_evi = claim_int_id[rep_index]                             # (M,)
        same_claim = claim_int_id.unsqueeze(1) == cid_per_evi.unsqueeze(0)  # (B, M)
        # An evidence belongs to sample i iff rep_index[j] == i. This covers
        # both sample i's own positive (at targets[i]) and sample i's own hard
        # negatives — both must stay (positive is the target, hard negs are
        # the very signal we want to push away).
        own_sample = (
            torch.arange(B, device=device).unsqueeze(1) == rep_index.unsqueeze(0)
        )                                                                  # (B, M)
        # Mask out: same underlying claim AND not from sample i itself.
        # This drops positives/negatives of OTHER samples that happen to share
        # i's claim, which is the self-pollution we want to fix.
        mask = same_claim & ~own_sample
        sim = sim.masked_fill(mask, float("-inf"))

    return F.cross_entropy(sim, targets)
