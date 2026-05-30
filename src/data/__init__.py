from .dataset import (
    RetrievalDataset,
    MochegRetrievalDataset,   # alias kept for backwards compatibility
    EvidenceCorpus,
    ClaimQueries,
)
from .collator import RetrievalCollator, EncodeCollator
