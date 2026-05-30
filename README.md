# Entropy-Aware Complementary Evidence Retrieval for Multimodal Fact Checking

> Working name: **ECER** — **Entropy-aware Complementary Evidence Retrieval**  
> Target task: **Evidence retrieval for multimodal fact checking**

## 1. Motivation

In multimodal fact checking, the goal of evidence retrieval is not only to retrieve evidence that is semantically similar to a claim, but also to retrieve evidence that contains sufficient and reliable complementary information for verifying the claim.

Recent multimodal retrieval methods such as CIEA emphasize **cross-modal complementary information**. Their core assumption is that visual regions that are dissimilar to the paired text may contain information not covered by the text. However, this assumption is not always reliable:

```text
large cross-modal difference != useful complementary evidence
```

A visual patch or textual span may have a large difference from the other modality because it contains useful complementary information, but it may also be caused by:

- background noise;
- irrelevant objects;
- blurry or low-quality visual regions;
- OCR errors;
- noisy captions;
- modality-specific artifacts;
- poor cross-modal alignment.

Therefore, directly emphasizing high-difference visual or textual signals may introduce noise into the retrieval process.

This project explores an **entropy-aware complementary evidence retrieval framework**. The key idea is to identify evidence that is simultaneously:

1. relevant to the claim;
2. complementary across modalities;
3. reliable or informative rather than noisy.

We formulate this as:

```text
good complementary evidence = claim relevance + cross-modal complementarity + entropy-based reliability
```

## 2. Problem Definition

Given a claim:

```text
c
```

and a candidate multimodal evidence item:

```text
e = {e_t, e_v}
```

where:

- `e_t` is textual evidence, such as title, caption, OCR text, metadata, or article text;
- `e_v` is visual evidence, such as an image or video frame.

The goal is to learn a retrieval model that ranks true evidence higher than irrelevant evidence:

```text
score(c, e+) > score(c, e-)
```

Different from standard dense retrieval, the model should retrieve evidence that contains not only claim-aligned information, but also useful complementary information that helps downstream fact verification.

## 3. Core Idea

The proposed method estimates the importance of each visual patch or textual unit using three factors:

1. **Claim relevance**  
   Whether the unit is related to the claim.

2. **Cross-modal complementarity**  
   Whether the unit contains information not already covered by the other modality.

3. **Entropy-based reliability**  
   Whether the unit is semantically reliable rather than noisy or uncertain.

For a visual patch `v_j`, the final weight is:

```math
W_j = S_j \cdot D_j \cdot R_j
```

where:

```math
S_j = \max_k \cos(v_j, c_k)
```

is the claim relevance score,

```math
D_j = 1 - \max_l \cos(v_j, t_l)
```

is the cross-modal complementarity score, and

```math
R_j = f(H_j)
```

is the entropy-based reliability score.

## 4. Model Overview

The overall framework contains the following modules:

```text
Claim Encoder
      |
      v
Claim Representation

Evidence Text Encoder        Evidence Image Encoder
      |                              |
      v                              v
Text Token Representations     Visual Patch Representations
      |                              |
      +--------------+---------------+
                     |
                     v
      Entropy-Aware Complementary Weighting
                     |
                     v
        Multimodal Evidence Representation
                     |
                     v
          Claim-Evidence Retrieval Score
```

## 5. Module Design

### 5.1 Claim Encoder

The claim encoder maps the input claim into contextual token representations:

```math
C = \{c_1, c_2, \ldots, c_m\}
```

where `m` is the number of claim tokens.

Possible encoder choices:

- BERT / RoBERTa;
- T5 encoder;
- DeBERTa;
- CLIP text encoder;
- a domain-specific fact-checking encoder.

### 5.2 Evidence Text Encoder

The textual part of the evidence is encoded as:

```math
T = \{t_1, t_2, \ldots, t_n\}
```

where `n` is the number of evidence text tokens.

The evidence text may include:

- image caption;
- OCR text;
- article title;
- webpage text;
- metadata;
- textual evidence sentence.

### 5.3 Evidence Image Encoder

The visual evidence is encoded into patch-level representations:

```math
V = \{v_1, v_2, \ldots, v_p\}
```

where `p` is the number of visual patches.

Possible visual encoders:

- CLIP ViT;
- BLIP / BLIP-2 vision encoder;
- EVA-CLIP;
- SigLIP;
- DINOv2;
- any pretrained vision-language encoder.

## 6. Claim Relevance Score

For each visual patch `v_j`, compute its maximum similarity with all claim token representations:

```math
S_j = \max_k \cos(v_j, c_k)
```

This term prevents the model from emphasizing clear but claim-irrelevant information.

For example, a background object may be visually clear and semantically certain, but if it is unrelated to the claim, its relevance score should be low.

## 7. Cross-Modal Complementarity Score

For each visual patch `v_j`, compute its dissimilarity from the evidence text tokens:

```math
D_j = 1 - \max_l \cos(v_j, t_l)
```

This term estimates whether the visual patch contains information not already expressed by the textual evidence.

A high value of `D_j` means that the patch is less similar to the evidence text and may contain complementary visual information.

However, high complementarity alone is not sufficient because noisy regions can also be highly dissimilar from text. This motivates entropy-based reliability modeling.

## 8. Entropy-Based Reliability Score

### 8.1 Semantic Uncertainty Entropy

If we can obtain a semantic prediction distribution for each visual patch:

```math
p_j = \{p_{j,1}, p_{j,2}, \ldots, p_{j,K}\}
```

then the entropy of patch `v_j` is:

```math
H_j = - \sum_{r=1}^{K} p_{j,r} \log p_{j,r}
```

A high entropy value indicates high uncertainty. In this setting, high entropy is more likely to correspond to noisy, ambiguous, or unreliable visual information.

The reliability score can be defined as:

```math
R_j = 1 - \frac{H_j}{\log K}
```

where `K` is the number of semantic classes or candidate concepts.

This gives:

```math
R_j \in [0, 1]
```

A low-entropy patch receives a high reliability score, while a high-entropy patch receives a low reliability score.

### 8.2 Medium-Entropy Preference

If entropy is computed from feature complexity, local visual distribution, or attention distribution, both extremely low entropy and extremely high entropy may indicate low-quality information.

In that case, a medium-entropy preference function can be used:

```math
R_j = \exp\left(-\frac{(H_j - \mu)^2}{2\sigma^2}\right)
```

where:

- `mu` is the preferred entropy level;
- `sigma` controls the tolerance range.

This design suppresses both trivial low-information regions and chaotic high-uncertainty regions.

## 9. Final Entropy-Aware Complementary Weight

The final visual patch weight is:

```math
W_j = S_j \cdot D_j \cdot R_j
```

where:

- `S_j` measures claim relevance;
- `D_j` measures visual-text complementarity;
- `R_j` measures entropy-based reliability.

The weighted visual representation is:

```math
\tilde{v} = \sum_{j=1}^{p} \alpha_j v_j
```

where:

```math
\alpha_j = \frac{\exp(W_j)}{\sum_{q=1}^{p} \exp(W_q)}
```

The resulting representation `tilde_v` is expected to focus on visual information that is relevant, complementary, and reliable.

## 10. Multimodal Evidence Representation

After obtaining the weighted visual representation, we construct the final evidence representation by fusing textual and visual information:

```math
h_e = Fusion(h_t, \tilde{v})
```

Possible fusion strategies:

### 10.1 Concatenation

```math
h_e = \text{MLP}([h_t ; \tilde{v}])
```

### 10.2 Cross-Attention

Use the claim as query and multimodal evidence tokens as key/value:

```math
h_e = \text{CrossAttention}(C, [T; V])
```

### 10.3 Transformer Fusion

Concatenate text tokens and weighted visual tokens, then feed them into a Transformer:

```math
h_e = \text{Transformer}([T; \alpha_1 v_1; \ldots; \alpha_p v_p])
```

## 11. Retrieval Score

The claim-evidence retrieval score can be computed using cosine similarity:

```math
score(c, e) = \cos(h_c, h_e)
```

or a learned scoring function:

```math
score(c, e) = \text{MLP}([h_c; h_e; h_c \odot h_e; |h_c - h_e|])
```

For efficient large-scale retrieval, a dual-encoder architecture is recommended:

```text
claim encoder -> claim vector
candidate evidence encoder -> evidence vector
similarity search with FAISS / ScaNN / Elasticsearch dense vector index
```

For higher accuracy but lower efficiency, a cross-encoder reranker can be added after the first-stage retriever.

## 12. Training Objective

### 12.1 Contrastive Retrieval Loss

Given a positive evidence item `e+` and a set of negative evidence items `{e-}`, use an InfoNCE-style contrastive loss:

```math
\mathcal{L}_{ret}
=
-\log
\frac{
\exp(score(c,e^+)/\tau)
}{
\exp(score(c,e^+)/\tau)
+
\sum_{e^-}
\exp(score(c,e^-)/\tau)
}
```

where `tau` is a temperature hyperparameter.

### 12.2 Complementary Alignment Loss

To explicitly encourage the model to use complementary evidence, construct a complementary claim query `c_comp` by masking claim spans already covered by the evidence text.

Example:

```text
Claim:
A photo shows a protester holding a sign saying "Save the Arctic" in London.

Evidence text:
A protester is holding a sign during a demonstration.

Complementary claim:
A photo shows a [MASK] in [MASK].
```

The model is then encouraged to align the complementary claim with the visual representation:

```math
\mathcal{L}_{comp}
=
-\log
\frac{
\exp(\cos(h_{c_{comp}}, \tilde{v}^+)/\tau)
}{
\exp(\cos(h_{c_{comp}}, \tilde{v}^+)/\tau)
+
\sum_{e^-}
\exp(\cos(h_{c_{comp}}, \tilde{v}^-)/\tau)
}
```

### 12.3 Final Loss

```math
\mathcal{L}
=
\mathcal{L}_{ret}
+
\lambda \mathcal{L}_{comp}
```

where `lambda` controls the strength of the complementary alignment loss.

## 13. Data Format

A suggested JSONL format:

```json
{
  "claim_id": "claim_0001",
  "claim": "A photo shows the Eiffel Tower illuminated in blue.",
  "label": "SUPPORTS",
  "positive_evidence": [
    {
      "evidence_id": "evi_0001",
      "image_path": "images/evi_0001.jpg",
      "text": "The Eiffel Tower is lit up at night.",
      "caption": "The Eiffel Tower at night.",
      "ocr": "",
      "source": "news_article"
    }
  ],
  "negative_evidence": [
    {
      "evidence_id": "evi_0321",
      "image_path": "images/evi_0321.jpg",
      "text": "A city landmark is shown at night.",
      "caption": "A bridge illuminated at night.",
      "ocr": "",
      "source": "news_article"
    }
  ]
}
```

## 14. Candidate Datasets

Possible multimodal fact-checking or evidence retrieval datasets:

- **MOCHEG**: multimodal claim-evidence fact checking;
- **Fakeddit**: multimodal misinformation detection;
- **NewsCLIPpings**: image-text mismatch detection;
- **VERITE**: vision-language evidence and claim verification;
- **Factify**: multimodal fact verification;
- **MultiFC / FEVER-style extensions** if adapted with image evidence;
- custom datasets with claim, text evidence, image evidence, and verification labels.

For datasets without explicit evidence retrieval annotations, pseudo-positive evidence can be constructed from gold verification evidence or aligned article-image pairs.

## 14.1 MR2 quick start

The retrieval pipeline supports MR2 out of the box. After downloading and
unzipping the dataset (`MR2.zip`) so that
`<MR2_ROOT>/queries_dataset_merge/dataset_items_{train,val,test}.json` exists
(e.g. under `/mnt/data/yangjun/data/mr2`):

```bash
# 1. Convert MR2 -> JSONL (one-shot)
bash scripts/preprocess_mr2.sh

# 2. Train (full ECER on MR2)
bash scripts/train_mr2.sh configs/mr2_ecer.yaml

# 3. Evaluate on test
bash scripts/evaluate_mr2.sh configs/mr2_ecer.yaml outputs/mr2_ecer_A/best.pt test
```

See `data/README.md` for the on-disk layout and the MR2 → JSONL schema.

## 15. Implementation Plan

### Step 1: Build a Baseline Retriever

Implement a standard multimodal dense retriever:

- claim encoder;
- evidence text encoder;
- image encoder;
- multimodal fusion module;
- contrastive retrieval loss.

Recommended baseline variants:

- text-only retriever;
- image-only retriever;
- text + image concatenation retriever;
- CIEA-style complementary retriever.

### Step 2: Add Cross-Modal Complementarity

Compute:

```math
D_j = 1 - \max_l \cos(v_j, t_l)
```

Use `D_j` to reweight visual patches.

### Step 3: Add Claim Relevance

Compute:

```math
S_j = \max_k \cos(v_j, c_k)
```

Use this term to filter out claim-irrelevant complementary signals.

### Step 4: Add Entropy-Based Reliability

Compute entropy using one of the following options:

1. semantic prediction distribution from a VLM or object classifier;
2. patch-to-token attention distribution;
3. visual feature distribution;
4. retrieval score distribution over candidate concepts.

Then compute reliability:

```math
R_j = 1 - H_j / log(K)
```

or:

```math
R_j = exp(-(H_j - mu)^2 / (2 * sigma^2))
```

### Step 5: Combine the Three Terms

```math
W_j = S_j \cdot D_j \cdot R_j
```

Use `W_j` to compute the weighted visual representation.

### Step 6: Train with Retrieval and Complementary Alignment Loss

Use:

```math
L = L_ret + lambda * L_comp
```

### Step 7: Evaluate Retrieval and Verification

Evaluate both evidence retrieval and downstream fact verification.

## 16. Evaluation Metrics

### 16.1 Evidence Retrieval Metrics

Recommended metrics:

- Recall@K;
- Precision@K;
- MRR@K;
- NDCG@K;
- Evidence F1;
- Gold evidence coverage.

For fact checking, `Recall@K` is especially important because the verifier can only make correct predictions if the necessary evidence is retrieved.

### 16.2 Downstream Verification Metrics

After retrieval, feed top-K evidence into a verifier and evaluate:

- Accuracy;
- Macro-F1;
- Label-wise F1;
- FEVER score or dataset-specific evidence-aware score;
- joint retrieval-verification score.

## 17. Ablation Studies

Recommended ablation settings:

| Model | Claim relevance | Complementarity | Entropy reliability | Complementary loss |
|---|---:|---:|---:|---:|
| Text-only | No | No | No | No |
| Multimodal baseline | No | No | No | No |
| Complementarity only | No | Yes | No | No |
| Claim + complementarity | Yes | Yes | No | No |
| Entropy + complementarity | No | Yes | Yes | No |
| Full ECER | Yes | Yes | Yes | Yes |

Suggested additional ablations:

- different entropy definitions;
- high-entropy suppression vs medium-entropy preference;
- with vs without OCR;
- with vs without caption;
- dual-encoder vs reranker;
- different visual encoders;
- different values of `lambda`;
- different top-K retrieval settings.

## 18. Expected Advantages

The proposed method is expected to:

- reduce the impact of noisy visual or textual differences;
- distinguish useful complementary information from irrelevant modality-specific noise;
- improve evidence recall for multimodal fact checking;
- retrieve more complete evidence for downstream verification;
- provide more interpretable patch-level evidence weights.

## 19. Potential Risks

### 19.1 Entropy Definition Ambiguity

Entropy can have different meanings depending on how it is computed. The method must clearly define whether entropy measures:

- semantic uncertainty;
- visual complexity;
- feature distribution diversity;
- attention dispersion;
- concept prediction uncertainty.

### 19.2 Low Entropy Does Not Always Mean Useful Information

A patch can be low-entropy but irrelevant to the claim. This is why claim relevance is necessary.

### 19.3 High Complementarity May Still Be Noise

A patch can be highly different from text because it is noisy or poorly aligned. This is why entropy-based reliability is necessary.

### 19.4 Retrieval and Verification May Prefer Different Evidence

Evidence that improves retrieval metrics may not always improve downstream verification. Both stages should be evaluated.

## 20. Suggested Repository Structure

```text
ECER/
├── README.md
├── configs/
│   ├── baseline.yaml
│   ├── ecer.yaml
│   └── ablation.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── collator.py
│   ├── models/
│   │   ├── encoders.py
│   │   ├── fusion.py
│   │   ├── entropy.py
│   │   ├── weighting.py
│   │   └── retriever.py
│   ├── losses/
│   │   ├── contrastive.py
│   │   └── complementary.py
│   ├── train.py
│   ├── evaluate_retrieval.py
│   └── evaluate_verification.py
├── scripts/
│   ├── preprocess_data.sh
│   ├── train_baseline.sh
│   ├── train_ecer.sh
│   └── evaluate.sh
└── outputs/
    ├── checkpoints/
    ├── logs/
    └── predictions/
```

## 21. Pseudocode

```python
# claim tokens: C = [c_1, ..., c_m]
# evidence text tokens: T = [t_1, ..., t_n]
# visual patches: V = [v_1, ..., v_p]

for each visual patch v_j in V:
    # claim relevance
    S_j = max_cosine_similarity(v_j, C)

    # cross-modal complementarity
    D_j = 1.0 - max_cosine_similarity(v_j, T)

    # entropy-based reliability
    H_j = compute_entropy(v_j)
    R_j = entropy_to_reliability(H_j)

    # final weight
    W_j = S_j * D_j * R_j

alpha = softmax(W)
weighted_visual = sum(alpha_j * v_j for each patch j)

evidence_repr = fuse(text_repr, weighted_visual)
claim_repr = pool(C)

score = cosine_similarity(claim_repr, evidence_repr)
loss = contrastive_loss(score)
```

## 22. Main Research Claim

This project is based on the following hypothesis:

> In multimodal fact-checking evidence retrieval, useful evidence should contain information that is not only semantically relevant to the claim, but also complementary across modalities and reliable under entropy-based uncertainty estimation. Entropy-aware complementary weighting can reduce noisy modality-specific differences and improve the retrieval of sufficient evidence for claim verification.

## 23. Possible Method Names

- **ECER**: Entropy-aware Complementary Evidence Retrieval
- **EACR**: Entropy-Aware Complementary Retrieval
- **EACER**: Entropy-Aware Complementary Evidence Retrieval
- **ReCoER**: Reliable Complementary Evidence Retrieval
- **CARE**: Claim-Aware Reliable Evidence Retrieval

Recommended name:

```text
ECER: Entropy-aware Complementary Evidence Retrieval
```

It is short, clear, and directly reflects the main idea.

## 24. TODO

- [ ] Select dataset and define evidence retrieval annotations.
- [ ] Implement text-only retrieval baseline.
- [ ] Implement multimodal retrieval baseline.
- [ ] Implement CIEA-style complementarity weighting.
- [ ] Implement claim relevance weighting.
- [ ] Implement entropy estimation module.
- [ ] Implement entropy-aware complementary weighting.
- [ ] Add complementary claim construction.
- [ ] Add complementary alignment loss.
- [ ] Evaluate retrieval performance.
- [ ] Evaluate downstream verification performance.
- [ ] Run ablation experiments.
- [ ] Visualize patch-level weights.
- [ ] Analyze retrieved evidence quality.

## 25. Notes for Future Extension

Possible future extensions include:

- applying entropy-aware complementarity to textual evidence tokens as well as visual patches;
- using OCR tokens as a separate modality;
- extending from image evidence to video evidence;
- adding label-aware retrieval for SUPPORTS / REFUTES / NEI;
- using retrieved evidence to reduce hallucination in multimodal fact-checking RAG;
- jointly training retriever and verifier.
