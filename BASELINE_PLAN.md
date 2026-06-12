# EACER Baseline 实验计划

> 目的：为 EACER 论文准备一份完整、可复现、按性价比排序的 baseline 矩阵。
> 主战场：MR2（已就绪），MOCHEG（preprocess_mocheg.py 待调通后加入）。
> 关联文档：`README.md`（方法定义）、`REPORT.md`（当前实验进展）、`IMPROVEMENT_PLAN.md`（ECER 自身改进路线）。

---

## 0. 主表骨架（论文最终要交付的表）

```
                                         | R@1 | R@5 | R@10 | R@100 | MRR | mAP
─────────────────────────────────────────┼─────┼─────┼──────┼───────┼─────┼─────
Sparse                                   │  → outputs/sparse_baselines_test.md
  BM25 (XLM-R tokenizer)                 │  ✅ all 23.84 / zh 14.12 / en 32.73  (R@10)
  TF-IDF (XLM-R tokenizer)               │  ✅ all 24.31 / zh 14.25 / en 33.42  (R@10)
  Jieba+BM25                             │  ✅ all 24.46 / zh 13.05 / en 34.91  (R@10)
                                         │
Text-only dense                          │
  BGE-M3 (zero-shot)                     │  ⬜ 待补 ★
  BGE-M3 (fine-tuned on MR2)             │  ⬜ 待补 ★
  E5-base-multilingual (zero-shot)       │  ⬜ 可选
  Contriever-msmarco                     │  ⬜ 可选
                                         │
Multimodal zero-shot                     │
  CLIP ViT-B/32 dual                     │  ⬜ 待补 ★
  SigLIP-base                            │  ⬜ 可选
  BLIP-2 ITC                             │  ⬜ 可选
                                         │
Multimodal fine-tuned (同任务对手)        │
  MR2 official baseline (CLIP+BERT)      │  ⬜ 必做 ★
  MOCHEG retriever                       │  ⬜ MOCHEG 调通后做
  CIEA 完整版（S+D+complementary）       │  ⬜ 必做 ★（目前只 D-only）
  PreCoFactv2 / Pre-CoFact               │  ⬜ 可选
  MEVER                                  │  ⬜ 可选（EACL'26）
                                         │
LLM/VLM zero-shot                        │
  Qwen2.5-VL embedding                   │  ⬜ 可选
  GPT-4o text embedding                  │  ⬜ 可选
                                         │
Late-interaction                         │
  ColBERT-v2 (multilingual)              │  ⬜ 可选
                                         │
Hybrid (BM25 + dense)                    │
  BM25 + Dense RRF                       │  ✅ 已有
  BM25 + ECER 凸融合 (α=0.5, hardneg)    │  ✅ 已有 — 当前 SOTA
                                         │
Ours                                     │
  ECER dense-only                        │  ✅
  ECER + BM25 rerank                     │  ✅
  Full ECER (final, after Step 2/3/4)    │  ⬜ 等 IMPROVEMENT_PLAN 完成
```

★ = 论文里**必须**有的对手，否则审稿会要求补。

---

## 1. 必做 baseline 详细规格

### 1.1 BGE-M3（多语言 dense text 强基线）

- **为什么必做**：现有 SOTA 多语言 dense retriever，证明 EACER 加视觉的收益不是来自 text encoder 升级。
- **模型**：`BAAI/bge-m3`（HuggingFace）
- **两种用法**：
  - zero-shot：直接 encode claim 和 (title+snippet+caption+ocr) 算 cos
  - fine-tuned on MR2：用 EACER 同 batch / 同 hard-neg 配置 fine-tune

**实施**：

```bash
# 新建 scripts/baseline_bge_m3.py
# - 复用 src.data.RetrievalDataset 读 jsonl
# - 用 FlagEmbedding 库或纯 transformers AutoModel.from_pretrained
# - encode 全 corpus → faiss → 算 R@K/MRR/mAP（沿用 evaluate_retrieval.py 的指标）
```

**预算**：半天写代码 + 1 张卡跑 ~2 小时（fine-tune ~6 小时）。
**预期**：zero-shot R@10 约 4–8%（dense 在 MR2 仍弱于 BM25），fine-tuned 约 12–18%。

---

### 1.2 CLIP zero-shot dual-encoder

- **为什么必做**：最便宜的多模态 baseline；如果 ECER 跑不过 CLIP zero-shot，说明 entropy + complementary 没贡献。
- **模型**：`openai/clip-vit-base-patch32`（与 EACER 视觉编码器一致，公平比较）
- **打分方式**：
  - claim 用 CLIP text encoder → claim 向量
  - evidence: `0.5 * CLIP_text(title+caption+ocr) + 0.5 * CLIP_image(image)` → evidence 向量
  - cos 相似度排名

**实施**：

```bash
# 新建 scripts/baseline_clip_zeroshot.py
# - 100 行内可完成
# - 注意：evidence 无图时退化为纯 text 路径
```

**预算**：半天。
**预期**：R@10 约 2–6%（zero-shot CLIP 在 MR2 中文/噪声数据上较弱）。

---

### 1.3 CIEA 完整版复现（最关键）

- **为什么必做**：你的方法定位是"改进 CIEA 的互补性假设"，但 REPORT 里 `mr2_ablation_ciea` 只跑了 D-only。**审稿人 100% 会要求完整 CIEA 而非只 D 项**。
- **CIEA 完整方法包含**（参考 ACL 2025 原论文）：
  1. 跨模态互补性 D_j（已有）
  2. **CIEA 原版的 alignment 损失**（不是 EACER 的 L_comp）
  3. **patch-level cross-attention fusion**（CIEA 的 Transformer 融合）

**实施**：

```bash
# 新建 configs/mr2_ciea_full.yaml
# - 去掉 S_j 和 R_j
# - 用 CIEA 原版 alignment loss（按 ACL'25 论文公式）
# - 保留 fusion transformer
# 修改：src/models/weighting.py 加 weighting_mode: "ciea" / "ecer"
# 修改：src/losses/complementary.py 加 mode: "ciea" / "ecer"
```

**预算**：1.5–2 天（包括读 CIEA 论文 + 写代码 + 训练 + 评估）。
**预期**：作为对照点的真值，预估 R@10 在 5–10% 区间。

---

### 1.4 MR2 官方 baseline（同数据集原作者方法）

- **为什么必做**：同数据集发表过的方法必须列。
- **方法**（MR2 SIGIR'23）：CLIP image encoder + BERT text encoder，dual-encoder，cos 相似度。
- **实施**：MR2 原论文有官方代码（`https://github.com/MR2-MMRRD`），可直接跑或在 EACER 内复刻一份。
- **预算**：1 天。
- **预期**：与 CLIP fine-tuned 相近。

---

### 1.5 MOCHEG 官方 retriever

- **为什么必做**：MOCHEG 是你 README 提到要扩展的数据集，必须有官方 retriever 作对照。
- **方法**（MOCHEG SIGIR'23）：CLIP-based retrieval + linear projection head，分别针对 text evidence 和 image evidence 检索。
- **依赖**：先完成 `scripts/preprocess_mocheg.py` 调通（IMPROVEMENT_PLAN 已列）。
- **预算**：2 天（含数据 + 官方代码跑通）。

---

## 2. 强烈建议补的 baseline

### 2.1 SigLIP / BLIP-2 zero-shot

- 现代 VLM 强 baseline，二选一即可
- 推荐 SigLIP（更轻量，`google/siglip-base-patch16-256-multilingual`）
- 预算：半天

### 2.2 ColBERT-v2（late interaction）

- IMPROVEMENT_PLAN Step 8 的对照
- 用 `colbert-ir/colbertv2.0` 或多语言版（如果想跑中文）
- 预算：2 天

### 2.3 Hybrid（BM25 + BGE-M3 RRF）

- 已经有 BM25+ECER 的 hybrid，再加一个 BM25+BGE-M3 RRF 作为非 ECER 的 hybrid 对照
- 证明 EACER 的 hybrid 增益不是任何 dense 都能拿到
- 预算：1 天

---

## 3. 消融补全（论文 ablation 表）

REPORT 已有：移除 S / 移除 R / 移除 L_comp / λ 敏感性 / 熵方案 A vs B vs K1000。

需要补的：

| 消融 | 配置 | 现状 |
|---|---|---|
| 渐进累加：D-only → S+D → S+D+R → S+D+R+L_comp | 新建 configs/mr2_progressive_*.yaml | ⬜ |
| 移除整个 R 项（不是换方案） | configs/mr2_ablation_no_R.yaml | ✅ 已有 |
| 只用 entropy（无 D，无 S） | 新建 configs/mr2_only_R.yaml | ⬜ |
| Fusion 替换：Transformer → concat MLP / cross-attention | 改 fusion_mode | ⬜ 可选 |
| Hard-neg 数量 0 / 1 / 4 / 8 | 已有 hardneg 配置，加扫描 | ⬜ |
| Top-K rerank 池：50 / 100 / 200 / 500 | rerank_bm25_ecer.py 已有 | ✅ |

---

## 4. 推荐执行顺序（按性价比 + 依赖）

```
Week 1
─────
Day 1  CLIP zero-shot dual-encoder         （半天写, 1张卡半天跑）
       BGE-M3 zero-shot                    （半天）
Day 2  BGE-M3 fine-tuned on MR2            （写脚本 + 训练）
Day 3  CIEA 完整复现：配置 + 代码改动      （读论文 + 实现）
Day 4  CIEA 完整复现：训练 + 评估
Day 5  MR2 官方 baseline 跑通

Week 2
─────
Day 6  渐进消融 D → S+D → S+D+R → full     （5 个 yaml 跑完）
Day 7  SigLIP 或 BLIP-2 zero-shot          （二选一）
Day 8  BM25 + BGE-M3 RRF hybrid 对照
Day 9  ColBERT-v2 多语言（可选）
Day 10 MOCHEG preprocess 调通 + MOCHEG retriever 跑

Week 3
─────
预留：等 IMPROVEMENT_PLAN Step 2 (claim image) 完成后，
     **所有 baseline 重跑一次最终对照**（如果时间紧，至少重跑 CIEA + BGE-M3 ft + ECER）
```

**关键依赖**：所有 fine-tuned baseline **必须等 IMPROVEMENT_PLAN Step 2/3/4 全部完成后再做最终对照**，否则需重跑一次。建议：先把 zero-shot baseline 全跑完（不依赖 ECER 改动），fine-tuned baseline 留到 ECER 主模型稳定后再跑。

---

## 5. 评估口径统一（避免比较不公）

所有 baseline 必须：

- 同一份 `data/processed/mr2/mr2_test.jsonl`
- 同一份 corpus（全量 evidence 池，含无图条目）
- 同一组指标：R@1 / R@5 / R@10 / R@100 / MRR / mAP（沿用 `src/evaluate_retrieval.py`）
- 同一份评估脚本：把 `scripts/evaluate_all.py` 扩展为可接 baseline 输出（统一 jsonl 格式 `{claim_id, ranked_evidence_ids}`）
- 如果 baseline 不输出全 corpus 排名（如只 top-100），在表里注明并说明 R@100 不可比

建议新建 `scripts/baseline_runner.py` 统一封装，所有 baseline 实现 `class Baseline { encode(corpus), score(claim, evidence) }` 接口。

---

## 6. 论文层叙事建议（直接影响 baseline 选择）

参考 IMPROVEMENT_PLAN 第 4 节：

- **主结论不是 "ECER end-to-end 击败 BM25"**，而是 **"ECER 作为 reranker / hybrid signal 与 BM25 互补"**
- 所以主表应**强调 R@100**（recall ceiling）和 **BM25→ECER rerank** 这一行的所有指标
- baseline 选择上，**不要漏掉强 sparse baseline 和强 hybrid baseline**，否则无法支撑"互补"叙事
- 跨语言（中/英子集）分别报指标可加分

---

## 7. 立即可启动的 3 件事

按优先级：

1. **写 `scripts/baseline_clip_zeroshot.py`**（半天，最快出数）
2. **写 `scripts/baseline_bge_m3.py`**（半天，含 zero-shot 和 fine-tune 两个 mode）
3. **新建 `configs/mr2_ciea_full.yaml` + 改 `weighting.py` 加 mode 开关**（1.5 天，最关键的对手）

如果你点头，我可以从 (1) 开始写代码。
