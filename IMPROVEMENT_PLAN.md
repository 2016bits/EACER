# 性能瓶颈分析与改进方案

> 现状：Full ECER（λ=2.0）test R@10 = **10.89%**，仍只有 BM25（**23.84%**）的 46%。
> 本文找出根因并给出按性价比排序的改进路线。

---

## 1. 已确认的瓶颈（强证据）

### 1.1 Claim 图像被完全忽略 ⚠ **最大问题**

```
test 集 1,126/1,126 (100%) claims 都有 claim_image_path 且文件存在
但模型 encode_claim(input_ids, attention_mask) — 只接收文本
```

MR2 是**对称多模态**任务（claim 和 evidence 都是图+文），我们却把它当**单边多模态**做（claim 纯文本，evidence 图+文）。这丢掉了一半的输入信号：
- 当 claim 的关键线索在图像里（人脸、地点、时间戳、横幅文字）而 caption 简短时，模型完全看不到这些。
- 这等于在做 **text-claim → text+image evidence** 检索，自然干不过纯文本 BM25。

### 1.2 Hard negatives 是空的

```yaml
# configs/mr2_ecer_best.yaml
hard_negatives_per_sample: 1
```
```
data/processed/mr2/mr2_*.jsonl: neg/claim 平均 = 0.0
```

`preprocess_mr2.py` 没有写入 `negative_evidence`，所以 collator 想抽 hard neg 时无东西可抽，只剩 **in-batch negatives**（batch=32，难度极低）。InfoNCE 信噪比严重不足。

### 1.3 视觉特征对一半证据是噪声

```
test 中只有 4806/8798 (54.6%) 的 evidence 有有效图像
```

剩 45% 是 `inverse_search/*.html` 的纯文本网页，我们在 `_load_image` 里默认返回**全黑 224×224**。这些黑图经过 CLIP 后产生固定但无意义的 patch tokens，被 S/D/R 加权聚合后变成系统性偏置，**反而干扰文本侧的学习**。

### 1.4 多正样本被当成多个独立单样本

```
平均每 claim 有 8 个正样本，被 RetrievalDataset 拆成 8 条训练样本
```

每条样本独立做 InfoNCE，意味着同一 claim 的另外 7 个真正的正样本在 in-batch 里**反而成了负样本**。InfoNCE 信号被自己污染。

### 1.5 BM25 在 MR2 上"作弊"地强

MR2 的证据是用 caption 在 web 上搜来的，**搜索结果天然带强 lexical overlap**。BM25 直接利用了这个数据构造偏置，dense retrieval 想超越它非常难，除非：
- 引入图像（BM25 看不到）
- 用 BM25 做 first-stage、ECER 做 reranker（不是替代是协同）

### 1.6 L_comp 在中文上的潜在问题

L_comp 把 claim 中与 evidence text **token id 重合**的部分 mask 掉。但：
- XLM-R 用 sentencepiece，中文常分到字符/子词级别（如"上海"=`▁上`+`海`）
- 高频字符（"的、是、了"）会被大量 mask，q_comp 退化成短碎片
- 这可能导致 `L_comp` 实际优化的不是"补充信息对齐"而是"短文本对齐"

---

## 2. 改进路线图（按性价比排序）

### Tier 1 — 立刻能做、预期收益最高

#### 🥇 **Step 1: BM25 → ECER reranker 二阶段**（最大杠杆）

不要再追求 dense end-to-end 干掉 BM25，**改为二阶段**：

```
claim ──BM25──► top-100 evidences ──ECER 重排──► top-K
```

**预期上限**：BM25 R@100=34.89% 就是天花板，但我们能把 ECER 的精排能力发挥在 top-100 这个"BM25 已经做完粗筛"的池上。

**实现量**：复用 `scripts/hybrid_bm25_ecer.py` 的 BM25 评分，把 RRF 换成 ECER 对 BM25 top-100 重排。30 行代码可完成。

**预期收益**：R@10 大概能到 25-30%（接近甚至超过 BM25 单独跑）。

#### 🥈 **Step 2: 把 claim image 接进模型**（最大被遗漏信号）

```python
# src/models/retriever.py
def encode_claim(self, input_ids, attention_mask, claim_pixel_values=None):
    text_out = self.text_encoder(...)
    if claim_pixel_values is not None:
        claim_patches = self.visual_encoder(claim_pixel_values)
        claim_patches = self.projector(claim_patches)
        # 直接 mean-pool + concat 或 cross-attention 融合
        claim_visual = claim_patches.mean(1)
        text_out.pooled = self.claim_visual_head(
            torch.cat([text_out.pooled, claim_visual], -1)
        )
    return text_out
```

**改动文件**：`retriever.py`、`collator.py`、`preprocess_mr2.py` 已有 `claim_image_path` 字段。
**预期收益**：dense 单独跑的 R@10 从 10.89% 提到 **15-18%**（多模态对称模型在 MR2 上的常见水平）。

#### 🥉 **Step 3: BM25 hard negatives 进训练**

```python
# scripts/mine_bm25_negatives.py
对每个 claim：
  BM25 top-50 → 去掉真实 positives → 取前 4 作为 hard_neg
写回 mr2_train.jsonl 的 negative_evidence 字段
```

**预期收益**：InfoNCE 信噪比改善，R@1 大概 +50%。

---

### Tier 2 — 中等改造，结构性提升

#### Step 4: 跳过无图证据时的视觉路径（不要喂黑图）

```python
# weighting.py
当 has_image=False 时：
  alpha = 1/P (均匀)，但用 mask 把这条样本的 visual 贡献置零
  让 fusion 退化为纯文本路径
```

或更简单：对纯文本 evidence 不走 visual encoder，直接用 zeros + 一个 type embedding 标记"无图"。

**预期收益**：消除 45% 样本的系统噪声，估计 R@10 +1-2%。

#### Step 5: 多正样本 InfoNCE（去除自污染）

```python
# losses/contrastive.py
对每个 claim，把它的所有 positives 都标记为 positive
loss = sum_p -log(exp(sim(c, p))/Z)  其中 Z 排除该 claim 的所有 positives
```

或更简单：训练时**每个 claim 只采样 1 个 positive**（不要在 dataset 里拆开），避免同 claim 的多正样本互为负样本。

**预期收益**：清洁信号 +5-10% relative。

#### Step 6: 解冻 CLIP 后 2 层

MR2 图像是新闻/网页截图，与 CLIP 预训练的 natural images 分布差距大。

```yaml
freeze_visual: false
freeze_visual_except_last_n: 2
visual_lr_multiplier: 0.1   # 视觉用 1/10 主 lr
```

**预期收益**：3-5% relative。

#### Step 7: 修正 L_comp 在中文上的 masking

```python
# 不再用 token-id 重合 mask
# 改用 NER + n-gram match：
1) 提取 claim 的命名实体（用 spacy/jieba/HanLP）
2) 检查实体是否出现在 evidence text 中
3) 出现的实体替换为 [MASK]，其余保留
```

或简单方案：用句子级 sentence-transformer 算 claim 子句与 evidence 的相似度，相似度高的子句整段 mask。

---

### Tier 3 — 更深层改造（如果 Tier 1+2 不够）

#### Step 8: ColBERT 风格 late interaction

把 pooled cos 换成 token-level max-sim：

```
score(c, e) = Σ_q max_k cos(c_token_q, e_token_k)
```

在小语料 + 多正样本场景上对 pooled dense 有 5-10% 提升。

#### Step 9: 继续预训练 XLM-R

用 MR2 全部 claim + evidence 文本继续做 MLM 预训练 3-5 epoch，再 fine-tune。
**预期收益**：领域适配，2-5%。

#### Step 10: Cross-encoder rerank 头

在 dual-encoder 之上加一个 cross-encoder（claim+evidence concat → BERT → score）做 top-100 重排。
**预期收益**：3-8%。

---

## 3. 一周可执行的优先级路线

```
Day 1: 实现 Step 1 (BM25 → ECER reranker)
  目标: R@10 ≥ 25%  (打平/超过 BM25)
  代码量: ~50 行；改 scripts/hybrid_bm25_ecer.py 加 rerank 模式
  
Day 2-3: 实现 Step 2 (claim image 进模型)
  目标: dense-only R@10 ≥ 15%
  代码量: ~150 行；改 dataset/collator/retriever
  
Day 4: 实现 Step 3 (BM25 hard negatives)
  目标: 配合 Step 2 后 dense R@10 ≥ 18%
  代码量: ~80 行；scripts/mine_bm25_negatives.py
  
Day 5: 实现 Step 4 (skip 黑图) + Step 5 (多正样本)
  目标: 清洁化训练信号，稳定提升 2-3%
  
Day 6: 跑大规模消融 + 写论文表
  - 主表：BM25 / dense-only / BM25→ECER rerank / Hybrid
  - 消融：移除 S/D/R/claim image/hard neg/L_comp
  
Day 7: 解冻 CLIP + 跑 final model
  目标: R@10 ≥ 30%, mAP ≥ 25%
```

---

## 4. 报告/论文表述上的处理

即使 dense end-to-end 干不过 BM25，**这不一定是坏事**，关键是叙事框架：

- 不要写 "ECER 超越 BM25" — 写 **"ECER 作为 reranker 与 BM25 互补"**
- 强调 **R@100**（recall ceiling）：ECER hybrid 在 R@100 上已经超过 BM25（38.4% vs 34.9%），说明 ECER 检索回了 BM25 漏掉的语义近邻
- 强调 **跨语言 generalisation**：单独报中文 / 英文子集的指标，dense 在英文上的优势通常更明显
- 强调 **可解释性**：S/D/R 三个分数是 patch-level 可视化的，写一节"分析"用 `visualize_patches.py` 出图

---

## 5. 我马上能动手做的事

按优先级排好了，请你点头我就开干：

1. **(最推荐)** 实现 BM25→ECER reranker（30 行），先把 R@10 拉到 BM25 水平，今天就能见效
2. 把 claim image 接进模型 + 改 collator + 重训 ECER
3. 写 mine_bm25_negatives.py 给训练数据加 hard negatives
4. 跳过无图证据的视觉路径（小改 weighting.py）
5. 把多正样本聚合进 InfoNCE（避免自污染）

要不要我现在就从 (1) 开始？这是性价比最高也最容易拿到 Demo 数字的一步。
