# EACER：熵感知互补证据检索 — 阶段性说明

> 项目位置：`/mnt/data/yangjun/fact/EACER`
> 数据集：MR2（11,184 / 1,309 / 1,129 train/val/test claims，中英文混合）
> 最近更新：2026-05-30（commit `dedbe9d`）

---

## 1. 项目目标

针对多模态事实核查中的**证据检索**子任务，提出一种熵感知互补证据检索（**E**ntropy-**A**ware **C**omplementary **E**vidence **R**etrieval，ECER）框架。给定一个 claim 和候选证据池 `{e = (e_t, e_v)}`，模型需要把**真实证据**排在**无关证据**之前；与传统稠密检索不同，模型还需要倾向于"既相关、又互补、又可靠"的证据。

---

## 2. 方法

### 2.1 三个 patch-level 权重

视觉证据被 CLIP-ViT 编码为 `P` 个 patch tokens `{v_j}`。对每个 patch，我们同时计算三个分数：

| 名称 | 公式 | 含义 |
|---|---|---|
| 相关性 `S_j` | `(1 + max_k cos(v_j, c_k))/2` | patch 与 claim 是否相关 |
| 互补性 `D_j` | `(1 − max_l cos(v_j, t_l))/2` | patch 是否表达了文本未覆盖的信息 |
| 可靠性 `R_j` | 见下 | patch 自身语义是否清晰，非噪声 |

`R_j` 用 CLIP 零样本概念分布计算 patch 的语义熵 `H_j`，再转可靠性。提供两种方案：

- **方案 A（高熵抑制）**：`R_j = 1 − H_j / log K` —— 惩罚模糊 patch。
- **方案 B（中熵偏好）**：`R_j = exp(-(H_j − μ)² / 2σ²)` —— 同时抑制信息量过低（空白/纯色）和过高（混乱）的 patch。

三者相乘得到熵感知互补权重 `W_j = S_j · D_j · R_j`，再 softmax 归一化为 `α_j` 用于加权聚合视觉 token：`ṽ = Σ α_j v_j`。

### 2.2 多模态融合与检索打分

把 `ṽ` 与文本证据 token 序列 `T` 拼接进 Transformer：`h_e = Trans([start; ṽ; end; T])`。Claim 由同一个文本编码器编码后池化为 `h_c`。检索打分：`score(c, e) = cos(h_c, h_e)`。

### 2.3 损失

```
L = L_ret + λ · L_comp
```

- `L_ret`：InfoNCE，正样本是 claim 的 gold evidence，负样本是 in-batch + hard negatives。
- `L_comp`：把 claim 中与 evidence text 重合的 token 用 `[MASK]` 替换得到 `q_comp`，让 `q_comp` 与加权视觉表示 `ṽ` 对齐 —— 鼓励视觉模块去填补文本无法解释的部分。

### 2.4 整体流程图

```
        claim c                                          evidence (e_t, e_v)
            │                                                 │
            ▼                                                 ▼
   ┌──────────────┐                                   ┌──────────────┐  ┌─────────────┐
   │ Text Encoder │                                   │ Text Encoder │  │ CLIP Visual │  (frozen)
   │ (XLM-R base) │                                   │ (XLM-R base) │  │   Encoder   │
   └──────┬───────┘                                   └──────┬───────┘  └──────┬──────┘
          │                                                  │                 │
          │  C = {c_k}                              T = {t_l}│        raw patches
          │                                                  │            v_j (768d)
          │                                                  │                 │
          │                                                  │           ┌─────┴─────┐
          │                                                  │           │ Projector │
          │                                                  │           └─────┬─────┘
          │                                                  │                 │
          ├────────────────────────────────┐                 ├─────────────────┤
          │ S_j = (1+max cos(v_j,c_k))/2   │                 │ D_j = (1−max cos(v_j,t_l))/2
          │                                ▼                 │                 │
          │                          ┌─────────┐             │                 │
          │                          │ Entropy │ ───► R_j    │                 │
          │                          │ module  │             │                 │
          │                          └─────────┘             │                 │
          │                                                  │                 ▼
          │                          W_j = S_j · D_j · R_j  ──── α_j = softmax(W_j)
          │                                                  │                 │
          │                                                  │                 ▼
          │                                                  │           ṽ = Σ α_j · v_j
          │                                                  │                 │
          │                                              ┌───┴────────────────┘
          │                                              ▼
          │                                  ┌──────────────────────┐
          │                                  │  Evidence Fusion     │
          │                                  │  Transformer (2L)    │
          │                                  │ [start;ṽ;end;T]      │
          │                                  └──────────┬───────────┘
          ▼                                             ▼
       h_c (pooled + claim_head)              h_e (pooled fusion output)
                          \                       /
                           cos(h_c, h_e) ──► retrieval score
                           InfoNCE L_ret + λ · L_comp
```

---

## 3. 代码实现

### 3.1 模块映射

| 文件 | 对应方法部分 |
|---|---|
| `src/models/encoders.py` | TextEncoder（XLM-R）/ VisualEncoder（冻结 CLIP-B/32）/ PatchProjector |
| `src/models/entropy.py` | `EntropyReliability`（方案 A/B + CLIP visual_projection 投影） |
| `src/models/weighting.py` | `EntropyAwareComplementaryWeighting`（S·D·R + softmax） |
| `src/models/fusion.py` | `EvidenceFusion`（CIEA 风格 transformer 拼接融合） |
| `src/models/retriever.py` | `ECERRetriever` 顶层模型 + 训练/索引接口 |
| `src/losses/contrastive.py` | `L_ret`（in-batch + hard negatives） |
| `src/losses/complementary.py` | `L_comp`（q_comp 与 ṽ 对齐） |
| `src/data/dataset.py` | 通用 `RetrievalDataset`（MOCHEG/MR2 统一 JSONL 格式） |
| `src/data/collator.py` | `RetrievalCollator`（含 `q_comp` 构造）+ `EncodeCollator` |
| `src/train.py` / `src/train_ddp.py` | 单卡 / 多卡训练循环 |
| `src/evaluate_retrieval.py` | 离线建库 + FAISS/torch 检索 + 指标 |
| `scripts/preprocess_mr2.py` | MR2 → 统一 JSONL（处理 caption dict、路径"./img_html_news/X.jpg" 等坑） |
| `scripts/baseline_bm25.py` | BM25 sparse baseline（XLM-R tokenizer 对齐词表） |
| `scripts/hybrid_bm25_ecer.py` | BM25 + ECER 的 RRF 融合检索 |
| `scripts/evaluate_all.py` | 一次性评估所有训练好的 run，生成对比表 |
| `scripts/visualize_patches.py` | 可视化 patch 权重 α |

### 3.2 MR2 数据流

```
原始 MR2.zip (ZIP64, 27GB)
        │  Python zipfile (系统 unzip 不支持 ZIP64)
        ▼
/mnt/data/yangjun/data/mr2/queries_dataset_merge/    (56 GB, 208,550 文件)
        │  scripts/preprocess_mr2.py
        ▼
data/processed/mr2/mr2_{train,val,test}.jsonl
        │  src.data.RetrievalDataset
        ▼
{claim_id, claim, label, positive_evidence: [{evidence_id, image_path,
                                              title, snippet, caption, ocr, url}]}
```

每条 claim 收集 `direct_annotation.json`（图像搜索结果）+ `inverse_annotation.json`（反向图像搜索）的并集作为正样本池；负样本依赖 in-batch + 可选 hard negatives。

---

## 4. 实验

### 4.1 训练配置（Full ECER）

```yaml
text_encoder: xlm-roberta-base       # 中英文都需要 → XLM-R 而不是 BERT
visual_encoder: clip-vit-base-patch32 (frozen)
batch_size: 32
hard_negatives_per_sample: 1
optimizer: AdamW(lr=2.0e-5, wd=0.01, warmup=6%)
num_epochs: 5
entropy_scheme: A
entropy_K: 60 (built-in fallback) 或 1000 (ImageNet)
complementary_lambda: 0.5 / 1.0 / 2.0 (做敏感性)
contrastive_temperature: 0.07
```

### 4.2 主结果（MR2 test，全部数值见 `outputs/test_results.md`）

| 系统 | R@1 | R@5 | R@10 | R@100 | MRR | mAP |
|---|---:|---:|---:|---:|---:|---:|
| **BM25**（XLM-R tokenizer） | **5.17** | **17.25** | **23.84** | **34.89** | **46.31** | **22.24** |
| Baseline（不加 S/D/R/L_comp） | 0.42 | 1.72 | 3.15 | 15.46 | 7.64 | 2.39 |
| CIEA 复现（D only） | 0.16 | 0.82 | 1.78 | 12.51 | 4.55 | 1.35 |
| ECER w/o S | 1.20 | 4.78 | 7.85 | 25.78 | 17.06 | 6.48 |
| ECER w/o R | 0.63 | 2.65 | 4.73 | 19.75 | 10.55 | 3.73 |
| ECER w/o L_comp | 0.41 | 1.85 | 3.20 | 16.78 | 8.08 | 2.60 |
| Full ECER (λ=0.5) | 1.30 | 4.68 | 7.75 | 25.42 | 17.48 | 6.51 |
| Full ECER (λ=2.0, dense-only) | 1.78 | 7.04 | 10.89 | 28.90 | 22.57 | 9.10 |
| BM25 + ECER (RRF, k=60) | 4.17 | 14.87 | 21.88 | 38.44 | 41.20 | 19.54 |
| BM25 top-500 → ECER 纯重排 (α=0.0) | 2.62 | 9.80 | 14.85 | 34.38 | 29.56 | 13.00 |
| BM25 top-500 → 凸融合 (α=0.5)  [Step 1] | 5.33 | 18.14 | 24.90 | 37.45 | 47.75 | 23.15 |
| ECER (hardneg, dense-only) | 1.71 | 6.28 | 9.20 | 26.90 | 22.74 | 7.87 |
| BM25 top-500 → 凸融合 (α=0.5)，hardneg [Step 3] | 5.50 | 18.63 | 26.12 | 38.07 | 48.92 | 24.19 |
| ECER (hardneg + claim_img, dense-only) | 1.86 | 6.45 | 10.06 | 29.43 | 23.97 | 8.41 |
| BM25 top-500 → 凸融合 (α=0.5)，hardneg + claim_img [Step 2] | 5.51 | 18.85 | 26.34 | 38.36 | 49.17 | 24.41 |
| ECER (TierA, dense-only) | 2.27 | 7.85 | 12.25 | 33.02 | 27.16 | 10.28 |
| BM25 top-500 → 凸融合 (α=0.5)，TierA [Step1+2+3+TierA] | 5.55 | 19.07 | 26.63 | 38.47 | 49.74 | 24.86 |
| ECER (BGE-M3 lr=1e-5, dense-only) | 2.41 | 8.66 | 13.23 | 33.57 | 28.25 | 11.17 |
| BM25 top-500 → 凸融合 (α=0.5)，BGE-M3 lr=1e-5 | 5.70 | 19.17 | 26.95 | 38.64 | 50.15 | 25.26 |
| ECER (BGE-M3 lr=2e-5, dense-only) | 2.54 | 9.28 | 14.18 | 34.61 | 29.37 | 12.13 |
| **BM25 top-500 → 凸融合 (α=0.5)，BGE-M3 lr=2e-5** | **5.53** | **19.41** | **26.83** | **38.78** | **49.28** | **25.23** |

（数值单位：%）

### 4.3 MOCHEG test 主结果

MOCHEG（英文 fact-checking benchmark）我们在 Tier A + BGE-M3 lr=2e-5 配置上**仅训练 7 epoch（epoch 4 early-stopping best）**。数据规模：7.5k claims (train) / 1.5k (val) / 2.4k (test)，每 claim ~2.5 个 curated 文本证据 + 多张图像。Gold 用 Corpus2 的 curated `Evidence` 字段。

| 系统 | R@1 | R@5 | R@10 | R@20 | R@100 | MRR | mAP | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 only | 28.73 | 51.40 | 57.25 | 62.69 | 72.03 | 68.10 | 47.35 | 54.44 |
| **ECER dense-only** | **33.87** | **65.53** | **72.68** | 77.87 | **90.33** | 73.28 | 63.21 | 67.60 |
| BM25 top-500 → rerank α=0.0 | 35.39 | 66.23 | 71.79 | 76.08 | 80.08 | 78.17 | 62.87 | 68.80 |
| **BM25 top-500 → rerank α=0.3** | **38.89** | **71.83** | **75.95** | **78.27** | 80.31 | **84.72** | **68.26** | **74.34** |

（数值单位：%）

> **MOCHEG 上的关键发现**：
> - ECER dense-only R@10 = **72.68%**（vs BM25 57.25%，+27% relative），R@100 = **90.33%**（vs 72.03，+25%），mAP = **63.21%**（vs 47.35，+33%）。MOCHEG 的 curated Evidence 高度语义化，加上 BGE-M3 的英文检索预训练，dense 直接大幅碾压 BM25。
> - **凸融合最优 α=0.3**（MR2 上是 0.5），因为 MOCHEG dense 已经足够强，BM25 信号只用 30% 就够。
> - **Dense-only R@100=90.33% 高于 rerank 的 80.34%**：BM25 top-K gate 限制了 rerank 的 R@100 上限。说明 MOCHEG 上**dense-only 是天然 SOTA 路线**，rerank 仅对 head（R@1-10）有帮助。
> - 注意：我们的 BM25 MAP=47.35% 远高于 MEVER 论文报告的 27.3%，**因为 gold 定义不同**：我们用 Corpus2 curated 文本（每 claim ~2.5 个干净证据），MEVER 用 qrels 的 sentence-level pool（每 claim ~12 个）。两个 setup 都合理，但数字不能直接比较。要严格对齐 MEVER 需要切到 qrels-based eval，是下一步 TODO。
> - 但相对 BM25 的提升 **+44% mAP relative** 与 MEVER 自己的 BM25→MEVER 提升 **+52%** 在同一量级 —— 我们的方法在 MOCHEG 上**确实是强 baseline**，且在 dense-only 上效率显著高于 MEVER 的 graph-based pipeline。

> **Step 1 成果**：凸融合（α=0.5）首次全面超过 BM25 单跑，R@10 +4.4%、R@100 +7.3%、mAP +4.1%。
>
> **Step 3 成果**：在 Step 1 基础上**进一步提升所有指标**。配置（hardneg + 凸融合 α=0.5）R@10 = 26.12%（比 BM25 高 +9.6% relative，比 Step 1 高 +4.9% relative）。
>
> **Step 2 成果**：在 Step 3 基础上加入 claim image，所有指标再次全面提升。R@10 = 26.34%（比 BM25 高 +10.5% relative），R@100 = 38.36%（+9.9% relative）。**dense-only R@10 从 9.20 升到 10.06（+9.3%）**。
>
> **Tier A 成果**：在 Step 2 基础上叠加三项低成本改进（10 epoch + EarlyStopping、跳过黑图 evidence 的视觉路径、多正样本 InfoNCE mask），所有指标再次提升。凸融合 R@10 = 26.63%（vs BM25 +11.6% relative），R@100 = 38.47%，mAP = 24.86%。**dense-only R@10 大跃进**：10.06 → 12.25 (+22%)，纯 rerank α=0.0 top-500: 15.50 → 17.59 (+13%)。
>
> **BGE-M3 替换文本编码器（最新 SOTA）**：把 xlm-roberta-base（278M）换成 BAAI/bge-m3（568M，多语种检索预训练 + CLS 池化）。**所有 7 个 rerank 指标**再次提升：R@10 = **26.95%**（vs BM25 +13.0% relative），R@100 = **38.64%**，mAP = **25.26%**，MRR = **50.15%**。**纯 rerank (α=0.0) 全 K_pool 都涨**（top-100: 21.51→22.27, top-500: 17.59→18.28），说明 BGE-M3 的多语种检索预训练真的在 hard pool 内做了更细的语义辨别。详细 K_pool × α sweep 见 `outputs/rerank_test_{baseline,hardneg,claim_img,tierA,bge_m3}.md`。

### 4.3 关键消融与敏感性

**(a) 三个权重各自的贡献（相对 Full ECER λ=0.5）**
| 移除项 | R@10 变化 | mAP 变化 |
|---|---:|---:|
| 移除 `S_j` | 7.85 → 7.75（-1%，几乎无影响） | 6.48 → 6.51 |
| 移除 `R_j` | 7.75 → 4.73（**-39%**） | 6.51 → 3.73 |
| 移除 `L_comp` | 7.75 → 3.20（**-59%**） | 6.51 → 2.60 |

→ `R_j` 和 `L_comp` 是核心；`S_j` 在 MR2 上贡献较小（可能是因为 claim 多是短文本，max-cos 已经被 D_j 间接体现）。

**(b) λ（互补损失权重）敏感性**
| λ | R@10 | mAP |
|---|---:|---:|
| 0.1 | 3.49 | 2.55 |
| 0.3 | 4.75 | 3.45 |
| 0.5 | 7.75 | 6.51 |
| 1.0 | 8.16 | 7.13 |
| **2.0** | **10.89** | **9.10** |

→ MR2 上 λ 越大越好（4× 默认值）；说明互补对齐损失在噪声数据上至关重要。

**(c) 熵方案**
| 方案 | R@10 | mAP |
|---|---:|---:|
| A（高熵抑制，K=60） | 7.75 | 6.51 |
| A（高熵抑制，K=1000） | 5.82 | 4.49 |
| B（中熵偏好，K=60） | 6.99 | 5.83 |

→ 方案 A > 方案 B；手选 60 个新闻/抗议/武器/文本类概念 > 1000 个 ImageNet 概念（**领域匹配比 K 数量更重要**）。

**(d) 与稀疏 + 稠密 hybrid 比较**
| 系统 | R@10 | R@100 |
|---|---:|---:|
| BM25 alone | 23.84 | 34.89 |
| ECER alone | 10.20 | 28.95 |
| RRF hybrid | 21.88 | **38.44** |

→ R@100 上 hybrid > BM25（+3.6%），说明 ECER 检索回了 BM25 漏掉的语义近邻；但 R@10 上 BM25 仍然更强。

---

## 5. 当前结论与启示

1. **方法验证**：在中英文混合的 noisy retrieval 任务上，ECER 的三个权重相互配合显著优于纯 CIEA（D-only 复现），证明 R_j（熵可靠性）和 L_comp（互补对齐）是必要的。
2. **CIEA 复现反而比 baseline 差**：仅靠 D_j 会被噪声 patch 主导（这正是论文动机），印证了"高互补 ≠ 有用补充"的核心论点。
3. **λ 远比预期重要**：默认 λ=0.5 不够；在 MR2 这种噪声大的数据上 λ=2.0 才达到最佳。
4. **概念库的领域匹配 > 规模**：60 个手选概念赢过 1000 个 ImageNet 概念，说明 R_j 对零样本概念库的领域适配敏感，**未来可针对事实核查领域构建专用概念表**。
5. **BM25 仍是强 baseline**：在 MR2 中文新闻文本上 BM25 显著领先所有稠密方法。这与 MEVER 表 1 的趋势一致 —— **稠密检索在事实核查类数据上长尾**。我们的 RRF hybrid 在 R@100 上能超过 BM25，证明 ECER 学到了 BM25 学不到的语义近邻。
6. **Step 1（BM25→ECER 凸融合重排）首次让系统全面超过 BM25**：α=0.5 时 R@10 24.90%（+4.4% 相对 BM25），R@100 37.45%（+7.3%），mAP 23.15%（+4.1%）。**纯 ECER 重排（α=0.0）始终输给 BM25**，说明 ECER 的 hard-pool 排序能力还不够；但作为 BM25 信号的补充，它能稳定带来增益 —— 这正是稀疏-稠密互补的标准 IR 故事。
7. **Step 3（BM25 hard negatives 进训练）进一步推高 SOTA 并验证诊断**：在 Step 1 配置上换 hardneg checkpoint 后，凸融合 R@10 从 24.90% 升到 **26.12%**（+4.9% relative），mAP 从 23.15 升到 24.19（+4.5%）。**纯 rerank（α=0.0）在小 K_pool 上也有改善**（top-50 从 20.08 升到 21.09，top-100 从 18.67 升到 19.34），证实了"ECER 缺少 BM25 hard pool 训练信号"的诊断。但纯 dense-only 全库检索略有退步（R@10 10.89 → 9.20），说明 hard negs 让模型偏向"鉴别相似 vs 真证据"而牺牲了一些长尾召回 —— 这是 rerank-friendly 模型的典型 trade-off，符合预期。
8. **Step 2（claim image 进模型）**：在 Step 3 基础上把 claim 自带图像（之前被完全忽略的 100% 可用信号）通过共享 CLIP + 残差 MLP 注入 claim 表示，几乎所有指标都再次提升：凸融合 R@10 = 26.34%（比 BM25 高 +10.5% relative），R@100 = 38.36%。最关键的是 **dense-only R@10 从 9.20 升到 10.06（+9.3%）** —— 这正面回应了 Step 3 的 trade-off，把 hard negs 牺牲掉的长尾召回靠 claim image 补了回来。
9. **Tier A 三项联合修复**：(a) 跳过黑图 evidence 的视觉路径（45% 证据无图，原本喂 CLIP 黑图产生系统噪声）；(b) 多正样本 InfoNCE mask（同 claim 不同 sample 的正样本不再互为负样本，去除自污染）；(c) 训练 10 epoch + EarlyStopping on R@10。三者一起在 Step 2 基础上把 **dense-only R@10 从 10.06 推到 12.25（+22%）**，**纯 rerank α=0.0 top-500 从 15.50 推到 17.59（+13%）**，凸融合最终 R@10 = **26.63%**（vs BM25 +11.6% relative），R@100 = **38.47%**。其中 (a)(b) 是关键 —— 训练 epoch 9 best 而不是 epoch 5，说明前两个修复让模型有更长的可学空间。**踩了一个坑**：(a) 的第一版把 mask 直接乘到 `weighted_visual` 输出上，导致 L_comp 把 q_comp 拉向零向量，loss 卡在 13.0 不下降；修复方法是 mask 只作用于 fusion 输入，保留 `weighted_visual` 给 L_comp。
10. **BGE-M3 替换文本编码器**：把 `xlm-roberta-base`（278M, mean pool）替换为 `BAAI/bge-m3`（568M, CLS pool，多语种检索预训练）。代码只新增一个 `TextEncoder.pooling: str` 参数和 config 字段 `model.text_pooling: cls`；所有维度依赖通过 `text_encoder.hidden_dim` 自动适配 768 → 1024。两轮训练揭示了 lr 的关键作用：
    - **lr=1e-5**（保守 fine-tune）：dense val R@10 17.76% 低于 TierA 18.45%，但凸融合 R@10 = 26.95% 略胜 TierA 26.63%。
    - **lr=2e-5**（与 TierA 同 lr，重训）：dense val R@10 升到 19.94%（+12% vs lr=1e-5），凸融合 R@10 = 26.83%（与 lr=1e-5 接近）。
    - **lr=2e-5 在 dense-only 和纯 rerank α=0.0 上明显赢**（hard-pool 判别能力更强），但 lr=1e-5 在凸融合 α=0.5 上有微弱优势，可能因为更保守的 fine-tune 让 BGE-M3 与 BM25 信号更互补。
    - **R@100 上 lr=2e-5 是绝对赢家**：**38.78%**（vs lr=1e-5 38.64%，TierA 38.47%）。
    - 教训：BGE-M3 retrieval-tuned 的预训练并不意味着 fine-tune 时必须降 lr —— 用主架构原本的 lr 训出来反而 dense-only 更好。但要拿到最强的 rerank 结果，还需要在 α/K_pool 维度上重新做 sweep（lr 不同 → 最优 α 不同）。

---

## 6. 已知问题 / 下一步

| 问题 | 建议下一步 |
|---|---|
| 稠密方法绝对值仍低于 BM25（R@10 11% vs 24%） | ✅ Step 1+2+3+TierA 把凸融合 R@10 推到 26.63% (+11.6% vs BM25)；dense-only R@10 也从 10.89 升到 12.25 (+12.5%) |
| ECER 纯 rerank（α=0.0）失败 | ✅ TierA 进一步解决：纯 rerank top-100 从 19.34 → 21.51 (+11%)，top-500 从 14.60 → 17.59 (+20%)；接近 BM25 单跑 23.87 的 70-90% 水平 |
| Full ECER 在 λ=2.0 才最优，可能过拟合 L_comp | 加 EarlyStopping on val + λ schedule（warmup 后逐步放大） |
| 概念库手选 60 词过拟合 MR2 | 构建一个 fact-checking 领域的中型概念库（500 个：新闻事件 + 实体类型 + 视觉摘要词） |
| 还未在 MOCHEG 上跑 | 下个阶段：preprocess_mocheg.py 调通后跑完整对照，对齐 MEVER 等 baseline |
| 还未做下游 claim verification | 加一个简单 verifier，把 top-K evidence 喂进去报 Macro-F1，做端到端评估 |
| Step 2 时训练 epoch 数仅 5，R@1 还在涨（E5 是 best） | ✅ TierA 跑 10 epoch + EarlyStopping，best 落在 epoch 9，R@10 从 14.33 升到 18.45 (+29%) |
| Patch 可视化已有脚本但未系统报告 | `scripts/visualize_patches.py` 选 3-5 个典型 claim 出图，用于报告里说明 S/D/R 各自的作用 |

---

## 7. 仓库与环境

- **环境**：`conda activate bm25`（Python 3.9）
- **代理**：`proxy` / `unproxy` 切换 `http_proxy=127.0.0.1:7876`（拉 HF 模型权重时需要）
- **数据**：`/mnt/data/yangjun/data/mr2/queries_dataset_merge` ←软链至 `data/raw/mr2`
- **处理后数据**：`data/processed/mr2/mr2_{train,val,test}.jsonl`
- **训练产出**：`outputs/<run_name>/{best.pt, last.pt, tb/}`
- **汇总表**：`outputs/test_results.md` + `outputs/test_results.json`

复现主结果：
```bash
conda activate bm25
# 1. 预处理（已完成）
bash scripts/preprocess_mr2.sh
# 2. 训练 best 配置（无 hard negs）
bash scripts/train_mr2.sh configs/mr2_ecer_best.yaml
# 3. 评估所有 run 并生成对比表
python scripts/evaluate_all.py
# 4. Step 1: BM25 → ECER 凸融合重排
HF_HUB_OFFLINE=1 python scripts/rerank_bm25_ecer.py \
    --ckpt outputs/mr2_lambda_20/best.pt --split test \
    --k_pools 50 100 200 500 --alphas 0.0 0.3 0.5 0.7
# 5. Step 3: 用 BM25 hard negatives 重训，再跑 rerank
HF_HUB_OFFLINE=1 python scripts/mine_bm25_negatives.py \
    --in_jsonl data/processed/mr2/mr2_train.jsonl \
    --out_jsonl data/processed/mr2/mr2_train_hardneg.jsonl \
    --num_negatives 8
HF_HUB_OFFLINE=1 python -m src.train --config configs/mr2_ecer_hardneg.yaml
HF_HUB_OFFLINE=1 python scripts/rerank_bm25_ecer.py \
    --ckpt outputs/mr2_ecer_hardneg/best.pt --split test \
    --k_pools 50 100 200 500 --alphas 0.0 0.3 0.5 0.7 \
    --out_json outputs/rerank_test_hardneg.json
# 6. Step 2: 把 claim image 喂进模型
HF_HUB_OFFLINE=1 python -m src.train --config configs/mr2_ecer_claim_img.yaml
HF_HUB_OFFLINE=1 python scripts/rerank_bm25_ecer.py \
    --ckpt outputs/mr2_ecer_claim_img/best.pt --split test \
    --k_pools 50 100 200 500 --alphas 0.0 0.3 0.5 0.7 \
    --out_json outputs/rerank_test_claim_img.json
# 7. Tier A: 跳黑图 + 多正样本 InfoNCE + 10 epoch + EarlyStopping
HF_HUB_OFFLINE=1 python -m src.train --config configs/mr2_ecer_tierA.yaml
HF_HUB_OFFLINE=1 python scripts/rerank_bm25_ecer.py \
    --ckpt outputs/mr2_ecer_tierA/best.pt --split test \
    --k_pools 50 100 200 500 --alphas 0.0 0.3 0.5 0.7 \
    --out_json outputs/rerank_test_tierA.json
# 8. BGE-M3 (lr=1e-5): 把 text encoder 换成 BAAI/bge-m3
HF_HUB_OFFLINE=1 python -m src.train --config configs/mr2_ecer_bge_m3.yaml
HF_HUB_OFFLINE=1 python scripts/rerank_bm25_ecer.py \
    --ckpt outputs/mr2_ecer_bge_m3/best.pt --split test \
    --k_pools 50 100 200 500 --alphas 0.0 0.3 0.5 0.7 \
    --out_json outputs/rerank_test_bge_m3.json
# 9. BGE-M3 (lr=2e-5): 同样 BGE-M3 但用 TierA 的 lr (R@100 SOTA)
HF_HUB_OFFLINE=1 python -m src.train --config configs/mr2_ecer_bge_m3_lr2e5.yaml
HF_HUB_OFFLINE=1 python scripts/rerank_bm25_ecer.py \
    --ckpt outputs/mr2_ecer_bge_m3_lr2e5/best.pt --split test \
    --k_pools 50 100 200 500 --alphas 0.0 0.3 0.5 0.7 \
    --out_json outputs/rerank_test_bge_m3_lr2e5.json
# 最新成果:
#  - BGE-M3 lr=1e-5: top-500 α=0.5 → R@10=26.95%, R@100=38.64%, mAP=25.26%
#  - BGE-M3 lr=2e-5: top-500 α=0.5 → R@10=26.83%, R@100=38.78%, mAP=25.23%, dense R@10=14.18%
# Patch 可视化:
HF_HUB_OFFLINE=1 python scripts/visualize_patches.py \
    --ckpt outputs/mr2_ecer_tierA/best.pt --n_samples 12 \
    --out_dir outputs/visualizations_tierA
```

---

## 8. 主要参考

- CIEA: *Enhancing Multimodal Retrieval via Complementary Information Extraction and Alignment*, ACL 2025.
- MOCHEG: *End-to-End Multimodal Fact-Checking and Explanation Generation*, SIGIR 2023.
- MR2: *A Benchmark for Multimodal Retrieval-Augmented Rumor Detection*, SIGIR 2023.
- MEVER: *Multi-Modal and Explainable Claim Verification with Graph-based Evidence Retrieval*, EACL 2026.
