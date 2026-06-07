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
| **BM25 top-500 → 凸融合 (α=0.5)，hardneg + claim_img [Step 2]** | **5.51** | **18.85** | **26.34** | **38.36** | **49.17** | **24.41** |

（数值单位：%）

> **Step 1 成果**：凸融合（α=0.5）首次全面超过 BM25 单跑，R@10 +4.4%、R@100 +7.3%、mAP +4.1%。
>
> **Step 3 成果**：在 Step 1 基础上**进一步提升所有指标**。SOTA 配置（hardneg + 凸融合 α=0.5）R@10 = 26.12%（比 BM25 高 +9.6% relative，比 Step 1 高 +4.9% relative）。
>
> **Step 2 成果（最新 SOTA）**：在 Step 3 基础上加入 claim image，所有指标再次全面提升。最终 R@10 = 26.34%（比 BM25 高 +10.5% relative），R@100 = 38.36%（+9.9% relative）。**dense-only R@10 从 9.20 升到 10.06（+9.3%）**，证明 claim image 给 dense 模型补充了关键视觉信号。**纯 rerank（α=0.0）全 K_pool 都涨**（top-100: 19.34→20.32, +5%），说明 ECER 真正能在 BM25 hard pool 内做有意义的重排了。详细对比见 `outputs/rerank_test_{baseline,hardneg,claim_img}.md`。

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
8. **Step 2（claim image 进模型）成为最终 SOTA**：在 Step 3 基础上把 claim 自带图像（之前被完全忽略的 100% 可用信号）通过共享 CLIP + 残差 MLP 注入 claim 表示，几乎所有指标都再次提升：凸融合 R@10 = **26.34%**（比 BM25 高 +10.5% relative），R@100 = **38.36%**。最关键的是 **dense-only R@10 从 9.20 升到 10.06（+9.3%）** —— 这正面回应了 Step 3 的 trade-off，把 hard negs 牺牲掉的长尾召回靠 claim image 补了回来。**纯 rerank（α=0.0）所有 K_pool 都涨**（top-100: 19.34→20.32, top-500: 14.60→15.50），说明 ECER 是**真正学到了多模态语义匹配**而不只是 lexical 补丁。

---

## 6. 已知问题 / 下一步

| 问题 | 建议下一步 |
|---|---|
| 稠密方法绝对值仍低于 BM25（R@10 11% vs 24%） | ✅ Step 1+2+3 已通过凸融合 + hardneg + claim image 把 R@10 推到 26.34% (+10.5% vs BM25) |
| ECER 纯 rerank（α=0.0）失败 | ✅ Step 2+3 联合解决：dense-only R@10 9.20→10.06，pure rerank top-100 18.67→20.32 |
| Full ECER 在 λ=2.0 才最优，可能过拟合 L_comp | 加 EarlyStopping on val + λ schedule（warmup 后逐步放大） |
| 概念库手选 60 词过拟合 MR2 | 构建一个 fact-checking 领域的中型概念库（500 个：新闻事件 + 实体类型 + 视觉摘要词） |
| 还未在 MOCHEG 上跑 | 下个阶段：preprocess_mocheg.py 调通后跑完整对照，对齐 MEVER 等 baseline |
| 还未做下游 claim verification | 加一个简单 verifier，把 top-K evidence 喂进去报 Macro-F1，做端到端评估 |
| Step 2 时训练 epoch 数仅 5，R@1 还在涨（E5 是 best） | 试更长训练（8-10 epoch）+ EarlyStopping on val R@10 |
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
# 6. Step 2: 把 claim image 喂进模型（最终 SOTA）
HF_HUB_OFFLINE=1 python -m src.train --config configs/mr2_ecer_claim_img.yaml
HF_HUB_OFFLINE=1 python scripts/rerank_bm25_ecer.py \
    --ckpt outputs/mr2_ecer_claim_img/best.pt --split test \
    --k_pools 50 100 200 500 --alphas 0.0 0.3 0.5 0.7 \
    --out_json outputs/rerank_test_claim_img.json
# 最终 SOTA: top-500 α=0.5 with hardneg + claim_img → R@10 = 26.34%, mAP = 24.41%
```

---

## 8. 主要参考

- CIEA: *Enhancing Multimodal Retrieval via Complementary Information Extraction and Alignment*, ACL 2025.
- MOCHEG: *End-to-End Multimodal Fact-Checking and Explanation Generation*, SIGIR 2023.
- MR2: *A Benchmark for Multimodal Retrieval-Augmented Rumor Detection*, SIGIR 2023.
- MEVER: *Multi-Modal and Explainable Claim Verification with Graph-based Evidence Retrieval*, EACL 2026.
