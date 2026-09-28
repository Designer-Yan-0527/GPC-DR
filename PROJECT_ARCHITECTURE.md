# FedTA → FedSMR 技术架构文档

---

## 一、原始 FedTA 架构 (CVPR 2025)

### 1.1 问题定义

联邦类增量持续学习 (Federated Class-Incremental Continual Learning, FCIL):
- K=5 个客户端，每个经历 T=5 个类增量任务
- **空间异质性**: 不同客户端类别分布不同 (私有类 + 公共类)
- **时间异质性**: 每个客户端按时间顺序经历任务，旧数据不可回放

### 1.2 双阶段 + 双 Head 设计

```
              Frozen Pretrained ViT-B/16
                        │
        ┌───────────────┴───────────────┐
        │   Stage 1: Input Enhancement  │
        │   L2P Prompt Pool             │
        │   vit.head (分类头)            │
        │        │                      │
        │   SIKF (prompt fusion)        │  ← 服务器通信
        │   vit.head FedAvg             │  ← 服务器通信
        │        │                      │
        │   输出: feat_prompt            │
        └───────────────┬───────────────┘
                        │
        ┌───────────────┴───────────────┐
        │   Stage 2: Tail Anchor        │
        │   Key Pool + Anchor Pool      │
        │   model.head (分类头)          │
        │        │                      │
        │   heads[task] per-task 快照    │  ← 本地保存，不做服务器聚合
        │        │                      │
        │   BGPS (prototype selection)  │  ← 服务器通信
        │   输出: feat_mixed             │
        └───────────────────────────────┘
```

**关键区分**:
| | vit.head | model.head |
|---|---|---|
| 所属阶段 | Stage 1: Input Enhancement | Stage 2: Tail Anchor |
| 定义位置 | `vision_transformer` / L2P prompt head | `Tail_Anchor.head` (Chead) |
| 服务器聚合 | ✅ FedAvg → `global_head` | ❌ 不做聚合 |
| 评估使用 | 不直接用于评估 | `evaluate()` 中 `load_head(heads[task])` |

### 1.3 服务器通信协议

```
for round 0..24:  # 5 tasks × 5 global_epochs
  task_id = round // global_epoch

  1. 客户端本地训练
     Phase 1: 训练 Prompt (L2P)
     Phase 2: 训练 Tail Anchor (Key + Anchor + Head)

  2. 服务器聚合 (仅以下三项):
     a. BGPS — 贪婪原型选择 (cosine similarity matrix → lowest avg sim)
     b. SIKF — 知识蒸馏 Prompt 融合 (surrogate data + KD)
     c. vit.head FedAvg — Input Enhancement 分类头联邦平均

  3. 分发全局模型给客户端:
     - global_protos[class] → 客户端 L_cons 对比学习 (InfoNCE)
     - global_head (vit.head FedAvg 结果) → 仅存储/分发，不 load 到 vit 也不 load 到 model.head
     - fused_prompt → 客户端 Prompt 初始化
```

**不在服务器通信中的**:
- ❌ Tail Anchor model.head 不做 FedAvg
- ❌ Tail Anchor key pool 不做 FedAvg
- ❌ Tail Anchor anchor pool 不做 FedAvg

### 1.4 原始 Tail Anchor (Hard Selection)

```python
similarity = x_norm @ key_norm.T         # (B, nb_class)
_, index = torch.topk(similarity, k=1)   # Hard top-1 key
anchor_feat = anchor_pool[index]          # (B, 768)
feat_mixed = concat(x, anchor_feat)       # (B, 1536)
logits = classification_head(feat_mixed)
```

### 1.5 损失函数 (FedTA 原始)

```
L = L_CE + λ_spatial·L_pull_off + λ_cons·L_cons
```

---

## 二、FedSMR 改进方案

### 2.1 改进原则

> **服务器通信和分类头机制保持不变，只针对 Tail Anchor 的 hard retrieval 和持续任务下的 memory structural drift 进行改进。**

所有 FedSMR 创新均在客户端本地 Tail Anchor memory (Key Pool + Anchor Pool) 上进行，不引入新的服务器通信。

### 2.2 改进总览

| 改进 | 位置 | 层级 | 解决的问题 |
|------|------|------|-----------|
| Residual Soft-Anchor | Tail_Anchor.forward | **Core** | 保持类别判别性 + 平滑梯度 |
| L_route 路由损失 | Client_DF.train | **Enhancement** | Key 路由正确性 |
| Seen-Only Diversity | Tail_Anchor + Client_DF | **Core (MSP)** | Anchor 坍缩防护 |
| Key+Anchor Temporal | Client_DF.train | **Core (MSP)** | 跨任务记忆结构漂移 |

### 2.3 Core A: Residual Soft-Anchor

```
a_final = (1-γ)·a_hard + γ·a_soft

其中:
  hard_idx = argmax(similarity)
  a_hard = anchor_pool[hard_idx]              ← 保留 FedTA 原始 hard anchor
  a_soft = softmax(similarity/τ) @ anchor_pool ← 新增 soft semantic mixture
  γ = soft_anchor_ratio
```

**实现**: `Tail_Anchor.py:forward()` (L98-126)

**设计动机**: 
- Hard anchor 保持强类别指向性（原 FedTA 优势）
- Soft branch 提供平滑梯度，缓解 hard argmax 的梯度断裂
- 残差结构确保当 γ=0 时完全等价于原 FedTA

### 2.4 Core B: Memory Structure Preservation (MSP)

MSP 由两个子损失组成，均仅作用于客户端本地 Tail Anchor memory:

#### (a) Seen-Only Diversity

```
seen_anchors = anchor_pool[seen_class_mask]
sim = normalize(seen_anchors) @ normalize(seen_anchors).T
L_div = max(0, sim[i≠j] - margin).mean()
```

- 只约束已见类别 (seen_class_mask)，避免未训练 anchor 被无意义推开
- margin=0.2 允许一定相关性，只防止高度坍缩

**实现**: `Tail_Anchor.anchor_diversity_loss()` (L152-166)

#### (b) Key + Anchor Temporal Stability

```
L_temporal = L_anchor_tmp + η·L_key_tmp

L_anchor_tmp = mean(conf_c · (1-cos(A_c^t, A_c^{t-1})))  对 old_seen_classes
L_key_tmp    = mean(conf_c · (1-cos(K_c^t, K_c^{t-1})))   对 old_seen_classes

conf_c = hard_usage_c / max(hard_usage)  基于 argmax 的使用频率加权
η = key_temporal_ratio
```

- 同时约束 Key 和 Anchor 的跨轮稳定性
- 只约束旧类（已见过但不是当前任务的类）
- 使用 hard usage (argmax-based) 置信度加权

**实现**: `Client_DF._compute_msp_losses()` (L245-304)

### 2.5 Enhancement: Supervised Routing Loss

```
L_route = CE(softmax(similarity[:, seen]/τ_route), target)
```

强制 sample feature → correct key 的语义对齐。与 Soft Anchor 配套使用。

**实现**: `Client_DF._compute_route_loss()` (L306-329)

---

## 三、完整损失函数 (FedSMR)

```
L_total = L_CE
        + 0.2·task_per_epoch·L_cons         # FedTA 原始: global-prototype contrastive loss (InfoNCE)
        - 0.1·L_pull_off                     # FedTA 原始: 拉约束
        + λ_route·L_route                    # FedSMR enhancement: 监督路由
        + α_div·L_div(seen, margin)          # FedSMR core: Seen-Only Diversity
        + α_tmp·(L_anchor_tmp + η·L_key_tmp) # FedSMR core: Key+Anchor Temporal
```

注: FedTA 论文中 L_cons (Eq.5-6) 是客户端使用 BGPS 全局原型进行的对比学习，
与 SIKF (Eq.7-8) 是服务器代理数据上的知识蒸馏 (L_KD)，两者是不同的机制。

---

## 四、消融实验设计

| 实验 | Anchor | Route | MSP | 说明 |
|------|:---:|:---:|:---:|------|
| E0 | Hard | | | FedTA 基线 |
| E1 | Residual Soft | | | + Soft-Anchor |
| E2 | Residual Soft | ✓ | | + Route Loss |
| E3 (FedSMR) | Residual Soft | ✓ | ✓ | + MSP (完整 FedSMR) |

**预期关键差值**:
- E1−E0: Soft-Anchor 带来的梯度平滑收益
- E2−E1: 路由损失消除 Soft-Anchor routing ambiguity
- E3−E2: MSP 提升跨任务记忆稳定性

---

## 五、代码结构

```
FedSTAR/
├── main.py                          — 入口：参数解析、数据集准备、模型初始化
├── Models/
│   ├── Tail_Anchor.py               — Residual Soft-Anchor, Seen-Only Diversity, hard usage
│   ├── Client_DF.py                 — 本地训练、Route Loss、MSP Temporal、per-task head 管理
│   ├── Server_DF.py                 — SIKF、BGPS、vit.head FedAvg、训练编排
│   ├── classification_head.py       — Chead: 两层 MLP 分类头
│   ├── vision_transformer.py        — ViT 基础模型
│   └── vision_transformer__l2p.py   — L2P Prompt Pool 实现
├── config/
│   ├── cifar100_delay.py            — CIFAR-100 参数配置
│   └── imagenet_r_delay.py          — ImageNet-R 参数配置
├── data/                            — 数据集处理
├── utils.py                         — 工具函数 (accuracy, distillation loss, cosine classifier)
├── RUN_COMMANDS.md                  — 实验运行命令
└── PROJECT_ARCHITECTURE.md          — 本文件
```

---

## 六、与原始 FedTA 代码的对齐清单

| 组件 | 原始 FedTA | 当前实现 | 状态 |
|------|-----------|---------|------|
| `fed_avg_head()` 聚合对象 | `vit.head` | `vit.head` | ✅ 已对齐 |
| `model.head` 管理 | per-task `heads[task]` 快照 | per-task `heads[task]` 快照 | ✅ 已对齐 |
| `evaluate()` 使用 | `model.load_head(heads[task])` | `model.load_head(heads[task])` | ✅ 已对齐 |
| `global_head` 分发 | 仅存储/分发，不 load 到 vit 或 model.head | 仅存储/分发，不 load 到 vit 或 model.head | ✅ 已对齐 |
| Tail Anchor 全模型聚合 | 不做 | 不做 | ✅ 已对齐 |
| SIKF (prompt KD fusion) | ✅ | ✅ | ✅ 保留 |
| BGPS (prototype selection, threshold=0.25) | ✅ | ✅ | ✅ 已对齐 |
| `choose_best_proto_greedy_similarity_fixed_key()` | ✅ | ✅ | ✅ 保留 |
| `kd_fusion_prompt()` | ✅ | ✅ | ✅ 保留 |

---

## 七、与官方代码的已知差异（有意修正）

### Chead 输出维度

| | 官方 FedTA | 本实现 | 影响 |
|---|---|---|---|
| `Chead(nb_class)` | 硬编码 `Chead(200)` | `Chead(args.nb_classes)` | CIFAR-100: 200→100; ImageNet-R: 200 (无变化) |

**原因**: 官方代码对所有数据集固定输出 200 类，在 CIFAR-100 (100 类) 上浪费了 100 个未使用维度。
本修正对 FedTA baseline 和 FedSMR 统一应用，属于公平的 bug fix。

### BGPS 选择阈值

| | 官方 FedTA | 本实现 |
|---|---|---|
| `threshold` | `train_clients()` 硬编码 `threshold=0.25` | config 默认 `0.25` |

**原因**: 官方 config 默认值 `0.15` 未被主训练路径实际使用（`train_clients()` 内硬编码 `0.25`）。
本实现统一使用 `0.25` 以对齐官方实际运行行为。