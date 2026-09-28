# FedSTAR 实验运行命令

> 架构：ViT-B/16 (frozen) + L2P Prompt + Tail_Anchor + per-task Head
>
> 双 head 设计（对齐原始 FedTA）：
> - `vit.head` — Input Enhancement 阶段分类头，服务器 FedAvg 聚合
> - `model.head` — Tail Anchor 阶段分类头，per-task 本地快照，不做服务器聚合
>
> 服务器通信仅包括：SIKF (prompt fusion) + BGPS (prototype selection) + vit.head FedAvg

---

## 实验方案总览

```
基线: FedTA（Hard Anchor + per-task Head）
  │
  ├── E1: + Residual Soft-Anchor (γ=0.25)
  │     a_mix = (1-γ)·a_hard + γ·a_soft
  │
  ├── E2: E1 + Anchor Routing Loss (L_route)
  │     监督 Key 路由到正确类别
  │
  └── E3: E2 + MSP (FedSMR 完整版)
        Seen-Only Diversity + Key&Anchor Temporal Stability
```

| 实验 | Soft Anchor | Route Loss | MSP | 说明 |
|------|:---:|:---:|:---:|------|
| E0 (FedTA Baseline) | | | | FedTA 基线 |
| E1 (+SA) | ✓ | | | + Soft-Anchor |
| E2 (+Route) | ✓ | ✓ | | + Route Loss |
| E3 (FedSMR) | ✓ | ✓ | ✓ | + MSP (完整 FedSMR) |
| SA+MSP (Route OFF) | ✓ | | ✓ | **关键消融**：剥离 Route，验证 MSP 独立贡献 |

---

## 通用参数

| 参数 | CIFAR-100 | ImageNet-R | 说明 |
|------|-----------|------------|------|
| `--batch-size` | 16 | 16 | |
| `--client_num` | 5 | 5 | |
| `--task_num` | 5 | 5 | |
| `--private_class_num` | 15 | 40 | |
| `--global_epoch` | 5 | 5 | |
| `--local_epoch` | 30 | 30 | |
| `--surrogate_num` | 20 | 5 | 服务器代理数据每类采样数 |
| `--threshold` | 0.25 | 0.25 | BGPS 原型选择阈值（对齐官方） |
| `--seed` | 42 | 42 | 建议 3 seeds: 42, 123, 2024 |

---

## 一、CIFAR-100

### E0. 基线 — FedTA (Hard Anchor + per-task Head)

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e0 \
  --use_soft_anchor=False --use_route_loss=False --use_msp=False
```

### E1. + Residual Soft-Anchor

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e1 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=False --use_msp=False
```

### E2. + Anchor Routing Loss

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e2 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=False
```

### E3. FedSMR = SA + Route + MSP

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e3 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.1 --key_temporal_ratio=0.5
```

### 关键消融：SA + MSP（Route OFF）

剥离 Route Loss，验证 MSP 的独立贡献（区分"MSP 本身有效" vs "MSP 只是修复 Route 损伤"）。
MSP 参数与 E3 完全一致，仅 `--use_route_loss=False`。

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_sa_msp \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=False \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.1 --key_temporal_ratio=0.5 \
  --seed 42
```

---

## 二、ImageNet-R

### E0. 基线 — FedTA

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e0 \
  --use_soft_anchor=False --use_route_loss=False --use_msp=False
```

### E1. + Residual Soft-Anchor

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e1 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=False --use_msp=False
```

### E2. + Anchor Routing Loss

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e2 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=False
```

### E3. FedSMR = SA + Route + MSP

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e3 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.15 --key_temporal_ratio=0.5
```

### 关键消融：SA + MSP（Route OFF）

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_sa_msp \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=False \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.15 --key_temporal_ratio=0.5 \
  --seed 42
```

---

## 三、参数速查

| 模块 | 参数 | CIFAR-100 | ImageNet-R | 说明 |
|------|------|-----------|------------|------|
| Residual Soft-Anchor | `--use_soft_anchor` | True (E1+) | True (E1+) | |
| 温度 | `--soft_temperature` | 0.17 | 0.17 | Softmax τ |
| 残差比 | `--soft_anchor_ratio` | 0.25 | 0.15 | γ |
| 路由损失 | `--use_route_loss` | True (E2+) | True (E2+) | L_route |
| 路由温度 | `--route_temperature` | 0.1 | 0.1 | |
| 路由权重 | `--lambda_route` | 0.05 | 0.05 | |
| MSP | `--use_msp` | True (E3+) | True (E3+) | |
| Diversity | `--msp_diversity_coeff` | 0.03 | 0.03 | α_div |
| Diversity Margin | `--diversity_margin` | 0.2 | 0.2 | |
| Temporal | `--msp_temporal_coeff` | 0.1 | 0.15 | α_tmp |
| Key 权重 | `--key_temporal_ratio` | 0.5 | 0.5 | η |

---

## 四、损失函数

### Phase 1 (Prompt)
```
L_P1 = L_CE_prompt - 0.1·L_pull
```

### Phase 2 (Tail Anchor)
```
L_P2 = L_CE
     + λ_cons · L_cons                     # FedTA: global-prototype contrastive (InfoNCE)
     - 0.1 · L_pull_off                    # FedTA: 拉约束
     + λ_route·L_route                      # E2+: 监督路由
     + α_div · L_div(seen, margin)          # E3+: Seen-Only Diversity
     + α_tmp · (L_anchor_tmp + η·L_key_tmp) # E3+: Key+Anchor Temporal
```

---

## 五、多 seed 运行

```bash
for seed in 42 123 2024; do
  for exp in e0 e1 e2 e3; do
    python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
      --data_name cifar100 --output_dir ./output/cifar100_${exp}_seed${seed} \
      --seed ${seed} \
      ...  # 按上面的 E0-E3 参数填
  done
done
```

---

## 六、CL 评估协议（Continual-Learning Accuracy Matrix）

每个 Task 最后一轮（`global_epoch` 末轮）聚合分发完成后，每个 client 自动评估**所有 seen tasks**（T0..T_cur），构建标准 CL accuracy matrix `R[i][j]`（i = 训练到第几个 task，j = 被评估 task）。其余轮次保持原行为（Task0 + 当前 task），避免每轮 5x 评估开销。

### 输出

1. **`logs/*_cl_matrix.csv`**（与主日志 CSV 同名，后缀 `_cl_matrix`），每行格式：

   ```
   after_task, client_id, task0, task1, ..., task{N-1}
   after_task0, 0, 91.02, , , ,
   after_task1, 0, 90.11, 93.58, , ,
   ...
   after_task4, 0, 90.57, 92.10, 93.50, 94.20, 94.62
   ```

   （未评估的 task 留空；每个 after_task × client 一行）

2. **训练结束终端输出**：

   ```
   [CL-Matrix] after_task=4 client=0 task0=90.57 task1=... task4=...
   ...
   ===== CL Metrics (Accuracy Matrix) =====
   [CL-Metrics] client=0 ACC=93.00 F=2.10 BWT=-1.50
   [CL-Metrics] MEAN ACC=... F=... BWT=...
   ```

### 指标定义

- **ACC**（Final Average Accuracy）= `mean_j R[N-1][j]`：学完全部任务后对所有任务的平均准确率
- **F**（Forgetting）= `mean_{j<N-1} (max_i R[i][j] - R[N-1][j])`：旧任务历史最高与最终值之差
- **BWT**（Backward Transfer）= `mean_{j<N-1} (R[N-1][j] - R[j][j])`：最终值与刚学完时之差（通常为负）

注意：旧实验 CSV 中的"T1–T4 endpoint avg"是刚学完各 task 时的拼接 proxy，**不是**标准 ACC，论文中不可混用；新协议下直接取 matrix 末行即可。

---

## 八、已移除的参数

| 参数 | 原因 |
|------|------|
| `--use_soft_prompt / --temperature_anneal / --use_sparse_softmax / --top_k_anchor` | 已废弃方案：Soft Prompt Retrieval / 温度退火 / 稀疏 Softmax，实现代码已一并删除（hard top-k 为唯一路径） |
| `--use_prompt_mask` | 已废弃方案：task-specific prompt mask，从未在任何实验中启用 |
| `--msp_coherence_coeff` | Coherence loss 实现已删，参数残留，一并清理 |
| `--train_mask / --task_inc / --initializer / --global_pool` | 零代码引用的死参数 |
| `--shared_prompt_pool / --shared_prompt_key / --predefined_key / --pull_constraint / --pull_constraint_coeff` | 零代码引用的死参数 |
| `--use_head_grad_mask` | per-task head 物理隔离，不需要梯度掩码；且该设计针对 model.head 而非 vit.head |
| `--use_class_aware_head_agg` | 原始 FedTA 聚合的是 vit.head，不需要 class-aware；model.head 是 per-task 快照 |
| `--use_fed_smr_aggregate` | 非原始 FedTA 通信协议；将 Tail Anchor 全模型聚合改变了 retrieval + memory + communication 三层 |
| `--use_proto_replay / --lambda_proto` | E4 Prototype Head Replay 实验代码已删除 |
| `--use_seen_routing` | E5 Seen Routing 实验代码已删除（模型内部 `use_seen_routing` 属性保留，供 RetrievalCF 诊断与 MSP diversity 临时使用） |
| `--use_proto_calibration / --proto_beta / --proto_temperature / --lambda_pcr` | E6 GPC-DR Proto Calibration / PCR 实验代码已删除 |
| `--use_diff_retrieval / --use_task_isolated_diff` | E6 Differentiable Retrieval / TIDR 实验代码已删除 |
| `--adaptive_gamma / --gamma_max` | E6 Adaptive Gamma 实验代码已删除 |
| `--use_gpa / --lambda_gpa` | E6 Phase1 Global Prototype Alignment 实验代码已删除 |
| `--anchor_no_wd` | E6a-v3a No-Anchor-WD 诊断实验代码已删除 |
| `--p2_seen_only / --p2_freeze_old_key / --p2_freeze_future_key` | E6a-v3b B-实验代码已删除 |
| `--use_hs_sg` | E6a-v3b HS-SG 算法候选代码已删除 |
