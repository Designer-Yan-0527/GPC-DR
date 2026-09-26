# FedSMR 实验运行命令

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
  ├── E3: E2 + MSP (FedSMR 完整版)
  │     Seen-Only Diversity + Key&Anchor Temporal Stability
  │
  ├── E4: E3 + Prototype Head Replay (ablation)
  │     全局原型重放保护旧类分类边界
  │
  └── E5: E4 + Seen Routing (ablation)
        Anchor 路由限制在 seen classes
```

| 实验 | Soft Anchor | Route Loss | MSP | Proto Replay | Seen Routing |
|------|:---:|:---:|:---:|:---:|:---:|
| E0 (FedTA Baseline) | | | | | |
| E1 (+SA) | ✓ | | | | |
| E2 (+Route) | ✓ | ✓ | | | |
| E3 (FedSMR) | ✓ | ✓ | ✓ | | |
| E4 (+Proto) | ✓ | ✓ | ✓ | ✓ | |
| E5 (+SeenRoute) | ✓ | ✓ | ✓ | ✓ | ✓ |
| E6a (+DiffRetrieval) | ✓ | ✓ | ✓ | | | *Diff. Retrieval |
| E6b (GPC-DR) | ✓ | PCR | ✓ | | | *Full GPC-DR |

\* E6a: adaptive_gamma + anchor_pool.detach(); E6b: + proto_calibration + GPA

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
  --use_soft_anchor=False --use_route_loss=False --use_msp=False \
  --use_proto_replay=False --use_seen_routing=False
```

### E1. + Residual Soft-Anchor

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e1 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=False --use_msp=False \
  --use_proto_replay=False --use_seen_routing=False
```

### E2. + Anchor Routing Loss

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e2 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=False \
  --use_proto_replay=False --use_seen_routing=False
```

### E3. FedSMR = SA + Route + MSP

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e3 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.1 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --use_proto_replay=False --use_seen_routing=False
```

### E4. + Prototype Head Replay (ablation)

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e4 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.1 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --use_proto_replay=True --lambda_proto=0.2 \
  --use_seen_routing=False
```

### E5. + Seen Routing (ablation)

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e5 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.1 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --use_proto_replay=True --lambda_proto=0.2 \
  --use_seen_routing=True
```

### E6a. E3 + Differentiable Retrieval (GPC-DR step 1)

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e6a \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.1 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --adaptive_gamma=False --gamma_max=0.35 \
  --use_diff_retrieval=True \
  --use_proto_replay=False --use_seen_routing=False \
  --use_proto_calibration=False --use_gpa=False
```

### E6b. Full GPC-DR (E3 + Diff. Retrieval + Proto Calibration + GPA)

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e6b \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.1 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --adaptive_gamma=True --gamma_max=0.35 \
  --use_diff_retrieval=True \
  --use_proto_calibration=True --proto_beta=0.5 --proto_temperature=0.10 \
  --lambda_pcr=0.05 \
  --use_gpa=True --lambda_gpa=0.2 \
  --use_proto_replay=False --use_seen_routing=False
```

---

## 二、ImageNet-R

### E0. 基线 — FedTA

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e0 \
  --use_soft_anchor=False --use_route_loss=False --use_msp=False \
  --use_proto_replay=False --use_seen_routing=False
```

### E1. + Residual Soft-Anchor

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e1 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=False --use_msp=False \
  --use_proto_replay=False --use_seen_routing=False
```

### E2. + Anchor Routing Loss

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e2 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=False \
  --use_proto_replay=False --use_seen_routing=False
```

### E3. FedSMR = SA + Route + MSP

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e3 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.15 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --use_proto_replay=False --use_seen_routing=False
```

### E4. + Prototype Head Replay (ablation)

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e4 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.15 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --use_proto_replay=True --lambda_proto=0.3 \
  --use_seen_routing=False
```

### E5. + Seen Routing (ablation)

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e5 \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.15 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --use_proto_replay=True --lambda_proto=0.3 \
  --use_seen_routing=True
```

### E6a. E3 + Differentiable Retrieval (GPC-DR step 1)

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e6a \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.15 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --adaptive_gamma=True --gamma_max=0.30 \
  --use_proto_replay=False --use_seen_routing=False \
  --use_proto_calibration=False --use_gpa=False
```

### E6b. Full GPC-DR (E3 + Diff. Retrieval + Proto Calibration + GPA)

```bash
python main.py imagenet_r_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name ImageNet-R --output_dir ./output/imagenet_r_e6b \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.15 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.15 --key_temporal_ratio=0.5 \
  --msp_coherence_coeff=0.0 \
  --adaptive_gamma=True --gamma_max=0.30 \
  --use_diff_retrieval=True \
  --use_proto_calibration=True --proto_beta=0.5 --proto_temperature=0.10 \
  --lambda_pcr=0.05 \
  --use_gpa=True --lambda_gpa=0.2 \
  --use_proto_replay=False --use_seen_routing=False
```

---

## 三、参数速查

| 模块 | 参数 | CIFAR-100 | ImageNet-R | 说明 |
|------|------|-----------|------------|------|
| Residual Soft-Anchor | `--use_soft_anchor` | True | True | |
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
| Coherence | `--msp_coherence_coeff` | 0.0 | 0.0 | 已禁用 |
| Proto Replay | `--use_proto_replay` | True (E4+) | True (E4+) | ablation |
| Proto 权重 | `--lambda_proto` | 0.2 | 0.3 | |
| Seen Routing | `--use_seen_routing` | True (E5) | True (E5) | ablation |
| **GPC-DR** | `--use_proto_calibration` | True (E6b) | True (E6b) | Proto-calibrated routing |
| Proto β | `--proto_beta` | 0.5 | 0.5 | 原型路由权重 |
| Proto τ | `--proto_temperature` | 0.10 | 0.10 | 原型相似度温度 |
| Adaptive γ | `--adaptive_gamma` | True (E6a+) | True (E6a+) | 置信度门控 |
| γ_max | `--gamma_max` | 0.35 | 0.30 | 自适应γ上限 |
| GPA | `--use_gpa` | True (E6b) | True (E6b) | Phase1 原型对齐 |
| GPA 权重 | `--lambda_gpa` | 0.2 | 0.2 | |
| PCR 权重 | `--lambda_pcr` | 0.05 | 0.05 | PCR 损失权重 |

---

## 四、损失函数

### Phase 1 (Prompt)
```
L_P1 = L_CE_prompt - 0.1·L_pull + λ_gpa·L_GPA     # E6b: GPA aligns prompt to global proto
```

### Phase 2 (Tail Anchor)
```
L_P2 = L_CE
     + λ_cons · L_cons                     # FedTA: global-prototype contrastive (InfoNCE)
     - 0.1 · L_pull_off                    # FedTA: 拉约束
     + λ_route·L_route (or λ_pcr·L_PCR)   # E2+/E6b: 监督路由 (PCR uses proto-calibrated logits)
     + α_div · L_div(seen, margin)         # E3+: Seen-Only Diversity
     + α_tmp · (L_anchor_tmp + η·L_key_tmp)  # E3+: Key+Anchor Temporal
     + λ_proto · L_proto_replay            # E4+: Prototype Head Replay (ablation)
```
```

---

## 五、多 seed 运行

```bash
for seed in 42 123 2024; do
  for exp in e0 e1 e2 e3 e4 e5; do
    python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
      --data_name cifar100 --output_dir ./output/cifar100_${exp}_seed${seed} \
      --seed ${seed} \
      ...  # 按上面的 E0-E5 参数填
  done
done
```

---

## 六、已移除的参数

| 参数 | 原因 |
|------|------|
| `--use_head_grad_mask` | per-task head 物理隔离，不需要梯度掩码；且该设计针对 model.head 而非 vit.head |
| `--use_class_aware_head_agg` | 原始 FedTA 聚合的是 vit.head，不需要 class-aware；model.head 是 per-task 快照 |
| `--use_fed_smr_aggregate` | 非原始 FedTA 通信协议；将 Tail Anchor 全模型聚合改变了 retrieval + memory + communication 三层 |
| `--use_soft_prompt / --temperature_anneal / --use_sparse_softmax` | 已废弃 |