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

## 七、诊断框架（Checkpoint 断点恢复 + 跨任务遗忘机制诊断）

说明：

- 所有诊断参数默认关闭；不开启时 E0/E1/E2/E3 行为完全不变。
- CP1 只依赖 `--save_checkpoints`；CP2 / CP3 的保存与全部诊断依赖 `--run_tidr_diagnostics`。
- 诊断目标默认为 **Round5 / Client0 / Task1**（要求 `--global_epoch=5`，与现有实验一致）；
  可通过 `--diag_cf_round` 指定其他 CF 轮（如 Task2 传 10）。
- Checkpoint 保存至 `checkpoints/E6a_v2_diag/{dataset}_seed{seed}_{timestamp}/`，
  每次运行独立目录，不覆盖其他实验（**恢复运行也会新建目录**，CP3 保存在恢复运行的新目录中）。
- 恢复运行必须使用与原运行**完全相同的超参数**（数据 split 依赖相同 seed）。

### 单次完整诊断运行（E3 超参）

一次跑完整个训练（Round0-5），保存 CP1/CP2/CP3 并输出全部诊断。

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e3_diag \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.1 --key_temporal_ratio=0.5 \
  --run_tidr_diagnostics --save_checkpoints
```

### 分步链式运行（bash 整段复制，自动接力、无人值守）

依次执行：跑到 CP2 停止 → 从 CP2 恢复进入 Phase2 并跑到训练结束 → 从 CP3 离线诊断。
自动定位每次运行生成的 checkpoint 目录，任一步失败立即中止。

```bash
# ===== 诊断框架无人值守链式运行（bash，整段复制执行） =====
# 公共超参（E3 完整超参）
common=(
  main.py cifar100_delay
  --batch-size 16 --data-path ./local_datasets/ --data_name cifar100
  --output_dir ./output/cifar100_e3_diag
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2
  --msp_temporal_coeff=0.1 --key_temporal_ratio=0.5
)

# Step 1: 跑到 CP2（Round5 Client0 Phase2 开始前）保存并停止
python "${common[@]}" --run_tidr_diagnostics --save_checkpoints --stop_at_checkpoint R5_C0_pre_phase2 \
  || { echo "Step 1 failed (stop at CP2)" >&2; exit 1; }

# Step 2: 定位 Step 1 生成的 run 目录，从 CP2 恢复直接进入 Phase2，跑到训练结束（CP3 存到新目录）
runDir1=$(ls -td ./checkpoints/E6a_v2_diag/*/ | head -n 1)
python "${common[@]}" --run_tidr_diagnostics --save_checkpoints --resume_checkpoint "${runDir1}R5_C0_pre_phase2.pth" \
  || { echo "Step 2 failed (resume from CP2)" >&2; exit 1; }

# Step 3: 定位 Step 2 生成的新 run 目录，从 CP3 做离线诊断（不训练，应复现保存时准确率）
runDir2=$(ls -td ./checkpoints/E6a_v2_diag/*/ | head -n 1)
python "${common[@]}" --run_tidr_diagnostics --resume_checkpoint "${runDir2}R5_C0_post_phase2.pth" \
  || { echo "Step 3 failed (CP3 offline diag)" >&2; exit 1; }

echo "Diagnostic chain finished. Checkpoints: ${runDir1} , ${runDir2}"
```

### 单步命令（手动分步调试用，替换 <run_dir> 为实际目录名）

```bash
# 公共超参前缀（与上方 common 数组相同的单行展开）

# 1) 完整诊断运行（同单次完整诊断运行）
#    末尾加: --run_tidr_diagnostics --save_checkpoints

# 2) 只跑到 CP2（Round5 Client0 Phase2 开始前保存并停止）
#    末尾加: --run_tidr_diagnostics --save_checkpoints --stop_at_checkpoint R5_C0_pre_phase2

# 3) 从 CP2 恢复，直接进入 Round5 Client0 Phase2（跳过 Round0-4 与 Phase1）
#    末尾加: --run_tidr_diagnostics --save_checkpoints --resume_checkpoint checkpoints/E6a_v2_diag/<run_dir>/R5_C0_pre_phase2.pth

# 4) 从 CP1 恢复（从 Round5 开始，不重复 Round4）
#    末尾加: --run_tidr_diagnostics --save_checkpoints --resume_checkpoint checkpoints/E6a_v2_diag/<run_dir>/R4_complete.pth

# 5) 只加载 CP3 做离线诊断（不训练、不聚合；重新评估应复现保存时准确率）
#    末尾加: --run_tidr_diagnostics --resume_checkpoint checkpoints/E6a_v2_diag/<run_dir>/R5_C0_post_phase2.pth
```

### 诊断参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--save_checkpoints` | False | 保存 CP1/CP2/CP3 三个诊断 Checkpoint（CP2/CP3 需同时开启 `--run_tidr_diagnostics`） |
| `--resume_checkpoint` | `''` | 从指定 Checkpoint 恢复 |
| `--stop_at_checkpoint` | `''` | 到达指定 Checkpoint 后停止（如 `R5_C0_pre_phase2`） |
| `--run_tidr_diagnostics` | False | 开启跨任务遗忘诊断（TIDR） |
| `--diag_cf_round` | 5 | 完整反事实套件生效的轮次 |
| `--save_task_checkpoints` | False | 每个 Task 边界保存 Checkpoint（`Task{k}_start.pth` / `Task{k}_C0_post_phase1.pth` / `Task{k}_C0_post_phase2.pth` / `Task{k}_complete.pth`） |

### 恢复点说明

- `R4_complete` (CP1) → 从 Round5 开始（不重复 Round4）
- `R5_C0_pre_phase2` (CP2) → **mid-round resume**：Phase1 已完成，恢复后跳过 Phase1 直接继续 Phase2
- `R5_C0_post_phase2` (CP3) → **standalone/offline 离线诊断**（不训练，复现保存时准确率与 CF）
- `Task{k}_C0_post_phase1` → mid-round resume（Phase1 已完成，仅 `--save_task_checkpoints` 也能独立恢复）
- `Task{k}_C0_post_phase2` → 离线诊断
- `Task{k}_start` / `Task{k}_complete` → 从 Task 边界继续训练

### 诊断日志

- `[TIDR-PhaseDiag]` task0/task1（及更多任务）在 Phase2 前后的准确率与变化
- `[TIDR-StepDiag]` Phase2 优化步数 / batch 数 / 样本数 / 平均 batch 大小
- `[TIDR-FeatureDiag]` mean/max/p95_old_feature_drift（固定样本，Phase2 前后 query 对比）
- `[TIDR-Diag-Normal]` / `[TIDR-Diag-Rollback]` 完整 old-new + old-old + 样本级错误分解
- `[TIDR-KeyCF]` acc_normal / acc_rollback / recovery / key_restoration_pass
- `[TIDR-LocalDiag]` / `[TIDR-GlobalDiag]` evaluate() 中的 old-new + old-old 指标（本地/聚合后）
- `[TIDR-KeyDiag]` 追加 median / p95（仅诊断开启时）
- `[TIDR-TaskBoundaryDiag]` 任务首轮、update_data 之前，全部旧任务 `task{t}_acc_pre_task_switch`
- `[TIDR-P1Drift]` Phase1 窗口漂移（key/anchor/head 预期 0；feature 与 retrieved anchor_feat 是核心）
- `[TIDR-RetrievedAnchorDrift]` Phase2 窗口 retrieved anchor feature 的 cos drift / norm ratio / L2 drift
- `[TIDR-HeadDriftDiag]` current_train_head_change / task0_snapshot_head_change / all_old_snapshot_heads_max_change
- `[TIDR-RetrievalCF]` 四组掩码 CF：`acc_seen_only` / `acc_exclude_current` / `acc_prev_seen_only` / `acc_t0_only`
- `[TIDR-RetrievalCF-Check]` 掩码 CF 的 sanity check（allowed_count / mask_copy_pass / hard_idx_in_allowed_rate / mean_finite_routing_logits / soft_attention_allowed_mass）
- `[TIDR-AnchorRouteDiag]` cur_task_to_any_old_anchor_rate
- `[TIDR-SoftMassDiag]` group-wise soft attention mass 的 post_p1 vs post_p2 对照
- 类级 CSV：`diagnostics/E6a_v2_R{round}_C{client}_T{task}_classwise.csv`

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
