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

\* E6a: Gradient-Decoupled Differentiable Retrieval only (fixed γ, no proto calib, no GPA)
\* E6b: E6a + Prototype-Calibrated Soft Routing + Adaptive Gamma + PCR + GPA

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
  --adaptive_gamma=False --gamma_max=0.35 \
  --use_diff_retrieval=True \
  --use_proto_replay=False --use_seen_routing=False \
  --use_proto_calibration=False --use_gpa=False
```

### E6a-v2. Task-Isolated Differentiable Retrieval

```bash
python main.py cifar100_delay \
  --batch-size 16 \
  --data-path ./local_datasets/ \
  --data_name cifar100 \
  --output_dir ./output/cifar100_e6a_v2 \
  --use_soft_anchor=True \
  --soft_temperature=0.17 \
  --soft_anchor_ratio=0.25 \
  --use_route_loss=True \
  --route_temperature=0.1 \
  --lambda_route=0.05 \
  --use_msp=True \
  --msp_diversity_coeff=0.03 \
  --diversity_margin=0.2 \
  --msp_temporal_coeff=0.1 \
  --key_temporal_ratio=0.5 \
  --use_diff_retrieval=True \
  --use_task_isolated_diff=True \
  --adaptive_gamma=False \
  --use_proto_replay=False \
  --use_seen_routing=False \
  --use_proto_calibration=False \
  --use_gpa=False
```

### E6a-v2-Diag. Checkpoint 断点恢复 + 跨任务遗忘机制诊断

说明：

- 所有新增参数默认关闭；不开启时 E3 / E6a / E6a-v2 行为完全不变。
- CP1 只依赖 `--save_checkpoints`；CP2 / CP3 的保存与全部诊断依赖 `--run_tidr_diagnostics`。
- 诊断目标默认为 **Round5 / Client0 / Task1**（要求 `--global_epoch=5`，与现有实验一致）；
  E6a-v3b-Diag 起可通过 `--diag_cf_round` 指定其他 CF 轮（如 Task2 传 10）。
- Checkpoint 保存至 `checkpoints/E6a_v2_diag/{dataset}_seed{seed}_{timestamp}/`，
  每次运行独立目录，不覆盖其他实验（**恢复运行也会新建目录**，CP3 保存在恢复运行的新目录中）。
- 恢复运行必须使用与原运行**完全相同的超参数**（数据 split 依赖相同 seed）。

#### 方案 A：单次完整诊断运行（最简单，一条命令无人值守）

一次跑完整个训练（Round0-5），保存 CP1/CP2/CP3 并输出全部诊断。

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e6a_v2_diag \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.1 --key_temporal_ratio=0.5 \
  --use_diff_retrieval=True --use_task_isolated_diff=True \
  --adaptive_gamma=False --use_proto_replay=False --use_seen_routing=False \
  --use_proto_calibration=False --use_gpa=False \
  --run_tidr_diagnostics --save_checkpoints
```

#### 方案 B：分步链式运行（bash 整段复制，自动接力、无人值守）

依次执行：跑到 CP2 停止 → 从 CP2 恢复进入 Phase2 并跑到训练结束 → 从 CP3 离线诊断。
自动定位每次运行生成的 checkpoint 目录，任一步失败立即中止。

```bash
# ===== E6a-v2-Diag 无人值守链式运行（bash，整段复制执行） =====
# 公共超参（E6a-v2 完整超参，与方案 A 完全一致）
common=(
  main.py cifar100_delay
  --batch-size 16 --data-path ./local_datasets/ --data_name cifar100
  --output_dir ./output/cifar100_e6a_v2_diag
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2
  --msp_temporal_coeff=0.1 --key_temporal_ratio=0.5
  --use_diff_retrieval=True --use_task_isolated_diff=True
  --adaptive_gamma=False --use_proto_replay=False --use_seen_routing=False
  --use_proto_calibration=False --use_gpa=False
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

echo "E6a-v2-Diag chain finished. Checkpoints: ${runDir1} , ${runDir2}"
```

#### 单步命令（手动分步调试用，替换 <run_dir> 为实际目录名）

```bash
# 公共超参前缀（与方案 B 中 common 数组相同的单行展开）

# 1) 完整诊断运行（同方案 A）
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

新增日志：

- `[TIDR-PhaseDiag]` task0/task1 在 Phase2 前后的准确率与变化
- `[TIDR-StepDiag]` Phase2 优化步数 / batch 数 / 样本数 / 平均 batch 大小
- `[TIDR-FeatureDiag]` mean/max/p95_old_feature_drift（固定样本，Phase2 前后 query 对比）
- `[TIDR-Diag-Normal]` / `[TIDR-Diag-Rollback]` 完整 old-new + old-old + 样本级错误分解
- `[TIDR-KeyCF]` acc_normal / acc_rollback / recovery / key_restoration_pass
- `[TIDR-LocalDiag]` / `[TIDR-GlobalDiag]` evaluate() 中的 old-new + old-old 指标（本地/聚合后）
- `[TIDR-KeyDiag]` 追加 median / p95（仅诊断开启时）
- 类级 CSV：`diagnostics/E6a_v2_R{round}_C{client}_T{task}_classwise.csv`

### E6a-v3b-Diag. Task2 崩溃定位（跨任务诊断扩展）

背景：E6a-v3a（No-Anchor-WD）解决 Task1 阶段 Anchor 范数塌缩后，Round10 / Task2 首轮
Client0 Task0 从 88.86% 崩到 39.40%（Key drift 小、old-old routing 高，形态不同于 Task1）。
本轮**只做诊断，不加任何修复**。

新增参数（均默认关闭）：

- `--diag_cf_round`（默认 5）：完整反事实套件（KeyCF/AnchorCF/NormCF/DirCF/JointCF/
  RetrievalCF + CP2/CP3 保存）生效的轮次。**Task2 诊断传 10**。
- `--save_task_checkpoints`：每个 Task 边界保存 Checkpoint（k≥1）：
  `Task{k}_start.pth` / `Task{k}_C0_post_phase1.pth` / `Task{k}_C0_post_phase2.pth` /
  `Task{k}_complete.pth`（独立于 `--run_tidr_diagnostics`，可与任何实验组合）。

诊断分级：

- **CF 轮**（`i == diag_cf_round`，需 `--run_tidr_diagnostics --save_checkpoints`）：
  完整诊断（PhaseDiag 三点 + drift + 全部 CF + CP2/CP3）。
- **其他任务首轮**（Round5/10/15/20，仅 `--run_tidr_diagnostics`）：
  轻量 PhaseDiag——Task0..k 的 `post_task_switch_pre_phase1 / post_phase1 /
  post_phase2` 三点准确率 +
  P1/P2 两窗口 drift + RetrievalCF。**这是定位 Task0 崩在 Phase1 还是 Phase2 的关键**。

恢复点新增：

- `Task_C0_post_phase1` → **mid-round resume**：Phase1 已完成，恢复后跳过
  Phase1 直接继续 Phase2（与 CP2 同协议；不依赖 `--run_tidr_diagnostics`，
  仅 `--save_task_checkpoints` 也能独立恢复）
- `Task_C0_post_phase2` → **standalone/offline 离线诊断**。注意：仅开
  `--save_task_checkpoints`（诊断关闭）时保存的轻量版 extras 不含 P1 三点
  准确率与 P1 snapshot，离线只能复跑 KeyCF/AnchorCF 等 Phase2 后 CF；
  "完整复现 P1Drift/RetrievedAnchorDrift" 需与 `--run_tidr_diagnostics`
  联合运行保存的版本
- `Task_start` / `Task_complete`（从 Task 边界继续训练）

CP1/CP2/CP3 文件名跟随
`diag_cf_round`（如 Round10 → `R9_complete.pth` / `R10_C0_pre_phase2.pth` /
`R10_C0_post_phase2.pth`；默认 Round5 时名称与旧版一致）。

#### 推荐：从原运行 CP2 恢复，复现 v3a 轨迹 + Task2 诊断（免重跑 Round0-4）

从 E6a-v3a 链式运行的 Step 1 目录（保存了 `R5_C0_pre_phase2.pth`）恢复，加上
`--anchor_no_wd=True` 即可精确复现 v3a 的 Round5-24 轨迹，同时拿到 Task2 诊断：

```bash
# common 同方案 B，另需加入: '--anchor_no_wd=True'
diagDir='checkpoints/E6a_v2_diag/<v3a_step1_run_dir>'   # 含 R5_C0_pre_phase2.pth 的目录
python "${common[@]}" --anchor_no_wd=True \
  --run_tidr_diagnostics --save_checkpoints --save_task_checkpoints \
  --diag_cf_round 10 \
  --resume_checkpoint "${diagDir}/R5_C0_pre_phase2.pth"
```

说明：

- Round5 起点恢复会自动视为 CF 轮（复现 v3a 的 Round5 诊断，可作 sanity check），
  Round10 仍是完整 CF 轮；Round15/20 为轻量 PhaseDiag。
- Round5 为 mid-round 恢复：`Task1_start.pth` 与 `Task1_C0_post_phase1.pth` 均
  不保存（Phase2-start 钩子被跳过，语义正确——`R5_C0_pre_phase2.pth` 本身即
  Task1 post-Phase1 等价 checkpoint）；`Task1_C0_post_phase2.pth` 与
  `Task1_complete.pth` 正常保存。
- 验收重点：Round10 日志中 `[TIDR-PhaseDiag]` 的
  `task0_acc_post_task_switch_pre_phase1 / task0_acc_post_phase1 / task0_acc_post_phase2`
  三个数——先确认 88.86 → 39.40 崩在 task-switch 边界、Phase1 还是 Phase2
  （注意 pre-Phase1 点位于 task switch 之后、vit 已用新任务 stage，若 88.86→pre-P1
  已大跌则崩在 Task1→Task2 边界本身，与 Phase1 优化无关），再按
  KeyCF/AnchorCF/NormCF/DirCF/JointCF/RetrievalCF 的 recovery 决定 v3b 方向。

#### 备选：全量重跑（从头，Round0-24）

```bash
python "${common[@]}" --anchor_no_wd=True \
  --run_tidr_diagnostics --save_checkpoints --save_task_checkpoints \
  --diag_cf_round 10
```

（Round5 为轻量 PhaseDiag、Round10 为完整 CF；Task1 的 CF 结论已有，无需重复。）

#### 离线反事实复跑（如需，从本运行保存的 Task2 checkpoint）

```bash
runDir=$(ls -td ./checkpoints/E6a_v2_diag/*/ | head -n 1)
# Task2 C0 Phase2 后离线诊断（复跑 CF 套件）
python "${common[@]}" --anchor_no_wd=True --run_tidr_diagnostics \
  --resume_checkpoint "${runDir}Task2_C0_post_phase2.pth"
```

新增日志：

- `[TIDR-TaskBoundaryDiag]`（任务首轮、update_data 之前，Server 端 Client0）
  `task{t}_acc_pre_task_switch`（全部旧任务 t=0..k-1）——Task{k-1} stage、
  上轮结束后的同协议基准。A(pre-switch) → B(post-switch/pre-P1) = task switch
  本身的因果贡献；B → C = Phase1；C → D = Phase2。若 T0 大跌而 T1 几乎
  不掉，支持 oldest-task-specific transition failure。注意普通 evaluate()
  的 R9 日志值（shuffle=True、无固定 seed）与 A 不可做精确差值，A 才是
  同协议基准
- `[TIDR-PhaseDiag]`（所有 task_id>0 首轮）task{k} 三点准确率
  post_task_switch_pre_phase1 / post_phase1(=pre_phase2) / post_phase2 +
  p1/p2/total change（注意 pre-Phase1 点位于 task switch 之后、vit 已用
  新任务 stage；若上轮末尾→pre-P1 已大跌，说明崩在任务边界本身而非 Phase1）
- `[TIDR-P1Drift]` Phase1 窗口漂移（key/anchor/head 预期 0；feature 与
  retrieved anchor_feat 是核心——判断 ViT/Prompt 表示漂移）。
  drift 全部使用缓存的固定输入张量（非仅固定 indices），无随机增强噪声；
  固定张量的缓存过程有 RNG save/restore 保护，不改变训练随机轨迹
  （旧 checkpoint 恢复时 indices 已存在但张量缺失会自动补缓存）
- `[TIDR-RetrievedAnchorDrift]` Phase2 窗口 retrieved anchor feature 的
  cos drift / norm ratio / L2 drift（raw Anchor 参数没坏但送进旧 head 的
  retrieval representation 可能已变）
- `[TIDR-HeadDriftDiag]` current_train_head_change（当前任务 head 正常训练
  幅度，不能解释 Task0 遗忘）/ task0_snapshot_head_change（应严格为 0，
  非 0 = 快照被篡改的 bug 信号）/ all_old_snapshot_heads_max_change
- `[TIDR-RetrievalCF]` 四组掩码（评 Task0，softmax 自然重归一化）:
  acc_seen_only（只允许 T0..T{cur}，recovery = future unseen 污染）/
  acc_exclude_current（全部类别但屏蔽当前任务类，recovery = 当前任务单独污染）/
  acc_prev_seen_only（只允许 union(class_mask[0..task_id-1])，recovery =
  当前任务 + future 联合污染）/
  acc_t0_only（只允许 Task0 anchors）——分解新任务 keys/anchors 与
  future unseen keys 的 retrieval contamination
- `[TIDR-AnchorRouteDiag]` cur_task_to_any_old_anchor_rate（当前任务样本
  hard route 到旧类 Anchor 的比例）
- **routing 指标更名/扩展**（`[TIDR-LocalDiag]`/`[TIDR-GlobalDiag]`/
  `[TIDR-Diag-Normal]` 等）：`old_to_new_collision` 更名为
  `to_current_task_collision`（只统计 T_eval→当前任务，**R9 与 R10 的该值
  不可直接比较**）；新增 `to_task{k}_collision` / `mean_vs_task{k}_margin` /
  `neg_vs_task{k}_margin_rate`（逐任务）、
  `to_any_non_eval_task_collision` / `mean_vs_any_non_eval_margin` /
  `neg_vs_any_non_eval_margin_rate`（全部已见任务类 - 被评估任务类）、
  `to_future_unseen_collision` / `mean_vs_future_unseen_margin` /
  `neg_vs_future_unseen_margin_rate`（未来任务未 seen 类——
  use_seen_routing=False 时检索可落到 future keys，这组单独拆出）与
  `to_any_non_eval_all_collision` / `mean_vs_any_non_eval_all_margin` /
  `neg_vs_any_non_eval_all_margin_rate`（全部类别 - 被评估任务类 =
  已见非评估 + future 的总和口径），用于研究 cross-task routing accumulation
- `[TIDR-SoftMassDiag]`（v3b-Diag r5，决定性诊断）Task0 评估时 group-wise
  soft attention mass 的 post_p1 vs post_p2 对照：
  `soft_mass_to_eval_task` / `soft_mass_to_task{k}`（各已见任务含当前）/
  `soft_mass_to_future_unseen` / `soft_mass_to_true_key` /
  `mean_soft_attn_entropy` / `mean_soft_attn_max_w`，以及同表 hard route
  各组 `to_eval_task_collision` / `to_task{k}_collision` /
  `to_future_unseen_collision`。格式 `{key}_post_p1=… {key}_post_p2=…`
  `{key}_p2_change=±…`。若 future soft mass 在 P2 后大幅上升，
  Failure Mode II（Phase2 retrieval geometry 崩坏）即钉死
- `[TIDR-RetrievalCF-Check]`（v3b-Diag r5）每次掩码 CF 评估的 sanity check：
  `allowed_count` / `allowed_digest`（允许类别索引 md5 短摘要，四组应不同）/
  `mask_copy_pass` / `hard_idx_in_allowed_rate`（应 =1.0000）/
  `mean_finite_routing_logits`（应 = allowed_count）/
  `soft_attention_allowed_mass`（应 ≈1.0000）。任一项不符 = 掩码未按预期
  生效，`[TIDR-RetrievalCF]` 的 acc 不可用

#### E6a-v3b B-实验：R10 同起点最小因果干预（B0/B1/B2/B3）

所有组从**同一个** `R10_C0_pre_phase2.pth`（Task0=82.34 / Task1=96.34 /
Task2=2.64 起点）恢复，只跑 C0 Phase2 + 完整诊断后立即停止
（`--stop_at_checkpoint R5_C0_post_phase2` 在 phase2_end 诊断与 CP3 保存后
触发 DiagStopException 干净退出，不跑 C1-C4、不聚合、不进 Round11）。

```bash
runDir=./checkpoints/E6a_v2_diag/cifar100_seed42_20260928_104012
# B0 基线（当前 No-WD 原样；预期复现 T0=39.40）
python "${common[@]}" --anchor_no_wd=True \
  --run_tidr_diagnostics --save_checkpoints --diag_cf_round 10 \
  --resume_checkpoint "${runDir}/R10_C0_pre_phase2.pth" \
  --stop_at_checkpoint R5_C0_post_phase2

# B1 Seen-only retrieval（只允许 T0+T1+T2 参与 Phase2 训练期检索）
python "${common[@]}" --anchor_no_wd=True --p2_seen_only=True \
  --run_tidr_diagnostics --save_checkpoints --diag_cf_round 10 \
  --resume_checkpoint "${runDir}/R10_C0_pre_phase2.pth" \
  --stop_at_checkpoint R5_C0_post_phase2

# B2 old-Key freeze（每步 optimizer.step() 后恢复旧 Key 行快照）
python "${common[@]}" --anchor_no_wd=True --p2_freeze_old_key=True \
  --run_tidr_diagnostics --save_checkpoints --diag_cf_round 10 \
  --resume_checkpoint "${runDir}/R10_C0_pre_phase2.pth" \
  --stop_at_checkpoint R5_C0_post_phase2

# B3 Seen-only + freeze old Key
python "${common[@]}" --anchor_no_wd=True --p2_seen_only=True --p2_freeze_old_key=True \
  --run_tidr_diagnostics --save_checkpoints --diag_cf_round 10 \
  --resume_checkpoint "${runDir}/R10_C0_pre_phase2.pth" \
  --stop_at_checkpoint R5_C0_post_phase2
```

判定参考 —— 注意 B1 是 **training-time 干预**：Phase2 训练时 seen-only ON，
结束后恢复 Normal routing，post-P2 诊断在 Normal（all-keys）协议下评估。
因此 B0/B1 各自的 normal 与 `acc_seen_only`（RetrievalCF）天然组成
**2×2 counterfactual**（训练端 × 推理端候选空间）：

| Phase2 训练 | 推理时候选空间 | 对应结果                              |
| ----------- | -------------- | ------------------------------------- |
| all keys    | all keys       | **B0 normal**                        |
| all keys    | seen-only      | **B0 的 `acc_seen_only` RetrievalCF** |
| seen-only   | all keys       | **B1 normal**                        |
| seen-only   | seen-only      | **B1 的 `acc_seen_only` RetrievalCF** |

三组因果分解（均相对 B0 normal，即 Task0 post-P2 = 39.40 起点）：

- `B0 acc_seen_only − B0 normal` 大幅为正
  → **inference-time future-key competition 是主因**（不需要重新训练，
  仅屏蔽推理候选空间即可恢复）
- `B1 normal − B0 normal` 大幅为正
  → future keys 在 **Phase2 训练过程中**也造成参数学习污染
  （训练时不在候选空间 → old key/anchor/head 不被 future 竞争挤压）
- `B1 acc_seen_only` 四格最高
  → train + inference 两端都应 seen-only（v3b 端到端方案）
- B2 ≈ +5~10pp（KeyCF 交叉验证 +7.61）：与现有诊断一致，非主修复
- B3 vs B1/B2 的叠加关系决定 v3b 是否需要双修（Seen-only + Key 保护）

结论映射更新：**"v3b 核心 = 不允许未训练 keys 参与 retrieval"需要由
上面第一/第三格共同支持**——B1 normal 单独走强只证明训练端屏蔽的作用，
推理端是否也要屏蔽由 B0 的 `acc_seen_only` 与 B1 的 `acc_seen_only` 决定。

**执行顺序：先只跑 B0。** B0 一次验证三件事：① 复现 Task0 post-P2
= 39.40；② RetrievalCF sanity check 四项全部通过
（`mask_copy_pass=True`、`hard_idx_in_allowed_rate=1.0000`、
`mean_finite_routing_logits=allowed_count`、
`soft_attention_allowed_mass≈1.0000`）；③ SoftMassDiag post-P1→post-P2
变化合理。若 B0 的 `acc_seen_only` 从 39.40 大幅恢复且 sanity check
全过，则在 B1 开始前即可直接证明 future-unseen keys 在推理候选空间中的
存在本身是 Failure Mode II 的重要因果因素。B0 通过后再连跑 B1/B2/B3。

干预自检日志：`[E6a-v3b-B1]`（allowed=N/100 + 结束后恢复确认）、
`[E6a-v3b-B2]`（n_old_keys + `old_key_bitwise_preserved=True` 必须成立）。
诊断钩子（phase2_start 的 pre-P2 基线、phase2_end 的全部 post-P2 诊断）
均在 Normal 协议（无掩码、原 routing 配置）下运行，四组之间可比。

### E6b. Full GPC-DR (E3 + Diff. Retrieval + Proto Calibration + GPA)

```bash
python main.py cifar100_delay --batch-size 16 --data-path ./local_datasets/ \
  --data_name cifar100 --output_dir ./output/cifar100_e6b \
  --use_soft_anchor=True --soft_temperature=0.17 --soft_anchor_ratio=0.25 \
  --use_route_loss=True --route_temperature=0.1 --lambda_route=0.05 \
  --use_msp=True --msp_diversity_coeff=0.03 --diversity_margin=0.2 \
  --msp_temporal_coeff=0.1 --key_temporal_ratio=0.5 \
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
  --use_diff_retrieval=True \
  --adaptive_gamma=False \
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
  --use_diff_retrieval=True \
  --adaptive_gamma=True --gamma_max=0.30 \
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
| Proto Replay | `--use_proto_replay` | True (E4+) | True (E4+) | ablation |
| Proto 权重 | `--lambda_proto` | 0.2 | 0.3 | |
| Seen Routing | `--use_seen_routing` | True (E5) | True (E5) | ablation |
| **GPC-DR** | `--use_proto_calibration` | True (E6b) | True (E6b) | Proto-calibrated routing |
| Proto β | `--proto_beta` | 0.5 | 0.5 | 原型路由权重 |
| Proto τ | `--proto_temperature` | 0.10 | 0.10 | 原型相似度温度 |
| Adaptive γ | `--adaptive_gamma` | True (E6b only) | True (E6b only) | 置信度门控 |
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
| `--use_soft_prompt / --temperature_anneal / --use_sparse_softmax / --top_k_anchor` | 已废弃方案：Soft Prompt Retrieval / 温度退火 / 稀疏 Softmax，实现代码已一并删除（hard top-k 为唯一路径） |
| `--use_prompt_mask` | 已废弃方案：task-specific prompt mask，从未在任何实验中启用 |
| `--msp_coherence_coeff` | Coherence loss 实现已删，参数残留，一并清理 |
| `--train_mask / --task_inc / --initializer / --global_pool` | 零代码引用的死参数 |
| `--shared_prompt_pool / --shared_prompt_key / --predefined_key / --pull_constraint / --pull_constraint_coeff` | 零代码引用的死参数 |
| `--use_head_grad_mask` | per-task head 物理隔离，不需要梯度掩码；且该设计针对 model.head 而非 vit.head |
| `--use_class_aware_head_agg` | 原始 FedTA 聚合的是 vit.head，不需要 class-aware；model.head 是 per-task 快照 |
| `--use_fed_smr_aggregate` | 非原始 FedTA 通信协议；将 Tail Anchor 全模型聚合改变了 retrieval + memory + communication 三层 |
