"""
FedSMR 框架客户端模块

该模块实现了 FedSMR（Federated Soft Memory Retrieval）框架的客户端类。
每个客户端维护自己的模型、训练数据，并执行本地训练和评估。

核心改进（FedSMR）：
- Unified Soft Memory Retrieval: Prompt 和 Anchor 均使用 softmax 加权检索
- Memory Structure Preservation (MSP): 三层正则化防止记忆坍缩
  - Intra-Pool Diversity: 保持 pool 内向量多样性
  - Cross-Pool Coherence: 保持 prompt 和 anchor 记忆空间结构一致
  - Temporal Stability: 防止记忆在联邦轮次间剧烈变化
- Temperature Annealing: 温度从高到低退火
- Usage Tracking: 追踪记忆使用频率供联邦聚合

核心功能：
- 基于提示学习的本地训练
- 多种评估方法（标准评估、余弦相似度、仅提示、仅分类头）
- InfoNCE 对比学习支持
- 准确率指标日志系统
"""

import csv
import hashlib
import os
import random
from copy import deepcopy
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split, Subset
from torch.nn import functional as F
from tqdm import tqdm

from Models.Tail_Anchor import Tail_Anchor
from Models.classification_head import Chead
from utils import CosineSimilarityClassifier
from checkpoint_utils import (
    save_checkpoint,
    DiagStopException,
    get_rng_state,
    set_rng_state,
)

# E6a-v3b-Diag r3: 诊断评估固定 seed 基准（_diag_eval_task 用
# TIDR_DIAG_EVAL_SEED + task 设置 torch/numpy/python/CUDA RNG，
# 保证同一 task 的多次诊断评估——post_task_switch_pre_phase1 /
# post_phase1 / post_phase2 / 各 CF——看到完全相同的增强实例；
# 评估结束后由 finally 中的 set_rng_state 恢复真实训练 RNG）
TIDR_DIAG_EVAL_SEED = 20260928


class Client_DF:
    """
    FedSMR 联邦客户端类

    每个客户端处理：
    - 本地数据管理和分割
    - 基于 Soft Memory Retrieval 的训练
    - 多种评估策略
    - 训练指标日志
    """

    def __init__(self, client_id, original_model, model_name, task_per_global_epoch,
                 subset, local_epoch, batch_size, lr, device, method, class_mask, args, vit, log_file=None):
        """
        初始化客户端实例（FedSMR 版本）

        Args:
            client_id: 客户端唯一标识符
            original_model: 预训练基础模型（如 ViT）
            model_name: 本地模型架构名称
            task_per_global_epoch: 每个全局训练轮次的任务数量
            subset: 该客户端的训练数据子集
            local_epoch: 本地训练轮次数量
            batch_size: 训练批次大小
            lr: 学习率
            device: 训练设备（CPU/GPU）
            method: 训练方法标识符
            class_mask: 每个任务的类别索引
            args: 命令行参数
            vit: 支持提示的 Vision Transformer 模型
        """
        self.id = client_id
        self.original_model = original_model
        self.vit = vit

        self.task_id = -1
        self.task_per_global_epoch = task_per_global_epoch
        self.test_loader = []
        self.train_data = subset
        self.local_epoch = local_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.method = method
        self.nb_classes = args.nb_classes

        # ---- FedSMR-v2 超参数 -----------------------------------------------
        # Residual Soft-Anchor
        self.use_soft_anchor = getattr(args, 'use_soft_anchor', False)
        self.soft_temperature = getattr(args, 'soft_temperature', 0.17)
        self.soft_anchor_ratio = getattr(args, 'soft_anchor_ratio', 0.25)
        # 路由损失
        self.use_route_loss = getattr(args, 'use_route_loss', False)
        self.route_temperature = getattr(args, 'route_temperature', 0.1)
        self.lambda_route = getattr(args, 'lambda_route', 0.05)
        # MSP v2
        self.use_msp = getattr(args, 'use_msp', False)
        self.msp_diversity_coeff = getattr(args, 'msp_diversity_coeff', 0.03)
        self.diversity_margin = getattr(args, 'diversity_margin', 0.2)
        self.msp_temporal_coeff = getattr(args, 'msp_temporal_coeff', 0.1)
        self.key_temporal_ratio = getattr(args, 'key_temporal_ratio', 0.5)
        # Proto Replay
        self.use_proto_replay = getattr(args, 'use_proto_replay', False)
        self.lambda_proto = getattr(args, 'lambda_proto', 0.2)
        self.use_seen_routing = getattr(args, 'use_seen_routing', False)
        # GPC-DR
        self.use_proto_calibration = getattr(args, 'use_proto_calibration', False)
        self.proto_beta = getattr(args, 'proto_beta', 0.5)
        self.proto_temperature = getattr(args, 'proto_temperature', 0.10)
        self.adaptive_gamma = getattr(args, 'adaptive_gamma', False)
        self.gamma_max = getattr(args, 'gamma_max', 0.35)
        self.use_diff_retrieval = getattr(args, 'use_diff_retrieval', False)
        self.use_task_isolated_diff = getattr(args, 'use_task_isolated_diff', False)
        self.use_gpa = getattr(args, 'use_gpa', False)
        self.anchor_no_wd = getattr(args, 'anchor_no_wd', False)
        # E6a-v3b B-实验（最小因果干预，默认关闭 = B0 原样）:
        #   p2_seen_only:     B1/B3 — Phase2 训练期 Seen-only retrieval
        #                     （候选空间限制为 T0..T_cur，屏蔽 future keys）
        #   p2_freeze_old_key: B2/B3 — Phase2 训练期 old-Key freeze
        #                     （每步 optimizer.step() 后恢复旧 Key 行快照）
        self.p2_seen_only = getattr(args, 'p2_seen_only', False)
        self.p2_freeze_old_key = getattr(args, 'p2_freeze_old_key', False)
        self.lambda_gpa = getattr(args, 'lambda_gpa', 0.2)
        self.lambda_pcr = getattr(args, 'lambda_pcr', 0.05)
        # -------------------------------------------------------------------

        self.model = self._init_local_model(model_name)

        # 存储上一轮记忆状态（用于 Key+Anchor temporal stability）
        self.prev_anchor_pool = None
        self.prev_key_pool = None

        # 已见类别追踪
        self.seen_classes = set()
        self.old_seen_classes = set()

        # E1-Repair v1: per-task hard anchor usage tracking (诊断用)
        self.task_anchor_usage = {}

        # 自适应: 是否有公共类
        total_private = args.client_num * args.private_class_num
        self._has_public_classes = (max(0, args.nb_classes - total_private) > 0)

        # Initialize class mask based on dataset type
        if args.data_name in ['cifar100', '5datasets', 'ImageNet-R', 'svhn-mnist']:
            self.class_mask = class_mask
        else:
            self.class_mask = []

        # Initialize model components
        self.local_protos = None
        self.global_protos = None
        self.prompts = None
        self.heads = [None] * 10  # Classifier heads for up to 10 tasks
        self.vit_heads = [None] * 10
        self.head = Chead(args.nb_classes)

        # Initialize logging system
        if log_file is not None:
            self.log_file = log_file
        else:
            self.log_file = self._init_log_file()
        self.round = 0

        # ---- E6a-v2-Diag: 诊断开关与状态（默认全部关闭，不影响训练） ----
        self.run_tidr_diagnostics = getattr(args, 'run_tidr_diagnostics', False)
        self.save_checkpoints = getattr(args, 'save_checkpoints', False)
        self._tidr_diag_config = None      # 由 Server_DF 在诊断轮注入
        self._resume_phase2_ctx = None     # CP2 恢复: 直接进入 Phase2
        self._task_ckpt_config = None      # E6a-v3b-Diag: Task 边界 Checkpoint（Server 注入）
        self._diag_key_before_phase2 = None
        self._diag_key_after_phase2 = None
        self._diag_anchor_before_phase2 = None
        self._diag_old_key_mask = None
        self._diag_feature_before = None
        self._diag_feature_after = None
        self._diag_feature_samples = None
        self._diag_phase_acc_pre = {}
        self._diag_phase_stats_pre = None
        self._diag_step_stats = None
        self._diag_task1_old_route_hits = 0
        self._diag_task1_old_route_total = 0
        # ---- E6a-v3b-Diag: Phase1 窗口快照（pre_phase1 点） ----
        self._diag_phase_acc_pre_p1 = {}          # {task k: acc_pre_phase1}
        self._diag_key_pre_p1 = None              # key 快照（Phase1 前）
        self._diag_anchor_pre_p1 = None           # anchor_pool 快照（Phase1 前）
        self._diag_head_pre_p1 = None             # model.head state_dict 快照
        self._diag_heads_pre_p1 = None            # heads[t<=task_id] state_dict 快照
        self._diag_feature_pre_p1 = None          # Task0 样本特征快照（Phase1 前）
        self._diag_anchor_feat_pre_p1 = None      # Task0 样本 retrieved anchor_feat 快照
        self._diag_anchor_feat_before_phase2 = None  # post-P1 点 retrieved anchor_feat
        # ---- E6a-v3b-Diag: 固定实际输入张量（消除随机增强噪声） ----
        # 只固定 indices 时，若 dataset 带随机 crop/flip，pre/post 提取的
        # 输入张量不同，drift 会混入 augmentation noise。
        # 因此 pre_phase1 时缓存这批样本的 input/target 张量本身，
        # 之后所有 drift / anchor_feat 提取复用完全相同的张量。
        self._diag_fixed_inputs = None           # [N, C, H, W] CPU
        self._diag_fixed_targets = None          # [N] CPU
        self._diag_anchor_feat_after_phase2 = None  # post-P2 点 retrieved anchor_feat

    def _init_local_model(self, model_name):
        """
        根据指定的架构初始化本地模型（传入 FedSMR 超参数）

        Args:
            model_name: 模型架构名称

        Returns:
            初始化的模型实例
        """
        if model_name == 'Tail_Anchor':
            return Tail_Anchor(
                anchor_size=10,
                key_size=768,
                nb_class=self.nb_classes,
                soft_anchor=self.use_soft_anchor,
                soft_temperature=self.soft_temperature,
                soft_anchor_ratio=self.soft_anchor_ratio,
                diversity_margin=self.diversity_margin,
                use_seen_routing=self.use_seen_routing,
                # GPC-DR
                use_proto_calibration=self.use_proto_calibration,
                proto_beta=self.proto_beta,
                proto_temperature=self.proto_temperature,
                adaptive_gamma=self.adaptive_gamma,
                gamma_max=self.gamma_max,
                use_diff_retrieval=self.use_diff_retrieval,
                use_task_isolated_diff=self.use_task_isolated_diff,
            )

    def _init_log_file(self):
        """
        初始化用于记录准确率指标的 CSV 日志文件

        Returns:
            创建的日志文件路径
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"log_{timestamp}.csv"
        filepath = os.path.join("logs", filename)

        os.makedirs("logs", exist_ok=True)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Round', 'Task_id', 'Client_id', 'Accuracy', 'Notes', 'Phase'])

        return filepath

    def _log_accuracy(self, accuracy, notes, phase, task_id=None):
        """
        将准确率指标记录到日志文件
        """
        if task_id is None:
            task_id = self.task_id

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, self.round, task_id, self.id, accuracy, notes, phase])

    def set_round(self, round_num):
        """更新当前全局轮次编号"""
        self.round = round_num

    def get_data_office_home(self, task_id, data, mask):
        """准备 Office-Home 数据集的训练数据"""
        self.train_dataset = data
        self.current_class = mask
        self.class_mask.append(mask)
        print(f'Client {self.id}, Task {task_id} has {len(self.current_class)} classes: {self.current_class}')

        trainset = self.train_dataset
        traindata, testdata = random_split(trainset, [int(len(trainset) * 0.7), len(trainset) - int(len(trainset) * 0.7)])
        testdata = deepcopy(testdata)
        self.test_loader.append(testdata)
        self.traindata = traindata

    def get_data(self, task_id):
        """准备标准数据集的训练数据"""
        self.train_dataset = self.train_data[task_id]
        self.current_class = self.class_mask[task_id]
        print(f'Client {self.id}, Task {task_id} has {len(self.current_class)} classes: {self.current_class}')

        trainset = self.train_dataset
        traindata, testdata = random_split(trainset, [int(len(trainset) * 0.7), len(trainset) - int(len(trainset) * 0.7)])
        self.test_loader.append(testdata)
        self.traindata = traindata
        print(len(traindata))

    def update_data(self, round, args):
        """切换到新任务时更新训练数据"""
        task = round // self.task_per_global_epoch
        if self.task_id != task:
            if args.data_name in ['cifar100', '5datasets', 'ImageNet-R', 'svhn-mnist']:
                self.get_data(task)
            self.task_id = task

    def _compute_msp_losses(self, feat_prompt, anchor_feat, round_num):
        """
        FedSMR-v2: MSP 损失 (Seen-Only Diversity + Key&Anchor Temporal)

        去除 Coherence，增加 Key temporal。
        只约束已见类别。
        """
        total_msp_loss = torch.tensor(0.0, device=self.device)

        # ---- 1. Seen-Only Diversity (带 margin) ----
        if self.msp_diversity_coeff > 0 and hasattr(self.model, 'anchor_diversity_loss'):
            loss_anchor_div = self.model.anchor_diversity_loss()
            total_msp_loss = total_msp_loss + self.msp_diversity_coeff * loss_anchor_div

        # ---- 2. Key + Anchor Temporal Stability (cosine loss, hard usage, old-class-only) ----
        if self.msp_temporal_coeff > 0:
            if self.prev_anchor_pool is not None and self.prev_key_pool is not None:
                # 只约束旧类（已见但非当前 task 的类）
                old_classes = sorted(self.old_seen_classes)
                if len(old_classes) > 0:
                    old_mask = torch.tensor(old_classes, device=self.device, dtype=torch.long)

                    # --- Anchor temporal (cosine) ---
                    curr_anchor = F.normalize(
                        self.model.anchor_pool[old_mask], dim=1
                    )
                    prev_anchor = F.normalize(
                        self.prev_anchor_pool[old_mask].to(self.device), dim=1
                    )
                    loss_anchor_tmp = (1.0 - (curr_anchor * prev_anchor).sum(dim=1))

                    # --- Key temporal (cosine) ---
                    curr_key_pool = self.model.key.reshape(-1, self.model.key_size)
                    curr_key = F.normalize(curr_key_pool[old_mask], dim=1)
                    prev_key = F.normalize(
                        self.prev_key_pool[old_mask].to(self.device), dim=1
                    )
                    loss_key_tmp = (1.0 - (curr_key * prev_key).sum(dim=1))

                    # Hard usage 置信度加权 (P1.7 FIX: 使用 round 开始前的 usage)
                    if hasattr(self, 'prev_anchor_usage') and self.prev_anchor_usage is not None:
                        usage = self.prev_anchor_usage.to(self.device)
                        confidence = usage[old_mask]
                        conf_max = confidence.max()
                        if conf_max > 0:
                            confidence = confidence / (conf_max + 1e-8)
                            confidence = 0.2 + 0.8 * confidence  # floor
                        else:
                            confidence = torch.ones_like(confidence)
                    else:
                        confidence = torch.ones(len(old_classes), device=self.device)

                    loss_anchor_tmp = (confidence * loss_anchor_tmp).mean()
                    loss_key_tmp = (confidence * loss_key_tmp).mean()

                    eta = self.key_temporal_ratio
                    loss_temporal = loss_anchor_tmp + eta * loss_key_tmp
                    total_msp_loss = total_msp_loss + self.msp_temporal_coeff * loss_temporal

        return total_msp_loss

    def _build_semantic_proto_bank(self):
        """GPC-DR: 从 global_protos 提取 prompt-feature prototype bank (前768维)"""
        bank = torch.zeros(self.nb_classes, 768, device=self.device)
        valid = torch.zeros(self.nb_classes, dtype=torch.bool, device=self.device)

        if self.global_protos is None:
            return bank, valid

        for c, proto in self.global_protos.items():
            proto = torch.as_tensor(proto, dtype=torch.float32, device=self.device).view(-1)
            if proto.numel() >= 768:
                bank[int(c)] = proto[:768]
                valid[int(c)] = True

        return bank, valid

    def _compute_pcr_loss(self, routing_logits, target, seen_classes):
        """GPC-DR: PCR loss on proto-calibrated routing logits (no extra temperature)"""
        if len(seen_classes) == 0:
            return torch.tensor(0.0, device=self.device)

        seen_list = sorted(seen_classes)
        seen_idx = torch.tensor(seen_list, device=target.device, dtype=torch.long)
        logits = routing_logits[:, seen_idx]
        global_to_local = {g: i for i, g in enumerate(seen_list)}
        local_target = torch.tensor(
            [global_to_local[int(t.item())] for t in target],
            device=target.device, dtype=torch.long
        )
        return F.cross_entropy(logits, local_target)

    def _build_proto_calib_mask(self, task):
        """GPC-DR: proto calibration restricted to classes of a specific task"""
        mask = torch.zeros(self.nb_classes, dtype=torch.bool, device=self.device)
        mask[torch.tensor(self.class_mask[task], dtype=torch.long, device=self.device)] = True
        return mask

    def _compute_route_loss(self, similarity, target, seen_classes):
        """
        FedSMR-v2: 监督路由损失 L_route

        强制样本特征与正确类别的 key 对齐。
        只计算客户端已见过的类别。
        """
        if len(seen_classes) == 0:
            return torch.tensor(0.0, device=self.device)

        seen_list = sorted(seen_classes)
        seen_idx = torch.tensor(seen_list, device=target.device, dtype=torch.long)

        # 只取 seen classes 的 logits
        route_logits = similarity[:, seen_idx] / self.route_temperature

        # 建立 global label → seen-index 映射
        global_to_local = {g: i for i, g in enumerate(seen_list)}
        local_target = torch.tensor(
            [global_to_local.get(t.item(), 0) for t in target],
            device=target.device, dtype=torch.long
        )

        return F.cross_entropy(route_logits, local_target)

    def _compute_proto_replay_loss(self, global_protos):
        """
        FedSMR-v2: Global Prototype Head Replay

        用服务器维护的全局原型重放旧类，保护旧类分类边界。
        """
        if global_protos is None or len(self.old_seen_classes) == 0:
            return torch.tensor(0.0, device=self.device)

        old_classes = sorted(self.old_seen_classes)
        proto_list = []
        valid_classes = []
        for c in old_classes:
            if c in global_protos:
                proto = global_protos[c]
                if isinstance(proto, torch.Tensor):
                    proto = proto.to(self.device)
                else:
                    proto = torch.from_numpy(np.array(proto)).float().to(self.device)
                # proto 是 1536 维 feat_mixed，直接输入 classification head
                proto_list.append(proto)
                valid_classes.append(c)

        if len(proto_list) == 0:
            return torch.tensor(0.0, device=self.device)

        proto_feat = torch.stack(proto_list)               # (N, 1536)
        proto_target = torch.tensor(valid_classes, device=self.device, dtype=torch.long)

        proto_logits = self.model.head(proto_feat)
        return F.cross_entropy(proto_logits, proto_target)

    def train(self, round, args):
        """
        FedSMR: 使用 Soft Memory Retrieval 执行本地训练

        Phase 1: 训练 prompt 参数（Soft Prompt Retrieval + MSP）
        Phase 2: 训练 anchor pool + classification head（Soft Anchor + MSP + InfoNCE）

        Args:
            round: 当前全局轮次
            args: 命令行参数
        """
        self.original_model.eval()

        # =============================================================
        # E6a-v2-Diag: CP2 (R5_C0_pre_phase2) 恢复分支
        # 跳过 setup 与 Phase1（避免重复初始化 / 重复训练 /
        # vit.load_prompts 用旧 prompt 覆盖已恢复的 Phase1 结果），
        # 直接进入 Phase2 第一个 batch
        # =============================================================
        if self._resume_phase2_ctx is not None:
            ctx = self._resume_phase2_ctx
            self._resume_phase2_ctx = None
            self.set_round(round)
            self.model.to(self.device)
            self.vit.to(self.device)
            train_loader = DataLoader(
                self.traindata, batch_size=args.batch_size,
                num_workers=args.num_workers, shuffle=True
            )
            print(f'Client {self.id} on Task {self.task_id} is training '
                  f'(FedSMR, resumed at Phase2)')

            # ---- v3b-Diag r3: legacy CP2 兼容（pre-P2 baseline 重建） ----
            # 旧 R5_C0_pre_phase2.pth 只有 feature_sample_indices 和旧
            # feature_before（旧运行当时那次增强实例 A 的像素），没有
            # fixed tensors。若不重建，Phase2 后诊断时补建的 fixed
            # tensors 是恢复运行重新增强的实例 B，pre(A)/post(B) 不逐位
            # 相同，P2 FeatureDrift / RetrievedAnchorDrift 会混入 A→B
            # 图像差异。此处（Phase2 第一个 batch 之前，模型仍为真正的
            # pre-Phase2 状态）立即重建 fixed tensors 并在同一模型状态上
            # 重算 pre-P2 baseline，保证 pre/post 严格可比。
            # 注意: 旧 CP2 未保存 pre-Phase1 fixed snapshots，Round5 的
            # P1Drift 无法凭空恢复（不影响 Round10 Task2 诊断）。
            # v3b-Diag r5: 重建条件扩展 —— 旧 checkpoint（r4 及更早代码
            # 保存，fixed tensors 已存在）的 phase_stats_pre 缺少
            # soft-mass 累积器，SoftMassDiag 会缺 post_p1 基线；B-实验
            # 从 R10_C0_pre_phase2.pth 恢复时必须同协议重建 stats。
            _stats_pre_missing_soft = (
                not isinstance(self._diag_phase_stats_pre, dict)
                or 'soft_mass_eval_sum' not in self._diag_phase_stats_pre
            )
            if (self._diag_feature_samples is not None
                    and (self._diag_fixed_inputs is None
                         or self._diag_fixed_targets is None
                         or _stats_pre_missing_soft)):
                self._ensure_diag_fixed_inputs()
                self._diag_feature_before = self._diag_extract_features()
                self._diag_anchor_feat_before_phase2 = (
                    self._diag_extract_anchor_feat()
                )
                # v3b-Diag r4: 同协议重建 pre-P2 accuracy/stats baseline。
                # 旧 CP2 extras 里的 phase_acc_pre 是旧评估协议（无固定
                # seed）算的，若不重建，resumed Round5 的 P2 acc change
                # 是"旧协议 pre - 新 fixed-seed 协议 post"的混合口径。
                self._diag_phase_acc_pre = {}
                self._diag_phase_stats_pre = None
                for t in range(0, self.task_id + 1):
                    acc_t, stats_t = self._diag_eval_task(
                        t, collect_stats=(t == 0)
                    )
                    self._diag_phase_acc_pre[f'task{t}'] = acc_t
                    if t == 0:
                        self._diag_phase_stats_pre = stats_t

            task_hard_usage = self._run_phase2(
                round, args, train_loader,
                global_step=ctx['phase2_global_step'],
                total_steps=ctx['total_steps'],
                skip_start_hook=True,
            )
            self._post_train(round, args, train_loader, task_hard_usage)
            return

        self.set_round(round)

        # 保存上一轮记忆状态（用于 Key+Anchor temporal stability）
        if hasattr(self.model, 'anchor_pool'):
            self.prev_anchor_pool = self.model.anchor_pool.data.clone().cpu()
        if hasattr(self.model, 'key'):
            self.prev_key_pool = self.model.key.data.clone().cpu()
        # P1.7 FIX: 保存本轮开始前的 usage（不含当前任务的新数据）
        if hasattr(self.model, 'get_anchor_usage'):
            self.prev_anchor_usage = self.model.get_anchor_usage().detach().clone().cpu()

        # 追踪已见类别
        old_seen = set(self.seen_classes)
        self.seen_classes.update(self.current_class)
        self.old_seen_classes = old_seen - set(self.current_class)

        # 更新 Tail_Anchor 的 seen_class_mask
        if hasattr(self.model, 'set_seen_classes'):
            self.model.set_seen_classes(self.current_class)

        # Load or initialize prompts
        if self.prompts is not None:
            self.vit.load_prompts(self.prompts)
        else:
            self.vit.init_prompts()

        self.model.to(self.device)
        self.vit.to(self.device)
        train_loader = DataLoader(
            self.traindata, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
        )
        print(f'Client {self.id} on Task {self.task_id} is training (FedSMR)')

        # =============================================================
        # E6a-v3b-Diag: Phase1 开始钩子
        # 位置: 数据/状态初始化完成、Phase1 第一个优化步之前。
        # 输出各任务 pre_phase1 准确率 + Phase1 前快照（key/anchor/head/
        # feature/retrieved anchor），用于切分 P1 / P2 两个窗口的漂移。
        # =============================================================
        if self._tidr_diag_config is not None:
            self._tidr_diag_phase1_start(round, args)

        # 计算总步数（用于温度退火）
        total_steps = self.local_epoch * len(train_loader)
        global_step = 0

        # =============================================================
        # Phase 1: Train prompts with Soft Prompt Retrieval + MSP
        # =============================================================
        optimizer = torch.optim.Adam(self.vit.parameters(), lr=self.lr, weight_decay=1e-3)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        for epoch in tqdm(range(self.local_epoch)):
            for input, target in train_loader:
                input = Variable(input, requires_grad=False).to(
                    self.device, non_blocking=True
                )
                target = target.long().to(self.device, non_blocking=True)

                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(input)
                        cls_features = output['pre_logits']

                output = self.vit(input, task_id=self.task_id,
                                  cls_features=cls_features, train=True)
                logits = output['logits']
                pull_off = output['reduce_sim']
                feat_prompt = output['feat']

                # Apply class mask to logits
                mask = self.current_class
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask,
                                           value=float('-inf'))

                loss = criterion(logits, target) - 0.1 * pull_off

                # ---- GPC-DR: Global Prototype Alignment (Phase 1, CE form) ----
                if self.use_gpa and self.global_protos is not None:
                    available_classes = [int(c) for c in self.current_class
                                         if int(c) in self.global_protos]
                    if len(available_classes) >= 2:
                        feat_norm = F.normalize(feat_prompt, dim=1)
                        proto_list = []
                        for c in available_classes:
                            proto = torch.as_tensor(self.global_protos[c],
                                                    dtype=torch.float32,
                                                    device=self.device).view(-1)[:768]
                            proto_list.append(F.normalize(proto, dim=0))
                        gpa_proto_bank = torch.stack(proto_list, dim=0)
                        gpa_logits = feat_norm @ gpa_proto_bank.T / self.proto_temperature

                        proto_to_idx = {c: i for i, c in enumerate(available_classes)}
                        valid_mask = torch.tensor(
                            [int(t.item()) in proto_to_idx for t in target],
                            dtype=torch.bool, device=self.device
                        )
                        if valid_mask.any():
                            valid_logits = gpa_logits[valid_mask]
                            valid_targets = target[valid_mask]
                            gpa_labels = torch.tensor(
                                [proto_to_idx[int(t.item())] for t in valid_targets],
                                dtype=torch.long, device=self.device
                            )
                            gpa_loss = F.cross_entropy(valid_logits, gpa_labels)
                            loss = loss + self.lambda_gpa * gpa_loss
                # ----------------------------------------------------

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                global_step += 1

        # =============================================================
        # Phase 2 + post（E6a-v2-Diag: 原样搬移至 _run_phase2 / _post_train）
        # =============================================================
        task_hard_usage = self._run_phase2(round, args, train_loader,
                                           global_step, total_steps)
        self._post_train(round, args, train_loader, task_hard_usage)

    def _run_phase2(self, round, args, train_loader, global_step, total_steps,
                    skip_start_hook=False):
        """
        Phase 2: Train classification head + anchor_pool (Soft Anchor + MSP + InfoNCE)
        （E6a-v2-Diag: 自 train() 原样搬移，训练逻辑不变）

        Args:
            skip_start_hook: CP2 恢复时跳过 Phase2 开始钩子（快照已在恢复时注入）
        """
        self.model.train()  # usage 只在训练模式累积
        # =============================================================
        # E6a-v3a-Diag: No-Anchor-WD
        # 诊断实验（默认关闭）: Phase2 中 anchor_pool 免 weight decay。
        # No-Anchor-WD is used to test whether weight decay is the primary
        # driver of old-anchor norm collapse during Phase2.
        # （旧 Anchor 行并非只剩 WD 梯度——仍可能收到 temporal / diversity /
        #   少量 route 等任务梯度；但 WD × Adam 自适应归一化是范数塌缩的
        #   主驱动，已由 E6a-v3a 因果验证: norm_ratio 0.0102→1.0031）
        # =============================================================
        if self.anchor_no_wd and hasattr(self.model, 'anchor_pool'):
            anchor_param = self.model.anchor_pool
            other_params = [
                p for p in self.model.parameters()
                if p is not anchor_param
            ]
            optimizer = torch.optim.Adam(
                [
                    {
                        'params': other_params,
                        'weight_decay': 1e-3,
                    },
                    {
                        'params': [anchor_param],
                        'weight_decay': 0.0,
                    },
                ],
                lr=self.lr,
            )
            print("[E6a-v3a-Diag] Phase2 optimizer: anchor_pool "
                  "weight_decay=0 (No-Anchor-WD), other params wd=1e-3")
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-3)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # =============================================================
        # E1-Repair v1 diagnostics: sample-weighted accumulation
        # =============================================================
        task_hard_usage = torch.zeros(self.nb_classes, device=self.device)

        diag_sample_count = 0
        diag_old_mass_sum = 0.0
        diag_hard_collision_sum = 0.0
        diag_entropy_sum = 0.0
        diag_max_weight_sum = 0.0

        old_anchor_idx = None
        t0_topk_coverage = None

        if self.task_id >= 1 and self.use_soft_anchor:
            usage_t0 = self.task_anchor_usage.get(0)
            if usage_t0 is not None and usage_t0.sum().item() > 0:
                usage_t0_dev = usage_t0.to(self.device)
                num_used = int((usage_t0_dev > 0).sum().item())
                k = min(8, num_used)
                if k > 0:
                    old_anchor_idx = torch.topk(usage_t0_dev, k=k).indices
                    t0_topk_coverage = (
                        usage_t0_dev[old_anchor_idx].sum()
                        / usage_t0_dev.sum().clamp_min(1e-12)
                    ).item()

        # =============================================================
        # E6a-v2-Diag / E6a-v3b-Diag: Phase2 开始钩子
        # 位于 Phase2 全部初始化之后、第一个 batch 之前
        # （诊断配置 → CP2 保存点 + 快照；Task Checkpoint 配置 →
        #   Task{k}_C0_post_phase1.pth 保存点）
        # =============================================================
        if (self._tidr_diag_config is not None
                and not skip_start_hook):
            head_id_before = id(self.model.head)

            self._tidr_diag_phase2_start(round, args, global_step, total_steps)

            # ---- TIDR-Safety: 诊断不得破坏 Phase2 训练结构 ----
            head_id_after = id(self.model.head)
            optimizer_param_ids = {
                id(p)
                for group in optimizer.param_groups
                for p in group['params']
            }
            head_param_ids = {
                id(p) for p in self.model.head.parameters()
            }
            optimizer_has_current_head = (
                head_param_ids.issubset(optimizer_param_ids)
            )
            print(f"[TIDR-Safety] head_object_preserved={head_id_before == head_id_after} "
                  f"model_training={self.model.training} "
                  f"optimizer_has_current_head={optimizer_has_current_head}")
        elif (self._task_ckpt_config is not None
              and not skip_start_hook):
            # 仅 Task Checkpoint（诊断关闭）: 只保存，不做评估
            self._task_ckpt_phase2_start(round, args, global_step, total_steps)

        # =============================================================
        # E6a-v3b B-实验: Phase2 最小因果干预（默认全部关闭 = B0 原样）
        # 位于 phase2_start 钩子之后 → pre-P2 诊断基线在 Normal 协议下
        # 采集；训练循环结束、返回之前恢复 → phase2_end 诊断（post-P2
        # PhaseDiag / SoftMassDiag / 各 CF）同样在 Normal 协议下评估。
        # =============================================================
        # B1/B3: Seen-only retrieval —— Phase2 训练期 hard route 与
        # soft attention 只允许 T0..T_cur 的 keys/anchors（屏蔽 future
        # unseen keys 的候选竞争）。route loss / MSP 不受影响（它们直接
        # 用 compute_similarity / 参数本身，不走 _get_routing_similarity）。
        b1_backup = None
        if self.p2_seen_only:
            b1_backup = (
                self.model.use_seen_routing,
                self.model.seen_class_mask.detach().clone(),
            )
            seen_union = sorted(set().union(*[
                set(int(c) for c in self.class_mask[k])
                for k in range(0, self.task_id + 1)
            ]))
            b1_mask = torch.zeros(self.nb_classes, dtype=torch.bool)
            b1_mask[torch.tensor(seen_union, dtype=torch.long)] = True
            self.model.use_seen_routing = True
            self.model.seen_class_mask.copy_(
                b1_mask.to(
                    device=self.model.seen_class_mask.device,
                    dtype=self.model.seen_class_mask.dtype
                )
            )
            print(f"[E6a-v3b-B1] Phase2 Seen-only retrieval: "
                  f"client={self.id} task={self.task_id} "
                  f"allowed={len(seen_union)}/{self.nb_classes} classes")

        # B2/B3: old-Key freeze —— 真正的冻结（不是仅梯度置零）:
        # 每步 optimizer.step() 之后用 index_copy_ 恢复旧 Key 行快照，
        # 连 weight decay / temporal / route loss 的任何更新一并消除。
        b2_old_idx = None
        b2_key_snapshot = None
        if (self.p2_freeze_old_key
                and self.task_id >= 1
                and len(self.old_seen_classes) > 0):
            b2_old_idx = torch.tensor(
                sorted(self.old_seen_classes), dtype=torch.long,
                device=self.model.key.device
            )
            b2_key_snapshot = (
                self.model.key.data[b2_old_idx].detach().clone()
            )
            print(f"[E6a-v3b-B2] Phase2 old-Key freeze: "
                  f"client={self.id} task={self.task_id} "
                  f"n_old_keys={b2_old_idx.numel()} "
                  f"(post-step index_copy_ restore)")

        # E6a-v2-Diag: 优化步数统计（StepDiag，默认关闭时零开销）
        diag_steps_active = self._tidr_diag_config is not None
        diag_num_batches = 0
        diag_num_samples = 0

        for epoch in tqdm(range(self.local_epoch)):
            for input, target in train_loader:
                input = Variable(input, requires_grad=False).to(
                    self.device, non_blocking=True
                )
                target = target.long().to(self.device, non_blocking=True)

                if diag_steps_active:
                    diag_num_batches += 1
                    diag_num_samples += len(target)

                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(input)
                        cls_features = output['pre_logits']
                    output = self.vit(input, task_id=self.task_id,
                                      cls_features=cls_features, train=True)
                    feat_prompt = output['feat']

                # GPC-DR: build proto bank + calib mask
                proto_bank, proto_valid = self._build_semantic_proto_bank()
                proto_calib_mask = self._build_proto_calib_mask(self.task_id)

                pre, output_mixed, pull_off2, anchor_feat, attn_weights, hard_idx, routing_logits = self.model(
                    feat_prompt.to(self.device), target.to(self.device),
                    proto_bank=proto_bank, proto_valid_mask=proto_valid,
                    proto_calib_mask=proto_calib_mask
                )
                logits = pre

                # E1-Repair v1: sample-weighted routing diagnostics
                task_hard_usage += torch.bincount(hard_idx, minlength=self.nb_classes).float()

                # E6a-v2-Diag: Task1 样本 hard route 到旧类 Anchor 的比例
                # （判断旧 Anchor 漂移来自 Task1 CE 直接更新还是 MSP diversity 全局梯度）
                if self._tidr_diag_config is not None and self._diag_old_key_mask is not None:
                    old_mask_dev = self._diag_old_key_mask.to(hard_idx.device)
                    self._diag_task1_old_route_hits += (
                        old_mask_dev[hard_idx.detach()].sum().item()
                    )
                    self._diag_task1_old_route_total += hard_idx.numel()

                if self.use_soft_anchor:
                    attn_det = attn_weights.detach()
                    hard_det = hard_idx.detach()
                    bs = hard_det.size(0)

                    clamped = attn_det.clamp_min(1e-12)
                    diag_entropy_sum += -(clamped * torch.log(clamped)).sum(dim=1).sum().item()
                    diag_max_weight_sum += attn_det.max(dim=1).values.sum().item()

                    if old_anchor_idx is not None:
                        diag_old_mass_sum += attn_det[:, old_anchor_idx].sum().item()
                        diag_hard_collision_sum += torch.isin(
                            hard_det, old_anchor_idx
                        ).float().sum().item()

                    diag_sample_count += bs

                # Calculate InfoNCE loss if global prototypes exist
                if self.global_protos is None:
                    loss_infonce = 0
                else:
                    count = 0
                    loss_infonce = None
                    for i, label in enumerate(target):
                        if label.item() in self.global_protos:
                            count += 1
                            feature = output_mixed[i].unsqueeze(0)
                            loss_instance = self._calculate_infonce(
                                feature, label.item(),
                                (round + 1) % self.task_per_global_epoch == 0,
                            )
                            loss_infonce = (
                                loss_instance
                                if loss_infonce is None
                                else loss_infonce + loss_instance
                            )

                    loss_infonce = loss_infonce / count if count != 0 else 0

                # Apply class mask
                mask = self.current_class
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask,
                                           value=float('-inf'))

                loss = (
                    criterion(logits, target)
                    + 0.2 * self.task_per_global_epoch * loss_infonce
                    - 0.1 * pull_off2
                )

                # ---- GPC-DR: PCR loss (no extra temperature) or legacy route loss ----
                if self.use_proto_calibration:
                    pcr_loss = self._compute_pcr_loss(
                        routing_logits, target, self.seen_classes
                    )
                    loss = loss + self.lambda_pcr * pcr_loss
                elif self.use_route_loss:
                    sim = self.model.compute_similarity(feat_prompt.to(self.device))
                    route_loss = self._compute_route_loss(
                        sim, target, self.seen_classes
                    )
                    loss = loss + self.lambda_route * route_loss

                # ---- FedSMR-v2: MSP (Diversity + Key&Anchor Temporal) ----
                if self.use_msp:
                    msp_loss = self._compute_msp_losses(None, None, round)
                    loss = loss + msp_loss

                # ---- FedSMR-v2: Prototype Head Replay ----
                if self.use_proto_replay and self.global_protos is not None:
                    proto_loss = self._compute_proto_replay_loss(self.global_protos)
                    loss = loss + self.lambda_proto * proto_loss
                # ----------------------------------------------------

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # B2/B3: old-Key freeze（post-step 恢复快照，消除包括
                # weight decay 在内的全部旧 Key 更新）
                if b2_key_snapshot is not None:
                    with torch.no_grad():
                        self.model.key.data.index_copy_(
                            0, b2_old_idx, b2_key_snapshot
                        )

                global_step += 1

        # E6a-v2-Diag: StepDiag 统计（仅诊断开启时）
        if diag_steps_active:
            self._diag_step_stats = {
                'phase2_optimizer_steps': diag_num_batches,  # 每个 batch 一次 optimizer.step()
                'phase2_num_batches': diag_num_batches,
                'phase2_num_samples': diag_num_samples,
                'phase2_mean_batch_size': (
                    diag_num_samples / diag_num_batches if diag_num_batches else 0.0
                ),
            }

        # =============================================================
        # E1 routing diagnostics (sample-weighted summary)
        # （E6a-v2-Diag: 依赖 _run_phase2 局部累计量，自 post 段搬移至此；
        #   可见日志顺序与原实现一致）
        # =============================================================
        if self.task_id >= 1 and self.use_soft_anchor and diag_sample_count > 0:
            entropy = diag_entropy_sum / diag_sample_count
            max_weight = diag_max_weight_sum / diag_sample_count

            if old_anchor_idx is not None:
                old_anchor_mass = diag_old_mass_sum / diag_sample_count
                hard_collision = diag_hard_collision_sum / diag_sample_count

                print(f"[E1-Diag] client={self.id} task={self.task_id} "
                      f"T0_top8_coverage={t0_topk_coverage:.4f} "
                      f"T0_anchor_mass={old_anchor_mass:.4f} "
                      f"hard_collision={hard_collision:.4f} "
                      f"entropy={entropy:.4f} max_w={max_weight:.4f}")
            else:
                print(f"[E1-Diag] client={self.id} task={self.task_id} "
                      f"entropy={entropy:.4f} max_w={max_weight:.4f}")

        # =============================================================
        # E6a-v3b B-实验: 干预状态恢复（phase2_end 诊断在 Normal 协议下运行）
        # B1: 恢复 use_seen_routing / seen_class_mask 原值
        # B2: 自检 —— 训练结束后旧 Key 行应与快照逐位一致
        # =============================================================
        if b1_backup is not None:
            self.model.use_seen_routing = b1_backup[0]
            self.model.seen_class_mask.copy_(b1_backup[1])
            print(f"[E6a-v3b-B1] Seen-only retrieval disabled after "
                  f"Phase2 (routing config restored)")

        if b2_key_snapshot is not None:
            with torch.no_grad():
                b2_pass = torch.equal(
                    self.model.key.data[b2_old_idx], b2_key_snapshot
                )
            print(f"[E6a-v3b-B2] old-Key freeze check: "
                  f"old_key_bitwise_preserved={b2_pass}")

        return task_hard_usage

    def _post_train(self, round, args, train_loader, task_hard_usage):
        """
        Phase 2 结束后的收尾（E6a-v2-Diag: 自 train() 原样搬移，逻辑不变）

        usage 保存 / TIDR-KeyDiag / 原型提取 / head 快照 / 本地评估 / prompt 保存
        """
        # =============================================================
        # Save hard-anchor usage across global rounds
        # =============================================================
        round_usage = task_hard_usage.detach().cpu()

        if self.task_id not in self.task_anchor_usage:
            self.task_anchor_usage[self.task_id] = round_usage.clone()
        else:
            self.task_anchor_usage[self.task_id] += round_usage

        # =============================================================
        # TIDR-KeyDiag: 本轮训练造成的旧 Key 漂移（旧 Key 是否被改动）
        # 区分 "新 Key 侵入" vs "旧 Key 被直接破坏"
        # =============================================================
        if (
            self.task_id >= 1
            and self.prev_key_pool is not None
            and len(self.old_seen_classes) > 0
        ):
            with torch.no_grad():
                old_idx = torch.tensor(
                    sorted(self.old_seen_classes),
                    dtype=torch.long,
                    device=self.device
                )

                curr_key = F.normalize(
                    self.model.key[old_idx], dim=1
                )
                prev_key = F.normalize(
                    self.prev_key_pool.to(self.device)[old_idx], dim=1
                )

                old_key_cos = (curr_key * prev_key).sum(dim=1)
                old_key_drift = 1.0 - old_key_cos

                # E6a-v2-Diag: 额外输出 median / p95（仅诊断开启时，避免改变默认日志）
                extra_fields = ''
                if self.run_tidr_diagnostics:
                    drift_sorted = old_key_drift.sort().values
                    n_drift = drift_sorted.numel()
                    median_drift = drift_sorted[n_drift // 2].item()
                    p95_drift = drift_sorted[
                        min(n_drift - 1, int(0.95 * n_drift))
                    ].item()
                    extra_fields = (
                        f" median_old_key_drift={median_drift:.6f}"
                        f" p95_old_key_drift={p95_drift:.6f}"
                    )

                print(
                    f"[TIDR-KeyDiag] client={self.id} task={self.task_id} "
                    f"mean_old_key_drift={old_key_drift.mean().item():.6f} "
                    f"max_old_key_drift={old_key_drift.max().item():.6f}"
                    f"{extra_fields}"
                )

        # Extract local prototypes from training data
        self.model.eval()  # 评估/原型提取不累积 usage
        target_list = []
        feature_list = []

        for input, target in train_loader:
            input = Variable(input, requires_grad=False).to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input.to(self.device))
                    output = output['pre_logits'].requires_grad_(False)
                output = self.vit(input, task_id=self.task_id, cls_features=output,
                                  train=True)
                proto_bank, proto_valid = self._build_semantic_proto_bank()
                proto_calib_mask = self._build_proto_calib_mask(self.task_id)

                _, output_mixed, _, _, _, _, _ = self.model(
                    output['feat'].to(self.device), target.to(self.device),
                    proto_bank=proto_bank, proto_valid_mask=proto_valid,
                    proto_calib_mask=proto_calib_mask
                )

            if not np.isnan(output_mixed.cpu().detach().numpy()).any():
                target_list.append(target)
                feature_list.append(output_mixed)

        local_protos = {}
        if target_list:
            target_list = torch.cat(target_list, dim=0)
            feature_list = torch.cat(feature_list, dim=0)

            for class_index in self.current_class:
                data_index = (target_list == class_index).nonzero().squeeze(-1)
                if data_index.shape[0] != 0:
                    all_features = feature_list[data_index]
                    proto = all_features.mean(0).cpu().detach().numpy()
                    local_protos[class_index] = proto

        self.local_protos = local_protos
        self.heads[self.task_id] = deepcopy(self.model.get_head())

        # =============================================================
        # E6a-v2-Diag / E6a-v3b-Diag: Phase2 结束钩子
        # PhaseDiag post / StepDiag / FeatureDiag / KeyCF Rollback /
        # Classwise CSV / CP3 保存（本地、服务器聚合之前 → Local 范畴）；
        # Task Checkpoint 配置（诊断关闭）→ Task{k}_C0_post_phase2.pth
        # =============================================================
        if self._tidr_diag_config is not None:
            self._tidr_diag_phase2_end(round, args)
        elif self._task_ckpt_config is not None:
            self._task_ckpt_phase2_end(round, args)

        # Evaluate after training
        if self.task_id == 0:
            self.evaluate(self.task_id, args.nb_classes,
                          phase="Local Training", notes_prefix="Local evaluation on task")
        else:
            self.evaluate(0, args.nb_classes,
                          phase="Local Training", notes_prefix="Local evaluation on task")
            self.evaluate(self.task_id, args.nb_classes,
                          phase="Local Training", notes_prefix="Local evaluation on task")

        self.prompts = deepcopy(self.vit.get_prompts())

    # =============================================================
    # E6a-v2-Diag: 跨任务遗忘机制诊断
    # （--run_tidr_diagnostics 开启且 Server 注入配置时才执行；
    #   全程 model.eval() + torch.no_grad()，不产生梯度、不污染训练/聚合）
    # =============================================================

    def _tidr_diag_phase1_start(self, round, args):
        """
        E6a-v3b-Diag: Phase1 开始钩子（任务首轮注入诊断配置时执行）
        位置: 数据/状态初始化完成、Phase1 第一个优化步之前。

        1) 各任务 pre_phase1 准确率（上一 Task 结束、聚合分发后的状态）
        2) Phase1 前快照: key / anchor_pool / model.head / heads[t] /
           固定 Task0 样本 feature / retrieved anchor_feat
           （用于切分 P1 / P2 两个窗口的漂移归因）

        只读诊断：train/eval 状态与 RNG 由各 _diag_* 函数内部恢复。
        """
        # 跨轮计数器复位（诊断轮不再只有 Round5 一次）
        self._diag_task1_old_route_hits = 0
        self._diag_task1_old_route_total = 0
        self._diag_step_stats = None

        # ---- 1) pre_phase1 准确率（0..task_id 全部任务） ----
        # 注意解释口径（v3b-Diag r2 改名）: 该点位于 task switch 之后
        # （task_id 已更新、update_data/状态初始化/prompt 加载已完成）、
        # Phase1 第一个优化步之前，即 post-task-switch / pre-Phase1，
        # 不是上一轮结束时的旧 task-stage 状态。评估协议
        # vit(task_id=self.task_id) 已使用新任务 stage——若 task switch
        # 本身改变 prompt-stage 行为，Task0 可能在一个优化步都没跑
        # 之前就先掉。若出现 R9 end=88.86 → pre-P1=40 → post-P1=39，
        # 结论应是 "collapse occurs at the Task1→Task2 boundary before
        # Phase1 optimization"，而非 "Phase1 打崩了"。
        accs = {}
        for t in range(0, self.task_id + 1):
            accs[f'task{t}'] = self._diag_eval_task(t, collect_stats=False)[0]
        self._diag_phase_acc_pre_p1 = accs
        fields = ' '.join(
            f"task{t}_acc_post_task_switch_pre_phase1={accs[f'task{t}']:.2f}"
            for t in range(0, self.task_id + 1)
        )
        print(f"[TIDR-PhaseDiag] client={self.id} task={self.task_id} {fields}")

        # ---- 2) Phase1 前快照 ----
        self._diag_key_pre_p1 = self.model.key.data.clone().cpu()
        self._diag_anchor_pre_p1 = self.model.anchor_pool.data.clone().cpu()
        self._diag_head_pre_p1 = {
            k: v.detach().clone().cpu()
            for k, v in self.model.head.state_dict().items()
        }
        self._diag_heads_pre_p1 = {}
        for t in range(self.task_id):
            if self.heads[t] is not None:
                self._diag_heads_pre_p1[t] = {
                    k: v.detach().clone().cpu()
                    for k, v in self.heads[t].state_dict().items()
                }
        # 幂等: indices 已存在但 fixed tensors 缺失（旧 checkpoint 恢复）
        # 时会用已有 indices 重新缓存张量（内部含 RNG 保护）
        self._ensure_diag_fixed_inputs()
        self._diag_feature_pre_p1 = self._diag_extract_features()
        self._diag_anchor_feat_pre_p1 = self._diag_extract_anchor_feat()

    def _diag_print_p1_drift(self):
        """
        E6a-v3b-Diag: Phase1 窗口漂移（pre_phase1 → post_phase1）
        在 phase2_start 钩子中调用（此时 Phase1 已结束）。
        Phase1 只训练 ViT/Prompt: key/anchor/head 预期不变（验证），
        feature / retrieved anchor_feat 预期漂移（ViT 表示变化）。
        """
        if (self._diag_feature_pre_p1 is None
                or self._diag_feature_before is None
                or len(self.old_seen_classes) == 0):
            return  # 无 pre_phase1 快照（CP2 恢复 / standalone）

        old_idx = torch.tensor(
            sorted(self.old_seen_classes),
            dtype=torch.long, device=self.device
        )

        # ---- key / anchor（Phase1 不应触碰，预期 drift≈0） ----
        key_pre = F.normalize(
            self._diag_key_pre_p1.to(self.device)[old_idx], dim=1)
        key_now = F.normalize(
            self.model.key.detach()[old_idx], dim=1)
        key_drift = 1.0 - (key_pre * key_now).sum(dim=1)

        a_pre = self._diag_anchor_pre_p1.to(self.device)[old_idx]
        a_now = self.model.anchor_pool.detach()[old_idx]
        a_cos = 1.0 - F.cosine_similarity(a_pre, a_now, dim=1)
        a_nr = a_now.norm(dim=1) / a_pre.norm(dim=1).clamp_min(1e-8)

        # ---- head（Phase1 不训练 head，预期不变） ----
        head_now = self.model.head.state_dict()
        head_l2_p1 = 0.0
        for k, v in self._diag_head_pre_p1.items():
            head_l2_p1 = max(
                head_l2_p1,
                (head_now[k].detach().cpu() - v).abs().max().item()
            )
        heads_max_p1 = 0.0
        for t, sd in self._diag_heads_pre_p1.items():
            cur = self.heads[t].state_dict()
            for k, v in sd.items():
                heads_max_p1 = max(
                    heads_max_p1,
                    (cur[k].detach().cpu() - v).abs().max().item()
                )

        # ---- feature / retrieved anchor_feat（P1 窗口核心指标） ----
        fb = self._diag_feature_pre_p1.to(self.device)
        fa = self._diag_feature_before.to(self.device)
        f_drift = 1.0 - F.cosine_similarity(fb, fa, dim=1)

        ab = self._diag_anchor_feat_pre_p1.to(self.device)
        aa = self._diag_anchor_feat_before_phase2.to(self.device)
        af_cos = 1.0 - F.cosine_similarity(ab, aa, dim=1)
        af_nr = aa.norm(dim=1) / ab.norm(dim=1).clamp_min(1e-8)
        af_l2 = (aa - ab).norm(dim=1)

        print(f"[TIDR-P1Drift] client={self.id} task={self.task_id} "
              f"mean_old_key_drift_p1={key_drift.mean().item():.6f} "
              f"mean_old_anchor_cos_drift_p1={a_cos.mean().item():.6f} "
              f"mean_old_anchor_norm_ratio_p1={a_nr.mean().item():.6f} "
              f"head_max_change_p1={head_l2_p1:.6f} "
              f"old_heads_max_change_p1={heads_max_p1:.6f} "
              f"mean_old_feature_drift_p1={f_drift.mean().item():.6f} "
              f"max_old_feature_drift_p1={f_drift.max().item():.6f} "
              f"mean_retrieved_anchor_cos_drift_p1={af_cos.mean().item():.6f} "
              f"mean_retrieved_anchor_norm_ratio_p1={af_nr.mean().item():.6f} "
              f"mean_retrieved_anchor_l2_drift_p1={af_l2.mean().item():.6f}")

    def _diag_snapshot_extras(self):
        """
        E6a-v3b-Diag: 诊断快照（写入 checkpoint extras）
        使离线恢复（CP2 / Task post-P1 / Task post-P2 / standalone）能
        完整复现 P1Drift / HeadDrift / RetrievedAnchorDrift 与固定输入
        （否则只存 indices 时随机增强会让 drift 混入 augmentation noise，
          且 pre_p1 快照缺失导致离线 P1 窗口不可复现）。
        """
        return {
            'anchor_feat_before_phase2': self._diag_anchor_feat_before_phase2,
            'feature_pre_p1': self._diag_feature_pre_p1,
            'anchor_feat_pre_p1': self._diag_anchor_feat_pre_p1,
            'key_pre_p1': self._diag_key_pre_p1,
            'anchor_pre_p1': self._diag_anchor_pre_p1,
            'head_pre_p1': self._diag_head_pre_p1,
            'heads_pre_p1': self._diag_heads_pre_p1,
            'fixed_inputs': self._diag_fixed_inputs,
            'fixed_targets': self._diag_fixed_targets,
        }

    def _tidr_diag_phase2_start(self, round, args, global_step, total_steps):
        """
        E6a-v2-Diag / E6a-v3b-Diag: Phase2 开始钩子
        位置: Phase1 结束、Phase2 全部初始化完成、第一个 batch 之前（CP2 保存点）

        E6a-v3b-Diag 扩展: pre_phase2 评估从 Task0/Task1 扩展到 0..task_id
        全部任务；新增 retrieved anchor_feat 快照与 P1 窗口漂移输出。
        """
        cfg = self._tidr_diag_config

        # 1) Key 快照 + old_key_mask
        #    old-key 定义与 TIDR-KeyDiag 完全一致: old_seen_classes
        #    （= 已见类别 - 当前任务类别，任务间共享类别自动排除）
        self._diag_key_before_phase2 = self.model.key.data.clone().cpu()
        # Anchor 快照（Phase2 前旧 Anchor，AnchorCF / AnchorDrift 用）
        self._diag_anchor_before_phase2 = self.model.anchor_pool.data.clone().cpu()
        old_idx = sorted(self.old_seen_classes)
        old_key_mask = torch.zeros(self.nb_classes, dtype=torch.bool)
        if old_idx:
            old_key_mask[torch.tensor(old_idx, dtype=torch.long)] = True
        self._diag_old_key_mask = old_key_mask

        # 2) PhaseDiag pre（= post_phase1）: 评估 0..task_id 全部任务
        #    （评估协议镜像 evaluate(): vit(task_id=self.task_id), heads[task]）
        accs = {}
        stats0_pre = None
        for t in range(0, self.task_id + 1):
            acc_t, stats_t = self._diag_eval_task(t, collect_stats=(t == 0))
            accs[f'task{t}'] = acc_t
            if t == 0:
                stats0_pre = stats_t
        self._diag_phase_acc_pre = accs
        self._diag_phase_stats_pre = stats0_pre
        fields = ' '.join(
            f"task{t}_acc_pre_phase2={accs[f'task{t}']:.2f} "
            f"(=post_phase1)"
            for t in range(0, self.task_id + 1)
        )
        print(f"[TIDR-PhaseDiag] client={self.id} task={self.task_id} {fields}")

        # 3) FeatureDrift: 固定 Task0 测试样本 + Phase2 前特征快照
        #    （实际参与 Key routing 的 query = vit forward_head 的 feat）
        # 幂等: indices 已存在但 fixed tensors 缺失（旧 checkpoint 恢复）
        # 时会用已有 indices 重新缓存张量（内部含 RNG 保护）
        self._ensure_diag_fixed_inputs()
        self._diag_feature_before = self._diag_extract_features()

        # 3b) Retrieved anchor feature 快照（post-P1 点）
        self._diag_anchor_feat_before_phase2 = self._diag_extract_anchor_feat()

        # 3c) P1 窗口漂移（pre_phase1 → post_phase1）
        self._diag_print_p1_drift()

        # 4) CP2 保存（仅 CF 轮）
        if cfg.get('save'):
            extras = {
                'key_before_phase2': self._diag_key_before_phase2,
                'anchor_before_phase2': self._diag_anchor_before_phase2,
                'old_key_mask': self._diag_old_key_mask,
                'phase2_global_step': global_step,
                'total_steps': total_steps,
                'phase_acc_pre': dict(self._diag_phase_acc_pre),
                'phase_acc_pre_p1': dict(self._diag_phase_acc_pre_p1 or {}),
                'phase_stats_pre': self._diag_phase_stats_pre,
                'feature_before': self._diag_feature_before,
                'feature_sample_indices': list(self._diag_feature_samples),
            }
            extras.update(self._diag_snapshot_extras())
            # 文件名跟随实际 round（默认 Round5 时与 CKPT_CP2 一致，向后兼容）
            save_checkpoint(cfg['server'], 'R5_C0_pre_phase2',
                            f'R{round}_C0_pre_phase2.pth', round, extras)

        # 5) Task Checkpoint: Task{k}_C0_post_phase1.pth
        if self._task_ckpt_config is not None and self._task_ckpt_config.get('save'):
            extras = {
                'diag_task': self.task_id,
                'diag_round': round,
                'key_before_phase2': self._diag_key_before_phase2,
                'anchor_before_phase2': self._diag_anchor_before_phase2,
                'old_key_mask': self._diag_old_key_mask,
                'phase2_global_step': global_step,
                'total_steps': total_steps,
                'phase_acc_pre': dict(self._diag_phase_acc_pre),
                'phase_acc_pre_p1': dict(self._diag_phase_acc_pre_p1 or {}),
                'feature_before': self._diag_feature_before,
                'feature_sample_indices': list(self._diag_feature_samples),
            }
            extras.update(self._diag_snapshot_extras())
            save_checkpoint(
                self._task_ckpt_config['server'], 'Task_C0_post_phase1',
                f"Task{self.task_id}_C0_post_phase1.pth", round, extras
            )

        if cfg.get('stop_at') == 'R5_C0_pre_phase2':
            raise DiagStopException('R5_C0_pre_phase2')

    def _task_ckpt_phase2_start(self, round, args, global_step, total_steps):
        """
        E6a-v3b-Diag: Task{k}_C0_post_phase1.pth 保存
        （仅 --save_task_checkpoints、诊断关闭时的轻量路径：只快照不评估）
        """
        cfg = self._task_ckpt_config
        self._diag_key_before_phase2 = self.model.key.data.clone().cpu()
        self._diag_anchor_before_phase2 = self.model.anchor_pool.data.clone().cpu()
        old_idx = sorted(self.old_seen_classes)
        old_key_mask = torch.zeros(self.nb_classes, dtype=torch.bool)
        if old_idx:
            old_key_mask[torch.tensor(old_idx, dtype=torch.long)] = True
        self._diag_old_key_mask = old_key_mask
        # 幂等: indices 已存在但 fixed tensors 缺失（旧 checkpoint 恢复）
        # 时会用已有 indices 重新缓存张量（内部含 RNG 保护）
        self._ensure_diag_fixed_inputs()

        extras = {
            'diag_task': self.task_id,
            'diag_round': round,
            'key_before_phase2': self._diag_key_before_phase2,
            'anchor_before_phase2': self._diag_anchor_before_phase2,
            'old_key_mask': self._diag_old_key_mask,
            'phase2_global_step': global_step,
            'total_steps': total_steps,
            # P2 窗口 retrieved-anchor drift 基线 + 固定输入张量
            'anchor_feat_before_phase2': self._diag_extract_anchor_feat(),
            'feature_before': self._diag_extract_features(),
            'feature_sample_indices': list(self._diag_feature_samples),
            'fixed_inputs': self._diag_fixed_inputs,
            'fixed_targets': self._diag_fixed_targets,
        }
        save_checkpoint(
            cfg['server'], 'Task_C0_post_phase1',
            f"Task{self.task_id}_C0_post_phase1.pth", round, extras
        )

    def _task_ckpt_phase2_end(self, round, args):
        """
        E6a-v3b-Diag: Task{k}_C0_post_phase2.pth 保存
        （仅 --save_task_checkpoints、诊断关闭时的轻量路径）
        key/anchor 快照来自 _task_ckpt_phase2_start（pre-Phase2）
        """
        cfg = self._task_ckpt_config
        if self._diag_key_before_phase2 is None:
            self._diag_key_before_phase2 = self.model.key.data.clone().cpu()
        if self._diag_anchor_before_phase2 is None:
            self._diag_anchor_before_phase2 = self.model.anchor_pool.data.clone().cpu()

        extras = {
            'diag_task': self.task_id,
            'diag_round': round,
            'key_before_phase2': self._diag_key_before_phase2,
            'key_after_phase2': self.model.key.data.clone().cpu(),
            'anchor_before_phase2': self._diag_anchor_before_phase2,
            'old_key_mask': self._diag_old_key_mask,
            'feature_sample_indices': list(self._diag_feature_samples or []),
        }
        save_checkpoint(
            cfg['server'], 'Task_C0_post_phase2',
            f"Task{self.task_id}_C0_post_phase2.pth", round, extras
        )

    def _tidr_diag_phase2_end(self, round, args):
        """
        E6a-v2-Diag / E6a-v3b-Diag: Phase2 结束钩子（本地、服务器聚合之前）
        PhaseDiag post（0..task_id 全部任务，三点准确率 + P1/P2 变化）/
        StepDiag / FeatureDiag / HeadDrift / RetrievedAnchorDrift /
        RetrievalCF（Seen-only / Exclude-current / Prev-seen-only / T0-only）/
        [CF 轮] KeyCF / AnchorCF / NormCF / DirCF / JointCF /
        Classwise CSV / CP3 保存
        """
        cfg = self._tidr_diag_config
        standalone = bool(cfg.get('standalone'))
        cf = bool(cfg.get('cf', True))

        # ---- 1) PhaseDiag post: 评估 0..task_id 全部任务 ----
        post_accs = {}
        stats0_post = None
        for t in range(0, self.task_id + 1):
            acc_t, stats_t = self._diag_eval_task(t, collect_stats=(t == 0))
            post_accs[f'task{t}'] = acc_t
            if t == 0:
                stats0_post = stats_t
        pre = self._diag_phase_acc_pre or {}
        pre_p1 = self._diag_phase_acc_pre_p1 or {}
        for t in range(0, self.task_id + 1):
            a_p1 = pre_p1.get(f'task{t}', float('nan'))
            a_pre = pre.get(f'task{t}', float('nan'))
            a_post = post_accs[f'task{t}']
            print(f"[TIDR-PhaseDiag] client={self.id} task={self.task_id} "
                  f"task{t}_acc_post_task_switch_pre_phase1={a_p1:.2f} "
                  f"task{t}_acc_post_phase1={a_pre:.2f} "
                  f"task{t}_acc_post_phase2={a_post:.2f} "
                  f"task{t}_acc_p1_change={a_pre - a_p1:+.2f} "
                  f"task{t}_acc_p2_change={a_post - a_pre:+.2f} "
                  f"task{t}_acc_total_change={a_post - a_p1:+.2f}")

        acc0_post = post_accs['task0']

        # ---- 1b) SoftMassDiag: Task0 评估的 group-wise soft attention mass ----
        # 决定性诊断（E6a-v3b Failure Mode II）: Phase2 期间 query feature /
        # raw Anchor 均未变，但 retrieved anchor 巨幅漂移 → 谁拿走了 soft
        # retrieval mass？分组: 被评估任务 / 各已见任务（含当前任务）/
        # future unseen / true key。post_p1 即 pre_phase2 点
        # （_diag_phase_stats_pre，Phase1 结束钩子采集）。
        # 同时输出 hard route 各组（to_eval/to_task{k}/to_future）的
        # post_p1 vs post_p2 —— hard 字段在旧 checkpoint extras 的 pre
        # stats（r2+ 版本）中已存在，可与 soft 字段一并对照。
        # 旧 extras 无 soft-mass 累积器时 soft 字段只打印 post_p2
        # （不打印假的 0 基线）。
        if stats0_post is not None:
            m_sm_post = self._format_diag_stats(stats0_post)
            stats_pre = self._diag_phase_stats_pre
            m_sm_pre = (
                self._format_diag_stats(stats_pre) if stats_pre else None
            )
            has_pre_soft = (
                m_sm_pre is not None
                and stats_pre is not None
                and 'soft_mass_eval_sum' in stats_pre
            )
            sm_groups = (
                ['soft_mass_to_eval_task', 'soft_mass_to_true_key']
                + [f'soft_mass_to_task{k}'
                   for k in sorted(stats0_post.get('per_task_classes') or {})]
                + (['soft_mass_to_future_unseen']
                   if stats0_post.get('future_unseen_classes') else [])
                + ['mean_soft_attn_entropy', 'mean_soft_attn_max_w']
                # hard route 各组（与 soft 同表对照）
                + ['to_eval_task_collision']
                + [f'to_task{k}_collision'
                   for k in sorted(stats0_post.get('per_task_classes') or {})]
                + (['to_future_unseen_collision']
                   if stats0_post.get('future_unseen_classes') else [])
            )
            sm_parts = []
            for sm_key in sm_groups:
                v_post = m_sm_post.get(sm_key)
                if v_post is None:
                    continue
                # soft 字段的 pre 值需要 r5 累积器存在；hard 字段旧版已有
                can_print_pre = (
                    m_sm_pre is not None
                    and m_sm_pre.get(sm_key) is not None
                    and (not sm_key.startswith('soft_') or has_pre_soft)
                )
                if can_print_pre:
                    sm_parts.append(
                        f"{sm_key}_post_p1={m_sm_pre[sm_key]:.4f} "
                        f"{sm_key}_post_p2={v_post:.4f} "
                        f"{sm_key}_p2_change={v_post - m_sm_pre[sm_key]:+.4f}"
                    )
                else:
                    sm_parts.append(f"{sm_key}_post_p2={v_post:.4f}")
            if sm_parts:
                print(f"[TIDR-SoftMassDiag] client={self.id} "
                      f"task={self.task_id} eval_task=0 "
                      + ' '.join(sm_parts))

        # ---- 2) StepDiag ----
        if self._diag_step_stats is not None:
            s = self._diag_step_stats
            print(f"[TIDR-StepDiag] client={self.id} "
                  f"phase2_optimizer_steps={s['phase2_optimizer_steps']} "
                  f"phase2_num_batches={s['phase2_num_batches']} "
                  f"phase2_num_samples={s['phase2_num_samples']} "
                  f"phase2_mean_batch_size={s['phase2_mean_batch_size']:.2f}")

        # 当前任务样本 → 旧类 Anchor 的 hard route 比例
        # （仅正常链路，standalone 无 Phase2）
        if self._diag_task1_old_route_total > 0:
            rate = self._diag_task1_old_route_hits / self._diag_task1_old_route_total
            print(f"[TIDR-AnchorRouteDiag] client={self.id} task={self.task_id} "
                  f"cur_task_to_any_old_anchor_rate={rate:.4f} "
                  f"({self._diag_task1_old_route_hits}/{self._diag_task1_old_route_total})")

        # ---- 3) FeatureDiag: P2 窗口（同一批固定 Task0 样本、相同 task_id） ----
        # 幂等: indices 已存在但 fixed tensors 缺失（旧 checkpoint 恢复）
        # 时会用已有 indices 重新缓存张量（内部含 RNG 保护）
        self._ensure_diag_fixed_inputs()
        self._diag_feature_after = self._diag_extract_features()
        if self._diag_feature_before is not None:
            fb = self._diag_feature_before.to(self.device)
            fa = self._diag_feature_after.to(self.device)
            drift = 1.0 - F.cosine_similarity(fb, fa, dim=1)
            d_sorted = drift.sort().values
            n_d = d_sorted.numel()
            p95_drift = d_sorted[min(n_d - 1, int(0.95 * n_d))].item()
            print(f"[TIDR-FeatureDiag] client={self.id} task={self.task_id} "
                  f"mean_old_feature_drift={drift.mean().item():.6f} "
                  f"max_old_feature_drift={drift.max().item():.6f} "
                  f"p95_old_feature_drift={p95_drift:.6f}")

        # ---- 3b) RetrievedAnchorDrift: P2 窗口 retrieved anchor 表征漂移 ----
        # raw Anchor 参数没坏 ≠ 送进 head 的检索表征没坏（attn 分布变化
        # 会使 soft_anchor 混入新任务 Anchor）
        if self._diag_anchor_feat_before_phase2 is not None:
            anchor_feat_after = self._diag_extract_anchor_feat()
            self._diag_anchor_feat_after_phase2 = anchor_feat_after
            ab = self._diag_anchor_feat_before_phase2.to(self.device)
            aa = anchor_feat_after.to(self.device)
            af_cos = 1.0 - F.cosine_similarity(ab, aa, dim=1)
            af_nr = aa.norm(dim=1) / ab.norm(dim=1).clamp_min(1e-8)
            af_l2 = (aa - ab).norm(dim=1)
            print(f"[TIDR-RetrievedAnchorDrift] client={self.id} "
                  f"task={self.task_id} "
                  f"mean_retrieved_anchor_cos_drift_p2={af_cos.mean().item():.6f} "
                  f"max_retrieved_anchor_cos_drift_p2={af_cos.max().item():.6f} "
                  f"mean_retrieved_anchor_norm_ratio_p2={af_nr.mean().item():.6f} "
                  f"mean_retrieved_anchor_l2_drift_p2={af_l2.mean().item():.6f}")

        # ---- 3c) HeadDrift: model.head / heads[t] 漂移 ----
        # 注意解释口径: Task0 评估用的是冻结快照 heads[0]，不是当前
        # model.head（Phase2 正在训练的当前任务 head）。
        # 因此 task0_snapshot_head_change 应严格为 0（若非 0 = 快照被
        # 意外篡改，是 bug 信号）；current_train_head_change 大本身
        # 不能解释 Task0 遗忘，只反映当前任务 head 的正常训练幅度。
        if self._diag_head_pre_p1 is not None:
            head_now = self.model.head.state_dict()
            head_max_total = 0.0
            for k, v in self._diag_head_pre_p1.items():
                head_max_total = max(
                    head_max_total,
                    (head_now[k].detach().cpu() - v).abs().max().item()
                )
            task0_head_change = 0.0
            old_heads_max_total = 0.0
            for t, sd in (self._diag_heads_pre_p1 or {}).items():
                cur = self.heads[t].state_dict()
                for k, v in sd.items():
                    ch = (cur[k].detach().cpu() - v).abs().max().item()
                    if t == 0:
                        task0_head_change = max(task0_head_change, ch)
                    old_heads_max_total = max(old_heads_max_total, ch)
            print(f"[TIDR-HeadDriftDiag] client={self.id} task={self.task_id} "
                  f"current_train_head_change={head_max_total:.6f} "
                  f"task0_snapshot_head_change={task0_head_change:.6f} "
                  f"all_old_snapshot_heads_max_change={old_heads_max_total:.6f}")

        # ---- 4) Routing stats（Normal） ----
        acc_normal = acc0_post
        metrics_normal = self._format_diag_stats(stats0_post) if stats0_post else None
        if metrics_normal:
            self._print_diag_stats('TIDR-Diag-Normal', self.id, 0, metrics_normal)

        key_after = self.model.key.data.clone().cpu()
        self._diag_key_after_phase2 = key_after

        # ---- 4e) RetrievalCF: 检索污染反事实（诊断专用，非算法） ----
        # 四组掩码（评 Task0；softmax 自然重归一化）:
        #   Seen-only:           只允许 T0..T{task_id}（含当前任务）
        #                        → Normal 的 recovery = future unseen 污染
        #   Exclude-current-only: 允许全部类别但屏蔽当前任务类
        #                        → recovery = 当前任务（如 T2）单独污染
        #                          （保留 future，与 prev-seen-only 区分）
        #   Previous-seen-only:  只允许 union(class_mask[0..task_id-1])
        #                        → recovery = 当前任务 + future 联合污染
        #                        （语义上严格于 old_seen_classes = seen-current，
        #                          任务间共享类时两者不等价）
        #   T0-only:             只允许 Task0 keys/anchors
        # 实现方式: 临时开启 use_seen_routing 并替换 seen_class_mask
        # （try/finally 完整恢复，不改变训练行为）
        prev_seen_classes = sorted(set().union(*[
            set(int(c) for c in self.class_mask[k])
            for k in range(0, self.task_id)
        ])) if self.task_id > 0 else []
        if len(prev_seen_classes) > 0:
            cf_parts = [f"acc_normal={acc_normal:.2f}"]

            # Seen-only: T0..T{task_id}（future unseen 存在时才有意义）
            seen_classes_cf = sorted(set().union(*[
                set(int(c) for c in self.class_mask[k])
                for k in range(0, self.task_id + 1)
            ]))
            if len(seen_classes_cf) < self.nb_classes:
                seen_only_mask = torch.zeros(
                    self.nb_classes, dtype=torch.bool)
                seen_only_mask[torch.tensor(
                    seen_classes_cf, dtype=torch.long)] = True
                acc_seen_only, _ = self._diag_eval_task(
                    0, collect_stats=False,
                    retrieval_allowed_mask=seen_only_mask
                )
                cf_parts.append(
                    f"acc_seen_only={acc_seen_only:.2f} "
                    f"recovery_seen_only={acc_seen_only - acc_normal:+.2f}")

            # Exclude-current-only: 全部类别 - 当前任务类（保留 future）
            # 口径 = current-exclusive（C_current - union(C_0..C_cur-1)）:
            # 互斥任务下与 "C_current - C_eval" 等价；跨任务共享类数据集
            # 下更严格——已在早期任务出现过的共享类不会被误算成
            # "当前任务单独污染"（它们仍被允许检索）。
            prev_cls_cf = set().union(*[
                set(int(c) for c in self.class_mask[k])
                for k in range(0, self.task_id)
            ]) if self.task_id > 0 else set()
            cur_only_classes = sorted(
                int(c) for c in self.class_mask[self.task_id]
                if int(c) not in prev_cls_cf
            )
            if len(cur_only_classes) > 0:
                excl_cur_mask = torch.ones(
                    self.nb_classes, dtype=torch.bool)
                excl_cur_mask[torch.tensor(
                    cur_only_classes, dtype=torch.long)] = False
                acc_excl_cur, _ = self._diag_eval_task(
                    0, collect_stats=False,
                    retrieval_allowed_mask=excl_cur_mask
                )
                cf_parts.append(
                    f"acc_exclude_current={acc_excl_cur:.2f} "
                    f"recovery_exclude_current={acc_excl_cur - acc_normal:+.2f}")

            # Previous-seen-only
            old_only_mask = torch.zeros(self.nb_classes, dtype=torch.bool)
            old_only_mask[torch.tensor(
                prev_seen_classes, dtype=torch.long)] = True
            acc_old_only, _ = self._diag_eval_task(
                0, collect_stats=False,
                retrieval_allowed_mask=old_only_mask
            )
            cf_parts.append(
                f"acc_prev_seen_only={acc_old_only:.2f} "
                f"recovery_prev_seen_only={acc_old_only - acc_normal:+.2f}")

            # T0-only
            t0_only_mask = torch.zeros(self.nb_classes, dtype=torch.bool)
            t0_only_mask[torch.tensor(
                self.class_mask[0], dtype=torch.long)] = True
            acc_t0_only, _ = self._diag_eval_task(
                0, collect_stats=False,
                retrieval_allowed_mask=t0_only_mask
            )
            cf_parts.append(
                f"acc_t0_only={acc_t0_only:.2f} "
                f"recovery_t0_only={acc_t0_only - acc_normal:+.2f}")

            print(f"[TIDR-RetrievalCF] client={self.id} task={self.task_id} "
                  + ' '.join(cf_parts))

        # ============================================================
        # 以下为完整反事实套件（仅 CF 轮 / standalone）
        # ============================================================
        if not cf:
            # 轻量 PhaseDiag 轮: 跳过 Key/Anchor/Joint CF 与 CP3
            if (self._task_ckpt_config is not None
                    and self._task_ckpt_config.get('save')):
                extras = {
                    'diag_task': self.task_id,
                    'diag_round': round,
                    'key_before_phase2': self._diag_key_before_phase2,
                    'key_after_phase2': key_after,
                    'anchor_before_phase2': self._diag_anchor_before_phase2,
                    'old_key_mask': self._diag_old_key_mask,
                    'phase_acc_pre': dict(self._diag_phase_acc_pre or {}),
                    'phase_acc_pre_p1': dict(self._diag_phase_acc_pre_p1 or {}),
                    'phase_acc_post': dict(post_accs),
                    'feature_before': self._diag_feature_before,
                    'feature_after': self._diag_feature_after,
                    'feature_sample_indices': list(self._diag_feature_samples or []),
                    'anchor_feat_after_phase2': self._diag_anchor_feat_after_phase2,
                }
                extras.update(self._diag_snapshot_extras())
                save_checkpoint(
                    self._task_ckpt_config['server'], 'Task_C0_post_phase2',
                    f"Task{self.task_id}_C0_post_phase2.pth", round, extras
                )
            return

        # ---- 4a) KeyCF: Old-Key Rollback 反事实实验 ----
        acc_rollback, recovery, restoration_pass = self._diag_key_rollback(acc_normal)
        print(f"[TIDR-KeyCF] client={self.id} task={self.task_id} "
              f"acc_normal={acc_normal:.2f} acc_rollback={acc_rollback:.2f} "
              f"recovery={recovery:+.2f} key_restoration_pass={restoration_pass}")

        # ---- 4b) AnchorDrift: 旧 Anchor Phase2 前后漂移度量 ----
        # anchor_before 来源优先级: phase2_start 快照 > prev_anchor_pool
        # （prev_anchor_pool 在本轮开始时 clone 保存，Phase1 只优化 ViT/Prompt，
        #   不修改 Tail Anchor，因此等价于 Phase2 前 Anchor）
        anchor_before = self._diag_anchor_before_phase2
        if anchor_before is None and self.prev_anchor_pool is not None:
            anchor_before = self.prev_anchor_pool
            self._diag_anchor_before_phase2 = anchor_before

        if anchor_before is not None and len(self.old_seen_classes) > 0:
            old_idx_a = torch.tensor(
                sorted(self.old_seen_classes),
                dtype=torch.long,
                device=self.device
            )
            ab = anchor_before.to(self.device).index_select(0, old_idx_a)
            aa = self.model.anchor_pool.detach().index_select(0, old_idx_a)

            cos_drift = 1.0 - F.cosine_similarity(ab, aa, dim=1)
            cd_sorted = cos_drift.sort().values
            n_a = cd_sorted.numel()
            norm_ratio = aa.norm(dim=1) / ab.norm(dim=1).clamp_min(1e-8)
            nr_sorted = norm_ratio.sort().values
            l2_drift = (aa - ab).norm(dim=1)

            print(f"[TIDR-AnchorDrift] client={self.id} task={self.task_id} "
                  f"mean_old_anchor_cos_drift={cos_drift.mean().item():.6f} "
                  f"median_old_anchor_cos_drift={cd_sorted[n_a // 2].item():.6f} "
                  f"p95_old_anchor_cos_drift={cd_sorted[min(n_a - 1, int(0.95 * n_a))].item():.6f} "
                  f"max_old_anchor_cos_drift={cos_drift.max().item():.6f} "
                  f"mean_old_anchor_norm_ratio={norm_ratio.mean().item():.6f} "
                  f"median_old_anchor_norm_ratio={nr_sorted[n_a // 2].item():.6f} "
                  f"max_abs_anchor_norm_change={((aa.norm(dim=1) - ab.norm(dim=1)).abs().max()).item():.6f} "
                  f"mean_old_anchor_l2_drift={l2_drift.mean().item():.6f}")

        # ---- 4c) Anchor 反事实组: Full / Norm-only / Direction-only ----
        if anchor_before is not None and len(self.old_seen_classes) > 0:
            acc_arb, a_recovery, a_restoration, _ = self._diag_anchor_cf(
                acc_normal, anchor_before, 'full'
            )
            print(f"[TIDR-AnchorCF] client={self.id} task={self.task_id} "
                  f"acc_normal={acc_normal:.2f} acc_anchor_rollback={acc_arb:.2f} "
                  f"recovery={a_recovery:+.2f} anchor_restoration_pass={a_restoration}")

            # NormCF: 方向保持 Phase2 后，只恢复 Phase2 前范数
            acc_ncf, n_recovery, n_restoration, _ = self._diag_anchor_cf(
                acc_normal, anchor_before, 'norm'
            )
            print(f"[TIDR-NormCF] client={self.id} task={self.task_id} "
                  f"acc_normal={acc_normal:.2f} acc_norm_rollback={acc_ncf:.2f} "
                  f"recovery={n_recovery:+.2f} anchor_restoration_pass={n_restoration}")

            # DirCF: 范数保持 Phase2 后，只恢复 Phase2 前方向
            acc_dcf, d_recovery, d_restoration, _ = self._diag_anchor_cf(
                acc_normal, anchor_before, 'dir'
            )
            print(f"[TIDR-DirCF] client={self.id} task={self.task_id} "
                  f"acc_normal={acc_normal:.2f} acc_direction_rollback={acc_dcf:.2f} "
                  f"recovery={d_recovery:+.2f} anchor_restoration_pass={d_restoration}")

            # JointCF: 旧 Key + 旧 Anchor 同时回滚（检测非线性交互）
            if self._diag_key_before_phase2 is not None:
                acc_jcf, j_recovery, j_key_pass, j_anchor_pass = (
                    self._diag_joint_cf(
                        acc_normal,
                        self._diag_key_before_phase2,
                        anchor_before
                    )
                )
                print(f"[TIDR-JointCF] client={self.id} task={self.task_id} "
                      f"acc_normal={acc_normal:.2f} acc_joint_rollback={acc_jcf:.2f} "
                      f"recovery={j_recovery:+.2f} "
                      f"key_restoration_pass={j_key_pass} "
                      f"anchor_restoration_pass={j_anchor_pass}")

        # ---- 5) Classwise CSV（原始数据，不做因果解释） ----
        try:
            self._diag_write_classwise_csv(self._diag_phase_stats_pre, stats0_post)
        except Exception as e:
            print(f"[TIDR-ClasswiseDiag] CSV 写入失败: {e}")

        # ---- 6) CP3 保存（standalone 模式不重复保存） ----
        if cfg.get('save') and not standalone:
            extras = {
                'key_before_phase2': self._diag_key_before_phase2,
                'key_after_phase2': key_after,
                'anchor_before_phase2': self._diag_anchor_before_phase2,
                'old_key_mask': self._diag_old_key_mask,
                'phase_acc_pre': dict(self._diag_phase_acc_pre or {}),
                'phase_acc_pre_p1': dict(self._diag_phase_acc_pre_p1 or {}),
                'phase_acc_post': dict(post_accs),
                'phase_stats_pre': self._diag_phase_stats_pre,
                'feature_before': self._diag_feature_before,
                'feature_after': self._diag_feature_after,
                'feature_sample_indices': list(self._diag_feature_samples or []),
                'anchor_feat_after_phase2': self._diag_anchor_feat_after_phase2,
            }
            extras.update(self._diag_snapshot_extras())
            save_checkpoint(cfg['server'], 'R5_C0_post_phase2',
                            f'R{round}_C0_post_phase2.pth', round, extras)

        # ---- 6b) Task Checkpoint: Task{k}_C0_post_phase2.pth ----
        if (self._task_ckpt_config is not None
                and self._task_ckpt_config.get('save')
                and not standalone):
            extras = {
                'diag_task': self.task_id,
                'diag_round': round,
                'key_before_phase2': self._diag_key_before_phase2,
                'key_after_phase2': key_after,
                'anchor_before_phase2': self._diag_anchor_before_phase2,
                'old_key_mask': self._diag_old_key_mask,
                'phase_acc_pre': dict(self._diag_phase_acc_pre or {}),
                'phase_acc_pre_p1': dict(self._diag_phase_acc_pre_p1 or {}),
                'phase_acc_post': dict(post_accs),
                'feature_before': self._diag_feature_before,
                'feature_after': self._diag_feature_after,
                'feature_sample_indices': list(self._diag_feature_samples or []),
                'anchor_feat_after_phase2': self._diag_anchor_feat_after_phase2,
            }
            extras.update(self._diag_snapshot_extras())
            save_checkpoint(
                self._task_ckpt_config['server'], 'Task_C0_post_phase2',
                f"Task{self.task_id}_C0_post_phase2.pth", round, extras
            )

        if cfg.get('stop_at') == 'R5_C0_post_phase2' and not standalone:
            raise DiagStopException('R5_C0_post_phase2')

    def _diag_key_rollback(self, acc_normal):
        """
        E6a-v2-Diag: Old-Key Rollback 反事实实验。

        控制变量：
        - Prompt 不变
        - Feature representation 不变
        - Head 不变
        - Anchor 不变
        - New-task keys 不变
        - 只恢复 old-task-exclusive keys（old_seen_classes 定义，共享类已排除）

        结束后无条件恢复完整 Phase2 后 Key。
        """
        if (
            self._diag_key_before_phase2 is None
            or len(self.old_seen_classes) == 0
        ):
            return float('nan'), float('nan'), False

        old_idx = torch.tensor(
            sorted(self.old_seen_classes),
            dtype=torch.long,
            device=self.device
        )

        # Phase2 后完整 Key 备份
        key_backup = (
            self.model.key
            .detach()
            .clone()
        )

        acc_rollback = float('nan')
        stats_rb = None

        try:
            # ========================================================
            # 仅恢复旧任务专属 Key
            # 注意：advanced indexing getitem 返回副本，必须用 setitem
            # （self.model.key[old_idx].copy_(...) 不会写回 Parameter）
            # ========================================================
            with torch.no_grad():
                key_before = (
                    self._diag_key_before_phase2
                    .to(
                        device=self.device,
                        dtype=self.model.key.dtype
                    )
                )

                self.model.key[old_idx] = (
                    key_before[old_idx]
                )

                # 写回验证：确认旧 Key 行确实恢复
                assert torch.equal(
                    self.model.key.data[old_idx],
                    key_before[old_idx]
                ), "[TIDR-KeyCF] rollback write-back failed"

            # ========================================================
            # Rollback 后重新评估 Task0
            # ========================================================
            acc_rollback, stats_rb = (
                self._diag_eval_task(
                    0,
                    collect_stats=True
                )
            )

            if stats_rb:
                self._print_diag_stats(
                    'TIDR-Diag-Rollback',
                    self.id,
                    0,
                    self._format_diag_stats(stats_rb)
                )

        finally:
            # ========================================================
            # 无论诊断是否成功，恢复完整 Phase2 后 Key
            # ========================================================
            with torch.no_grad():
                self.model.key.copy_(
                    key_backup
                )

        # ============================================================
        # 严格检查恢复是否成功
        # ============================================================
        restoration_pass = bool(
            torch.equal(
                self.model.key.detach(),
                key_backup
            )
        )

        recovery = (
            acc_rollback - acc_normal
            if not np.isnan(acc_rollback)
            else float('nan')
        )

        return (
            acc_rollback,
            recovery,
            restoration_pass
        )

    def _diag_anchor_cf(self, acc_normal, anchor_before, mode):
        """
        E6a-v2-Diag: Anchor 反事实实验通用实现（Full / Norm-only / Dir-only）。

        控制变量：
        - Key / Head / Prompt / Feature 全部保持 Phase2 后状态
        - 只修改 old_seen_classes 对应的 anchor_pool 行

        mode:
          'full' — 完整恢复 Phase2 前旧 Anchor（方向 + 范数）
          'norm' — 方向保持 Phase2 后，只恢复 Phase2 前范数
                   （钉死"范数坍缩"是否为主要遗忘机制）
          'dir'  — 范数保持 Phase2 后，只恢复 Phase2 前方向
                   （方向因素的反事实对照）

        写回使用 index_copy_（in-place），避免 advanced-indexing getitem
        返回副本导致回滚不生效的陷阱（KeyCF 曾踩过）。

        Returns:
            (acc_cf, recovery, anchor_restoration_pass, max_abs_change)
        """
        if anchor_before is None or len(self.old_seen_classes) == 0:
            return float('nan'), float('nan'), False, 0.0

        old_idx = torch.tensor(
            sorted(self.old_seen_classes),
            dtype=torch.long,
            device=self.device
        )

        # Phase2 后完整 anchor_pool 备份
        anchor_backup = (
            self.model.anchor_pool
            .detach()
            .clone()
        )

        acc_cf = float('nan')
        stats_cf = None
        rollback_applied = False
        max_abs_change = 0.0

        try:
            # ========================================================
            # 构造反事实 Anchor 并写回旧类行
            # ========================================================
            with torch.no_grad():
                ab = (
                    anchor_before
                    .to(
                        device=self.device,
                        dtype=self.model.anchor_pool.dtype
                    )
                    .index_select(0, old_idx)
                )
                aa = self.model.anchor_pool.detach().index_select(0, old_idx)

                if mode == 'full':
                    cf_vals = ab
                elif mode == 'norm':
                    # 方向: Phase2 后; 范数: Phase2 前
                    aa_dir = F.normalize(aa, dim=1)
                    ab_norm = ab.norm(dim=1, keepdim=True)
                    cf_vals = aa_dir * ab_norm
                elif mode == 'dir':
                    # 方向: Phase2 前; 范数: Phase2 后
                    ab_dir = F.normalize(ab, dim=1)
                    aa_norm = aa.norm(dim=1, keepdim=True)
                    cf_vals = ab_dir * aa_norm
                else:
                    raise ValueError(f"unknown mode: {mode}")

                self.model.anchor_pool.index_copy_(
                    0,
                    old_idx,
                    cf_vals
                )

                # 真实性检查 1: 反事实值确实写回
                rollback_applied = bool(
                    torch.equal(
                        self.model.anchor_pool.index_select(0, old_idx),
                        cf_vals
                    )
                )

                # 真实性检查 2: 旧行相对 Phase2 后确实发生了变化
                max_abs_change = (
                    cf_vals - aa
                ).abs().max().item()

            print(f"[TIDR-AnchorCF-Check] mode={mode} "
                  f"rollback_applied={rollback_applied} "
                  f"max_abs_change={max_abs_change:.6f}")

            # 保险: 写回失败立即中止，绝不允许"日志正常但实际没回滚"
            assert rollback_applied, \
                f"[TIDR-AnchorCF] rollback write-back failed (mode={mode})"

            # ========================================================
            # 反事实状态下重新评估 Task0
            # ========================================================
            acc_cf, stats_cf = (
                self._diag_eval_task(
                    0,
                    collect_stats=True
                )
            )

            if stats_cf:
                self._print_diag_stats(
                    f'TIDR-Diag-AnchorCF-{mode}',
                    self.id,
                    0,
                    self._format_diag_stats(stats_cf)
                )

        finally:
            # ========================================================
            # 无论诊断是否成功，恢复完整 Phase2 后 anchor_pool
            # ========================================================
            with torch.no_grad():
                self.model.anchor_pool.copy_(
                    anchor_backup
                )

        # ============================================================
        # 严格检查恢复是否成功
        # ============================================================
        restoration_pass = bool(
            torch.equal(
                self.model.anchor_pool.detach(),
                anchor_backup
            )
        )

        recovery = (
            acc_cf - acc_normal
            if not np.isnan(acc_cf)
            else float('nan')
        )

        return (
            acc_cf,
            recovery,
            restoration_pass,
            max_abs_change
        )

    def _diag_joint_cf(self, acc_normal, key_before, anchor_before):
        """
        E6a-v2-Diag: Key + Anchor Joint Rollback 反事实实验。

        同时恢复旧 Key 行和旧 Anchor 行到 Phase2 前，
        检测 Key 与 Anchor 的非线性交互对剩余遗忘缺口的贡献
        （Full AnchorCF 89.13 vs pre 96.74，剩余 ~7.6pp 的来源）。

        Returns:
            (acc_joint, recovery, key_restoration_pass, anchor_restoration_pass)
        """
        if (key_before is None or anchor_before is None
                or len(self.old_seen_classes) == 0):
            return float('nan'), float('nan'), False, False

        old_idx = torch.tensor(
            sorted(self.old_seen_classes),
            dtype=torch.long,
            device=self.device
        )

        key_backup = self.model.key.detach().clone()
        anchor_backup = self.model.anchor_pool.detach().clone()

        acc_joint = float('nan')
        stats_j = None

        try:
            with torch.no_grad():
                kb = (
                    key_before
                    .to(
                        device=self.device,
                        dtype=self.model.key.dtype
                    )
                    .index_select(0, old_idx)
                )
                ab = (
                    anchor_before
                    .to(
                        device=self.device,
                        dtype=self.model.anchor_pool.dtype
                    )
                    .index_select(0, old_idx)
                )

                self.model.key.index_copy_(0, old_idx, kb)
                self.model.anchor_pool.index_copy_(0, old_idx, ab)

                applied = bool(
                    torch.equal(
                        self.model.key.index_select(0, old_idx), kb
                    )
                    and torch.equal(
                        self.model.anchor_pool.index_select(0, old_idx), ab
                    )
                )

            print(f"[TIDR-JointCF-Check] rollback_applied={applied}")
            assert applied, "[TIDR-JointCF] rollback write-back failed"

            acc_joint, stats_j = self._diag_eval_task(
                0,
                collect_stats=True
            )

            if stats_j:
                self._print_diag_stats(
                    'TIDR-Diag-JointRollback',
                    self.id,
                    0,
                    self._format_diag_stats(stats_j)
                )

        finally:
            with torch.no_grad():
                self.model.key.copy_(key_backup)
                self.model.anchor_pool.copy_(anchor_backup)

        key_pass = bool(
            torch.equal(self.model.key.detach(), key_backup)
        )
        anchor_pass = bool(
            torch.equal(self.model.anchor_pool.detach(), anchor_backup)
        )

        recovery = (
            acc_joint - acc_normal
            if not np.isnan(acc_joint)
            else float('nan')
        )

        return (
            acc_joint,
            recovery,
            key_pass,
            anchor_pass
        )

    def _diag_eval_task(self, task, collect_stats=False,
                        retrieval_allowed_mask=None):
        """
        E6a-v2-Diag: 只读、可恢复的确定性单任务评估。

        关键原则：
        1. 不替换 self.model.head 模块对象（保护 Phase2 optimizer 的参数引用）；
        2. 诊断结束后恢复 model / vit 的 train-eval 状态；
        3. 恢复 RNG，避免诊断改变后续训练数据顺序和随机增强；
        4. DataLoader 使用 shuffle=False；
        5. 评估协议仍保持：vit(task_id=self.task_id, train=True) + heads[task]。

        E6a-v3b-Diag: retrieval_allowed_mask（诊断专用，默认 None = 正常评估）
        传入 bool [nb_classes] 掩码时，临时开启 use_seen_routing 并替换
        seen_class_mask，使 hard/soft 检索只允许掩码内的 keys/anchors
        （softmax 自然重归一化）。finally 中完整恢复原状态。

        Returns:
            (acc, stats); stats 仅在 collect_stats 且 task < self.task_id 时填充
        """
        # ============================================================
        # 1. 保存进入诊断前的运行状态
        # ============================================================
        model_was_training = self.model.training
        vit_was_training = self.vit.training

        if self.original_model is not None:
            original_model_was_training = self.original_model.training
        else:
            original_model_was_training = None

        # 诊断不能改变后续训练随机轨迹
        rng_backup = get_rng_state()

        # E6a-v3b-Diag r3: 固定诊断 seed（与 task 绑定）
        # test_loader[task] 若继承训练集的随机 crop/flip transform，
        # 三点 accuracy（post_task_switch_pre_phase1 / post_phase1 /
        # post_phase2）与各 CF 会分别在不同增强实例上测量。RNG restore
        # 只保证"诊断不影响训练"，不保证"不同时刻的诊断看到相同图片"。
        # 用固定 seed 使同一 task 的每次诊断评估产生完全相同的增强序列；
        # finally 中 set_rng_state 恢复真实训练 RNG。
        _diag_seed = TIDR_DIAG_EVAL_SEED + int(task)
        torch.manual_seed(_diag_seed)
        np.random.seed(_diag_seed % (2 ** 32))
        random.seed(_diag_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_diag_seed)

        # ============================================================
        # 2. 原位保存当前 head 参数（只保存 state_dict，不替换 head 对象）
        # ============================================================
        head_backup = {
            k: v.detach().clone()
            for k, v in self.model.head.state_dict().items()
        }

        # E6a-v3b-Diag: Retrieval CF 掩码备份（None = 不干预检索）
        retrieval_backup = None
        if retrieval_allowed_mask is not None:
            retrieval_backup = (
                self.model.use_seen_routing,
                self.model.seen_class_mask.detach().clone(),
            )

        try:
            # --------------------------------------------------------
            # 保证 standalone CP3 时设备正确
            # --------------------------------------------------------
            self.model.to(self.device)
            self.vit.to(self.device)

            if self.original_model is not None:
                self.original_model.to(self.device)
                self.original_model.eval()

            # Tail Anchor 进入 eval
            self.model.eval()

            # --------------------------------------------------------
            # E6a-v3b-Diag: 应用检索掩码（Old-only / T0-only）
            # 借用 use_seen_routing 机制: masked_fill(-inf) 限制
            # hard route + soft attention 只在允许的 keys/anchors 上
            # --------------------------------------------------------
            mask_verify = None
            allowed_idx_dev = None
            if retrieval_allowed_mask is not None:
                self.model.use_seen_routing = True
                self.model.seen_class_mask.copy_(
                    retrieval_allowed_mask.to(
                        device=self.model.seen_class_mask.device,
                        dtype=self.model.seen_class_mask.dtype
                    )
                )
                # ---- v3b-Diag r5: RetrievalCF sanity check ----
                # 四组不同 mask 曾给出完全相同的异常准确率（2.17%），
                # 无法静态定位 → 评估过程中直接验证模型实际看到的状态:
                #   mask_copy_pass:            copy_ 后 seen_class_mask 是否
                #                              与预期掩码逐位一致
                #   hard_idx_in_allowed_rate:  应 = 1.0000（hard route 全部
                #                              落在允许集合内）
                #   mean_finite_routing_logits: 应 = allowed_count（被
                #                              mask_fill(-inf) 的列数正确）
                #   soft_attention_allowed_mass: 应 ≈ 1.0000（softmax 后
                #                              允许集合内的概率质量）
                #   allowed_digest:            允许类别索引的 md5 短摘要
                #                              （四组 mask 应各不相同）
                allowed_idx = retrieval_allowed_mask.to(
                    torch.bool).nonzero(as_tuple=False).squeeze(-1)
                mask_verify = {
                    'allowed_count': int(allowed_idx.numel()),
                    'allowed_digest': hashlib.md5(
                        allowed_idx.cpu().numpy().tobytes()
                    ).hexdigest()[:8],
                    'mask_copy_pass': torch.equal(
                        self.model.seen_class_mask.detach().cpu().bool(),
                        retrieval_allowed_mask.to(torch.bool).cpu()
                    ),
                    'n': 0,
                    'hard_in_allowed': 0.0,
                    'finite_logits_sum': 0.0,
                    'soft_allowed_mass_sum': 0.0,
                }
                allowed_idx_dev = allowed_idx.to(self.device)

            # --------------------------------------------------------
            # 临时加载被评估任务对应的 head
            # 关键：原位 load_state_dict，不使用 self.model.load_head()
            # --------------------------------------------------------
            if self.heads[task] is not None:
                self.model.head.load_state_dict(
                    self.heads[task].state_dict()
                )

            # --------------------------------------------------------
            # 固定样本顺序
            # --------------------------------------------------------
            test_data = self.test_loader[task]

            loader = DataLoader(
                test_data,
                batch_size=8,
                shuffle=False
            )

            # GPC/TIDR 所需状态
            proto_bank, proto_valid = (
                self._build_semantic_proto_bank()
            )

            proto_calib_mask = (
                self._build_proto_calib_mask(task)
            )

            do_stats = (
                collect_stats
                and task < self.task_id
            )

            stats = (
                self._init_diag_stats(task)
                if do_stats
                else None
            )

            correct = 0
            total = 0

            # ========================================================
            # 3. 正式评估
            # ========================================================
            with torch.no_grad():

                for input, target in loader:

                    input = input.to(
                        self.device,
                        non_blocking=True
                    )

                    target = target.to(
                        self.device,
                        non_blocking=True
                    )

                    # -----------------------------------------------
                    # frozen original ViT features
                    # -----------------------------------------------
                    if self.original_model is not None:
                        output = self.original_model(input)
                        cls_features = (
                            output['pre_logits']
                            .requires_grad_(False)
                        )
                    else:
                        cls_features = None

                    # -----------------------------------------------
                    # 与当前 evaluate() 保持同一协议：
                    # 旧任务评估仍使用当前 task-stage prompt
                    # -----------------------------------------------
                    output = self.vit(
                        input,
                        task_id=self.task_id,
                        cls_features=cls_features,
                        train=True
                    )

                    feat = output['feat'].to(self.device)

                    # -----------------------------------------------
                    # Tail Anchor inference
                    # （v3b-Diag r5: 第 5 个返回值 attn_weights 即真实
                    #   soft_attn，用于 group-wise soft attention mass；
                    #   第 7 个 routing_logits 用于 RetrievalCF sanity
                    #   check 的 finite 列数验证）
                    # -----------------------------------------------
                    (
                        pre,
                        _,
                        _,
                        _,
                        attn_weights,
                        hard_idx,
                        routing_logits
                    ) = self.model(
                        feat,
                        target,
                        proto_bank=proto_bank,
                        proto_valid_mask=proto_valid,
                        proto_calib_mask=proto_calib_mask
                    )

                    logits = pre

                    # -----------------------------------------------
                    # Task-incremental class mask
                    # -----------------------------------------------
                    mask = self.class_mask[task]

                    not_mask = np.setdiff1d(
                        np.arange(self.nb_classes),
                        mask
                    )

                    not_mask = torch.tensor(
                        not_mask,
                        dtype=torch.int64,
                        device=self.device
                    )

                    logits = logits.index_fill(
                        dim=1,
                        index=not_mask,
                        value=float('-inf')
                    )

                    predicts = torch.max(
                        logits,
                        dim=1
                    )[1]

                    correct += (
                        predicts == target
                    ).sum().item()

                    total += len(target)

                    # -----------------------------------------------
                    # TIDR routing diagnostics
                    # -----------------------------------------------
                    if do_stats:
                        similarity = (
                            self.model.compute_similarity(feat)
                        )

                        self._accumulate_diag_stats(
                            stats,
                            similarity,
                            hard_idx,
                            predicts,
                            target,
                            task,
                            attn_weights=attn_weights
                        )

                    # -----------------------------------------------
                    # v3b-Diag r5: RetrievalCF sanity check 累积
                    # -----------------------------------------------
                    if mask_verify is not None:
                        mask_verify['n'] += len(target)
                        mask_verify['hard_in_allowed'] += torch.isin(
                            hard_idx, allowed_idx_dev
                        ).float().sum().item()
                        mask_verify['finite_logits_sum'] += (
                            torch.isfinite(routing_logits)
                            .sum().item()
                        )
                        mask_verify['soft_allowed_mass_sum'] += (
                            attn_weights[:, allowed_idx_dev]
                            .sum().item()
                        )

            acc = (
                100.0 * correct / max(total, 1)
            )

            # ---- v3b-Diag r5: RetrievalCF sanity check 输出 ----
            if mask_verify is not None and mask_verify['n'] > 0:
                n_v = mask_verify['n']
                print(f"[TIDR-RetrievalCF-Check] client={self.id} "
                      f"task={task} "
                      f"allowed_count={mask_verify['allowed_count']} "
                      f"allowed_digest={mask_verify['allowed_digest']} "
                      f"mask_copy_pass={mask_verify['mask_copy_pass']} "
                      f"hard_idx_in_allowed_rate="
                      f"{mask_verify['hard_in_allowed'] / n_v:.4f} "
                      f"mean_finite_routing_logits="
                      f"{mask_verify['finite_logits_sum'] / n_v:.4f} "
                      f"soft_attention_allowed_mass="
                      f"{mask_verify['soft_allowed_mass_sum'] / n_v:.4f}")

            return acc, stats

        finally:
            # ========================================================
            # 4. 无论诊断成功还是异常，都必须恢复状态
            # ========================================================

            # 恢复检索掩码（E6a-v3b-Diag: Retrieval CF）
            if retrieval_backup is not None:
                self.model.use_seen_routing = retrieval_backup[0]
                self.model.seen_class_mask.copy_(
                    retrieval_backup[1]
                )

            # 原位恢复当前 head 参数
            self.model.head.load_state_dict(
                head_backup
            )

            # 恢复进入诊断前的 train/eval 状态
            self.model.train(model_was_training)
            self.vit.train(vit_was_training)

            if (
                self.original_model is not None
                and original_model_was_training is not None
            ):
                self.original_model.train(
                    original_model_was_training
                )

            # 最后恢复 RNG
            set_rng_state(rng_backup)

    def _init_diag_stats(self, task):
        """E6a-v2-Diag: 初始化单次评估的统计累积器"""
        eval_classes = [int(c) for c in self.class_mask[task]]
        eval_class_set = set(eval_classes)
        # new-only 定义与 evaluate() TIDR-Diag 一致: 当前任务类 - 被评估任务类
        new_only_classes = [int(c) for c in self.class_mask[self.task_id]
                            if int(c) not in eval_class_set]

        # ---- E6a-v3b-Diag: cross-task routing 分组 ----
        # old_to_new_collision 只统计 T_eval → 当前任务，跨任务累积碰撞
        # （如 T0 → T1）完全没算进去——R9 与 R10 的该指标不可直接比较。
        # 因此新增: 每个其他已见任务 k 的 to_task{k}（collision + margin），
        # 以及 to_any_non_eval（全部已见任务类 - 被评估任务类）。
        # v3b-Diag r2: use_seen_routing=False 时 hard/soft 可访问全部
        # nb_classes（含未来 Task3/4 尚未 seen 的 keys），T_eval → future
        # 的碰撞未被上述任何组覆盖，故再增 future_unseen 组与
        # all_non_eval（全部类别 - 被评估任务类）组。
        per_task_classes = {}
        all_seen_classes = set()
        if task < self.task_id:
            for k in range(0, self.task_id + 1):
                for c in self.class_mask[k]:
                    all_seen_classes.add(int(c))
                if k == task:
                    continue
                cls_k = sorted(
                    int(c) for c in self.class_mask[k]
                    if int(c) not in eval_class_set
                )
                if cls_k:
                    per_task_classes[k] = cls_k
        any_non_eval_classes = sorted(all_seen_classes - eval_class_set)
        all_classes = set(range(self.nb_classes))
        future_unseen_classes = sorted(all_classes - all_seen_classes)
        all_non_eval_classes = sorted(all_classes - eval_class_set)

        return {
            'eval_classes': eval_classes,
            'new_only_classes': new_only_classes,
            'total': 0,
            'wrong_total': 0,
            # old-new（与 evaluate() TIDR-Diag 同定义；即 to_current_task）
            'old_to_new_collision': 0.0,
            'old_new_margin_sum': 0.0,
            'old_new_margin_negative': 0.0,
            # E6a-v3b-Diag: cross-task routing
            'per_task_classes': per_task_classes,
            'any_non_eval_classes': any_non_eval_classes,
            'per_task_collision': {k: 0.0 for k in per_task_classes},
            'per_task_margin_sum': {k: 0.0 for k in per_task_classes},
            'per_task_margin_negative': {k: 0.0 for k in per_task_classes},
            'any_collision': 0.0,
            'any_margin_sum': 0.0,
            'any_margin_negative': 0.0,
            # v3b-Diag r2: future-unseen / all-non-eval
            'future_unseen_classes': future_unseen_classes,
            'future_collision': 0.0,
            'future_margin_sum': 0.0,
            'future_margin_negative': 0.0,
            'all_non_eval_classes': all_non_eval_classes,
            'all_collision': 0.0,
            'all_margin_sum': 0.0,
            'all_margin_negative': 0.0,
            # v3b-Diag r5: group-wise soft attention mass（决定性诊断）
            # Phase2 期间 query feature / raw Anchor 均未变而 retrieved
            # anchor 巨幅漂移 → 到底是谁拿走了 soft retrieval mass。
            # 分组口径与 per_task_classes / future_unseen_classes 完全
            # 一致，各组 mass 之和（含被评估任务）= 1.0。
            'soft_mass_eval_sum': 0.0,
            'soft_mass_true_sum': 0.0,
            'soft_mass_per_task_sum': {k: 0.0 for k in per_task_classes},
            'soft_mass_future_sum': 0.0,
            'soft_entropy_sum': 0.0,
            'soft_max_w_sum': 0.0,
            # old-old
            'old_old_top1_correct': 0.0,
            'old_old_margin_sum': 0.0,
            'old_old_margin_negative': 0.0,
            # 样本级错误分解
            'wrong_and_correct_old_route': 0.0,
            'wrong_and_old_old_collision': 0.0,
            'wrong_and_old_new_collision': 0.0,
            'correct_and_non_true_route': 0.0,
            # 类级
            'per_class_correct': torch.zeros(self.nb_classes),
            'per_class_total': torch.zeros(self.nb_classes),
            'per_class_oo_margin_sum': torch.zeros(self.nb_classes),
        }

    def _accumulate_diag_stats(self, stats, similarity, hard_idx, predicts,
                               target, task, attn_weights=None):
        """
        E6a-v2-Diag: 样本级统计累积
        - old-old margin: cos(f, K_y) - max_{c∈旧任务其他类} cos(f, K_c)
        - old-new margin: cos(f, K_y) - max_{c∈new-only 类} cos(f, K_c)
        - 每类一个 Key（Tail_Anchor.key 为 [nb_class, key_size]），
          类别聚合规则与实际推理一致（无需额外聚合）
        - v3b-Diag r5: attn_weights = forward 返回的真实 soft_attn（含
          proto calibration / 温度等实际推理行为），按任务分组累积
          soft attention mass。RetrievalCF 调用 collect_stats=False，
          不会进入此分支，mass 始终是 Normal 口径。
        """
        n = len(target)
        stats['total'] += n

        eval_idx = torch.tensor(stats['eval_classes'], dtype=torch.long,
                                device=similarity.device)
        new_idx = None
        if stats['new_only_classes']:
            new_idx = torch.tensor(stats['new_only_classes'], dtype=torch.long,
                                   device=similarity.device)

        true_score = similarity.gather(1, target.unsqueeze(1)).squeeze(1)

        # ---- old-old ----
        other_sim = similarity.clone()
        other_sim.scatter_(1, target.unsqueeze(1), float('-inf'))
        max_other_old = other_sim[:, eval_idx].max(dim=1).values
        oo_margin = true_score - max_other_old
        stats['old_old_margin_sum'] += oo_margin.sum().item()
        stats['old_old_margin_negative'] += (oo_margin < 0).float().sum().item()

        # old-old top1（被评估旧任务类别集合内部）
        top1_old = eval_idx[similarity[:, eval_idx].argmax(dim=1)]
        stats['old_old_top1_correct'] += (top1_old == target).float().sum().item()

        # ---- old-new ----
        if new_idx is not None:
            stats['old_to_new_collision'] += torch.isin(
                hard_idx, new_idx
            ).float().sum().item()
            max_new_sim = similarity[:, new_idx].max(dim=1).values
            on_margin = true_score - max_new_sim
            stats['old_new_margin_sum'] += on_margin.sum().item()
            stats['old_new_margin_negative'] += (on_margin < 0).float().sum().item()

        # ---- E6a-v3b-Diag: cross-task routing ----
        # T_eval → 每个其他已见任务 k 的 collision / margin
        # + T_eval → any-non-eval（跨任务累积碰撞，old_to_new 不含的部分）
        if stats.get('any_non_eval_classes'):
            any_idx = torch.tensor(
                stats['any_non_eval_classes'], dtype=torch.long,
                device=similarity.device
            )
            stats['any_collision'] += torch.isin(
                hard_idx, any_idx
            ).float().sum().item()
            max_any = similarity[:, any_idx].max(dim=1).values
            any_margin = true_score - max_any
            stats['any_margin_sum'] += any_margin.sum().item()
            stats['any_margin_negative'] += (
                any_margin < 0
            ).float().sum().item()

        for k, cls_k in (stats.get('per_task_classes') or {}).items():
            idx_k = torch.tensor(
                cls_k, dtype=torch.long, device=similarity.device
            )
            stats['per_task_collision'][k] += torch.isin(
                hard_idx, idx_k
            ).float().sum().item()
            max_k = similarity[:, idx_k].max(dim=1).values
            margin_k = true_score - max_k
            stats['per_task_margin_sum'][k] += margin_k.sum().item()
            stats['per_task_margin_negative'][k] += (
                margin_k < 0
            ).float().sum().item()

        # ---- v3b-Diag r2: future-unseen / all-non-eval ----
        # use_seen_routing=False 时检索可落到未来任务 keys 上；
        # future 组单独拆出来，all 组 = 全部类别 - 被评估任务类
        # （= any_non_eval（已见）+ future 的总和口径）。
        if stats.get('future_unseen_classes'):
            fut_idx = torch.tensor(
                stats['future_unseen_classes'], dtype=torch.long,
                device=similarity.device
            )
            stats['future_collision'] += torch.isin(
                hard_idx, fut_idx
            ).float().sum().item()
            max_fut = similarity[:, fut_idx].max(dim=1).values
            fut_margin = true_score - max_fut
            stats['future_margin_sum'] += fut_margin.sum().item()
            stats['future_margin_negative'] += (
                fut_margin < 0
            ).float().sum().item()

        if stats.get('all_non_eval_classes'):
            all_idx = torch.tensor(
                stats['all_non_eval_classes'], dtype=torch.long,
                device=similarity.device
            )
            stats['all_collision'] += torch.isin(
                hard_idx, all_idx
            ).float().sum().item()
            max_all = similarity[:, all_idx].max(dim=1).values
            all_margin = true_score - max_all
            stats['all_margin_sum'] += all_margin.sum().item()
            stats['all_margin_negative'] += (
                all_margin < 0
            ).float().sum().item()

        # ---- v3b-Diag r5: group-wise soft attention mass ----
        # 每样本先在组内求和（attn 列求和），再按样本累积；
        # entropy / max_w 同时记录（对照 soft attention 失焦程度）
        if attn_weights is not None:
            stats['soft_mass_eval_sum'] += attn_weights[
                :, eval_idx].sum(dim=1).sum().item()
            stats['soft_mass_true_sum'] += attn_weights.gather(
                1, target.unsqueeze(1)).squeeze(1).sum().item()
            # v3b-Diag r6 修正: 类别来源 = per_task_classes，
            # 累计结果 = soft_mass_per_task_sum（此前误把累积字典当
            # 类别来源遍历，cls_k 拿到的是 0.0 而非类别列表）
            for k, cls_k in (stats.get('per_task_classes')
                             or {}).items():
                idx_k = torch.tensor(
                    cls_k, dtype=torch.long, device=similarity.device
                )
                stats['soft_mass_per_task_sum'][k] += attn_weights[
                    :, idx_k].sum(dim=1).sum().item()
            if stats.get('future_unseen_classes'):
                fut_idx = torch.tensor(
                    stats['future_unseen_classes'], dtype=torch.long,
                    device=similarity.device
                )
                stats['soft_mass_future_sum'] += attn_weights[
                    :, fut_idx].sum(dim=1).sum().item()
            soft_ent = -(
                attn_weights * torch.log(attn_weights.clamp_min(1e-8))
            ).sum(dim=1)
            stats['soft_entropy_sum'] += soft_ent.sum().item()
            stats['soft_max_w_sum'] += attn_weights.max(
                dim=1).values.sum().item()

        # ---- 样本级错误分解（关联分解，非因果分解） ----
        final_correct = (predicts == target)
        route_is_true = (hard_idx == target)
        route_is_old = torch.isin(hard_idx, eval_idx) & ~route_is_true
        route_is_new = torch.isin(hard_idx, new_idx) if new_idx is not None \
            else torch.zeros_like(route_is_true)

        stats['wrong_total'] += (~final_correct).float().sum().item()
        stats['wrong_and_correct_old_route'] += (
            (~final_correct) & route_is_true).float().sum().item()
        stats['wrong_and_old_old_collision'] += (
            (~final_correct) & route_is_old).float().sum().item()
        stats['wrong_and_old_new_collision'] += (
            (~final_correct) & route_is_new).float().sum().item()
        stats['correct_and_non_true_route'] += (
            final_correct & ~route_is_true).float().sum().item()

        # ---- 类级 ----
        for i in range(n):
            y = int(target[i].item())
            stats['per_class_total'][y] += 1
            stats['per_class_correct'][y] += int(bool(final_correct[i]))
            stats['per_class_oo_margin_sum'][y] += float(oo_margin[i].item())

    @staticmethod
    def _format_diag_stats(stats):
        """E6a-v2-Diag: 汇总统计累积器 → 标量指标（含两种比例形式）"""
        total = max(stats['total'], 1)
        wrong_total = max(stats['wrong_total'], 1)
        result = {
            # E6a-v3b-Diag: old_to_new_collision 更名为 to_current_task_collision
            # （明确其只统计 T_eval → 当前任务；旧键保留同值以兼容下游解析）
            'to_current_task_collision': stats['old_to_new_collision'] / total,
            'old_to_new_collision': stats['old_to_new_collision'] / total,
            'mean_old_new_margin': stats['old_new_margin_sum'] / total,
            'negative_margin_rate': stats['old_new_margin_negative'] / total,
            'old_old_top1_accuracy': stats['old_old_top1_correct'] / total,
            'mean_old_old_margin': stats['old_old_margin_sum'] / total,
            'negative_old_old_margin_rate': stats['old_old_margin_negative'] / total,
            'wrong_and_correct_old_route': stats['wrong_and_correct_old_route'] / total,
            'wrong_and_correct_old_route_of_wrong': stats['wrong_and_correct_old_route'] / wrong_total,
            'wrong_and_old_old_collision': stats['wrong_and_old_old_collision'] / total,
            'wrong_and_old_old_collision_of_wrong': stats['wrong_and_old_old_collision'] / wrong_total,
            'wrong_and_old_new_collision': stats['wrong_and_old_new_collision'] / total,
            'wrong_and_old_new_collision_of_wrong': stats['wrong_and_old_new_collision'] / wrong_total,
            'correct_and_non_true_route': stats['correct_and_non_true_route'] / total,
        }

        # ---- E6a-v3b-Diag: cross-task routing 四组 ----
        cross_fields = []
        for k in sorted(stats.get('per_task_classes') or {}):
            col_k = stats['per_task_collision'][k] / total
            mgn_k = stats['per_task_margin_sum'][k] / total
            neg_k = stats['per_task_margin_negative'][k] / total
            result[f'to_task{k}_collision'] = col_k
            result[f'mean_vs_task{k}_margin'] = mgn_k
            result[f'neg_vs_task{k}_margin_rate'] = neg_k
            cross_fields.append(
                f"to_task{k}_collision={col_k:.4f} "
                f"mean_vs_task{k}_margin={mgn_k:.4f} "
                f"neg_vs_task{k}_margin_rate={neg_k:.4f}"
            )
        if stats.get('any_non_eval_classes'):
            any_col = stats['any_collision'] / total
            any_mgn = stats['any_margin_sum'] / total
            any_neg = stats['any_margin_negative'] / total
            result['to_any_non_eval_task_collision'] = any_col
            result['mean_vs_any_non_eval_margin'] = any_mgn
            result['neg_vs_any_non_eval_margin_rate'] = any_neg
            cross_fields.append(
                f"to_any_non_eval_task_collision={any_col:.4f} "
                f"mean_vs_any_non_eval_margin={any_mgn:.4f} "
                f"neg_vs_any_non_eval_margin_rate={any_neg:.4f}"
            )
        # v3b-Diag r2: future-unseen（use_seen_routing=False 时检索可
        # 落到未来任务 keys）+ all-non-eval（全部类别 - 被评估任务类）
        if stats.get('future_unseen_classes'):
            fut_col = stats['future_collision'] / total
            fut_mgn = stats['future_margin_sum'] / total
            fut_neg = stats['future_margin_negative'] / total
            result['to_future_unseen_collision'] = fut_col
            result['mean_vs_future_unseen_margin'] = fut_mgn
            result['neg_vs_future_unseen_margin_rate'] = fut_neg
            cross_fields.append(
                f"to_future_unseen_collision={fut_col:.4f} "
                f"mean_vs_future_unseen_margin={fut_mgn:.4f} "
                f"neg_vs_future_unseen_margin_rate={fut_neg:.4f}"
            )
        if stats.get('all_non_eval_classes'):
            all_col = stats['all_collision'] / total
            all_mgn = stats['all_margin_sum'] / total
            all_neg = stats['all_margin_negative'] / total
            result['to_any_non_eval_all_collision'] = all_col
            result['mean_vs_any_non_eval_all_margin'] = all_mgn
            result['neg_vs_any_non_eval_all_margin_rate'] = all_neg
            # v3b-Diag r5: hard route → 被评估任务（= 1 - all 口径），
            # 补全 SoftMassDiag / AttnMass 表格的 hard → T_eval 一列
            result['to_eval_task_collision'] = 1.0 - all_col
            cross_fields.append(
                f"to_any_non_eval_all_collision={all_col:.4f} "
                f"mean_vs_any_non_eval_all_margin={all_mgn:.4f} "
                f"neg_vs_any_non_eval_all_margin_rate={all_neg:.4f} "
                f"to_eval_task_collision={1.0 - all_col:.4f}"
            )
        if cross_fields:
            result['cross_task_fields'] = ' '.join(cross_fields)

        # ---- v3b-Diag r5: group-wise soft attention mass ----
        # .get(default 0.0): 兼容旧 checkpoint extras 恢复的
        # phase_stats_pre（无 soft-mass 累积器；SoftMassDiag 的 pre/post
        # 对照另有键存在性检查，不会误把缺失打印成 0）
        soft_fields = []
        soft_items = [
            ('soft_mass_to_eval_task',
             stats.get('soft_mass_eval_sum', 0.0) / total),
            ('soft_mass_to_true_key',
             stats.get('soft_mass_true_sum', 0.0) / total),
        ]
        for k in sorted(stats.get('soft_mass_per_task_sum') or {}):
            soft_items.append((
                f'soft_mass_to_task{k}',
                stats['soft_mass_per_task_sum'][k] / total))
        if stats.get('future_unseen_classes'):
            soft_items.append((
                'soft_mass_to_future_unseen',
                stats.get('soft_mass_future_sum', 0.0) / total))
        soft_items.append((
            'mean_soft_attn_entropy',
            stats.get('soft_entropy_sum', 0.0) / total))
        soft_items.append((
            'mean_soft_attn_max_w',
            stats.get('soft_max_w_sum', 0.0) / total))
        for name, val in soft_items:
            result[name] = val
            soft_fields.append(f"{name}={val:.4f}")
        result['soft_mass_fields'] = ' '.join(soft_fields)

        return result

    @staticmethod
    def _print_diag_stats(tag, client_id, task, m):
        """E6a-v2-Diag: 打印完整诊断指标（括号内为占所有分类错误样本的比例）"""
        cross_str = m.get('cross_task_fields', '')
        if cross_str:
            cross_str = ' ' + cross_str
        soft_str = m.get('soft_mass_fields', '')
        if soft_str:
            soft_str = ' ' + soft_str
        print(f"[{tag}] client={client_id} task={task} "
              f"to_current_task_collision={m['to_current_task_collision']:.4f} "
              f"mean_old_new_margin={m['mean_old_new_margin']:.4f} "
              f"negative_margin_rate={m['negative_margin_rate']:.4f} "
              f"old_old_top1_accuracy={m['old_old_top1_accuracy']:.4f} "
              f"mean_old_old_margin={m['mean_old_old_margin']:.4f} "
              f"negative_old_old_margin_rate={m['negative_old_old_margin_rate']:.4f} "
              f"wrong_and_correct_old_route={m['wrong_and_correct_old_route']:.4f}"
              f"({m['wrong_and_correct_old_route_of_wrong']:.4f}) "
              f"wrong_and_old_old_collision={m['wrong_and_old_old_collision']:.4f}"
              f"({m['wrong_and_old_old_collision_of_wrong']:.4f}) "
              f"wrong_and_old_new_collision={m['wrong_and_old_new_collision']:.4f}"
              f"({m['wrong_and_old_new_collision_of_wrong']:.4f}) "
              f"correct_and_non_true_route={m['correct_and_non_true_route']:.4f}"
              f"{cross_str}"
              f"{soft_str}")

    def _ensure_diag_fixed_inputs(self, max_samples=256):
        """
        E6a-v2-Diag / E6a-v3b-Diag: 固定 Task0 诊断样本（幂等，两步独立判断）

        1) indices: 取 test_loader[0] Subset 的前 N 个索引（前后两次提取
           使用完全相同的样本集合与顺序）。已存在则不覆盖——保持旧
           checkpoint 恢复（extras 只带 indices）时的原样本集合。
        2) 张量: 缓存这批样本的实际 input/target（CPU，一次性）。
           只固定 indices 不够——若数据带随机增强（crop/flip），每次
           重新取数会得到不同输入，feature/anchor_feat drift 会混入
           augmentation noise。缓存张量后所有 pre/post 提取使用逐位
           相同的输入。

        P0 修复（v3b-Diag r2）:
        - RNG 保护: 缓存过程会真正遍历 DataLoader，随机增强会消耗
          torch/numpy/python RNG。必须 get_rng_state/set_rng_state 包裹，
          否则诊断/task-checkpoint 会改变后续 Phase1 的 shuffle/增强
          随机轨迹，"精确复现 v3a 轨迹"不成立（实验核心是因果对照，
          诊断本身不能改变训练轨迹）。
        - 独立判断: indices 已存在（旧 CP2/CP3 extras 恢复）但
          fixed tensors 缺失时，必须用已有 indices 重新缓存张量。
          不能因 indices 非空就整体跳过——否则固定张量机制对旧
          checkpoint 永远不生效，fallback 重新取数会重新引入增强噪声。
        """
        if self._diag_feature_samples is None:
            subset = self.test_loader[0]
            indices = list(subset.indices)
            self._diag_feature_samples = indices[:min(max_samples, len(indices))]

        if (self._diag_fixed_inputs is None
                or self._diag_fixed_targets is None):
            rng_backup = get_rng_state()
            try:
                # 缓存实际输入张量（CPU，一次性；增强实现固定在本次缓存）
                sub = Subset(self.train_data[0], self._diag_feature_samples)
                loader = DataLoader(sub, batch_size=16, shuffle=False)
                inputs, targets = [], []
                for x, y in loader:
                    inputs.append(x)
                    targets.append(y)
                if inputs:
                    self._diag_fixed_inputs = torch.cat(inputs, dim=0).cpu()
                    self._diag_fixed_targets = torch.cat(targets, dim=0).cpu()
            finally:
                set_rng_state(rng_backup)

    def _diag_fixed_input_batches(self, batch_size=8):
        """
        E6a-v3b-Diag: 固定诊断输入的 batch 迭代器
        优先使用 _diag_fixed_inputs（缓存张量，无随机增强噪声）。
        _ensure_diag_fixed_inputs 保证旧 checkpoint 恢复后也会补缓存
        张量；fallback 按 indices 取数仅作安全网（缓存失败的极端情况）。
        """
        self._ensure_diag_fixed_inputs()

        if self._diag_fixed_inputs is not None:
            n = self._diag_fixed_inputs.size(0)
            for s in range(0, n, batch_size):
                yield (
                    self._diag_fixed_inputs[s:s + batch_size],
                    self._diag_fixed_targets[s:s + batch_size],
                )
        else:
            sub = Subset(self.train_data[0], self._diag_feature_samples)
            loader = DataLoader(sub, batch_size=batch_size, shuffle=False)
            for x, y in loader:
                yield x, y

    def _diag_extract_features(self):
        """
        E6a-v2-Diag: 提取实际参与 Key routing 的 query
        = vit forward_head 输出的 feat（与 Tail_Anchor.key 做 cosine
        similarity 的表示），调用方式与 evaluate() 完全一致
        （task_id=self.task_id，不人为切换回 Task0 Prompt）

        特征提取是纯诊断操作：
        1. 不改变 model / vit / original_model 的 train-eval 状态；
        2. 不改变外部 RNG 状态；
        3. 使用固定 Task0 样本和固定顺序；
        4. 不调用 self.model()，因此完全不碰 Tail Anchor 的 train/eval 状态。
        """
        # ============================================================
        # 保存进入诊断前的状态
        # ============================================================
        model_was_training = self.model.training
        vit_was_training = self.vit.training

        if self.original_model is not None:
            original_model_was_training = self.original_model.training
        else:
            original_model_was_training = None

        rng_backup = get_rng_state()

        try:
            # ========================================================
            # 固定 Task0 样本（E6a-v3b-Diag: 使用缓存张量，无增强噪声）
            # ========================================================
            self.vit.to(self.device)

            if self.original_model is not None:
                self.original_model.to(self.device)
                self.original_model.eval()

            feats = []

            with torch.no_grad():
                for input, _target in self._diag_fixed_input_batches():
                    input = input.to(self.device, non_blocking=True)

                    if self.original_model is not None:
                        output = self.original_model(input)
                        cls_features = (
                            output['pre_logits']
                            .requires_grad_(False)
                        )
                    else:
                        cls_features = None

                    output = self.vit(
                        input,
                        task_id=self.task_id,
                        cls_features=cls_features,
                        train=True
                    )

                    feats.append(
                        output['feat'].detach().cpu()
                    )

            if len(feats) == 0:
                return torch.empty(0, 768)

            return torch.cat(feats, dim=0)

        finally:
            # ========================================================
            # 完整恢复进入诊断前的状态
            # ========================================================
            self.model.train(model_was_training)
            self.vit.train(vit_was_training)

            if (
                self.original_model is not None
                and original_model_was_training is not None
            ):
                self.original_model.train(original_model_was_training)

            set_rng_state(rng_backup)

    def _diag_extract_anchor_feat(self):
        """
        E6a-v3b-Diag: 提取固定 Task0 样本实际送入 head 的 retrieved anchor 表征
        = Tail_Anchor forward 的 anchor_feat（hard + ratio*(soft-hard)），
        即 head 输入 (x ⊕ anchor_feat) 的检索分量。

        判断: raw Anchor 参数没坏 ≠ 送进 head 的检索表征没坏
        （attention 分布变化会使 soft_anchor 混入新任务 Anchor）。

        协议与 _diag_eval_task 完全一致:
        - vit(task_id=self.task_id) 提取 feat（当前 task-stage prompt）
        - model(feat, ...) 的第 4 个返回值 anchor_feat
        - model.eval() 下 usage buffer 不累积
        - 不加载 heads[task]（只用 anchor_feat，logits 被忽略）
        - try/finally 恢复 train/eval 状态 + RNG
        """
        # 幂等: indices 已存在但 fixed tensors 缺失（旧 checkpoint 恢复）
        # 时会用已有 indices 重新缓存张量（内部含 RNG 保护）
        self._ensure_diag_fixed_inputs()

        model_was_training = self.model.training
        vit_was_training = self.vit.training

        if self.original_model is not None:
            original_model_was_training = self.original_model.training
        else:
            original_model_was_training = None

        rng_backup = get_rng_state()

        try:
            # E6a-v3b-Diag: 使用缓存张量，无增强噪声
            self.model.to(self.device)
            self.vit.to(self.device)

            if self.original_model is not None:
                self.original_model.to(self.device)
                self.original_model.eval()

            # eval 模式: usage 不累积
            self.model.eval()

            proto_bank, proto_valid = self._build_semantic_proto_bank()
            proto_calib_mask = self._build_proto_calib_mask(0)

            feats = []

            with torch.no_grad():
                for input, target in self._diag_fixed_input_batches():
                    input = input.to(self.device, non_blocking=True)
                    target = target.to(self.device, non_blocking=True)

                    if self.original_model is not None:
                        output = self.original_model(input)
                        cls_features = (
                            output['pre_logits']
                            .requires_grad_(False)
                        )
                    else:
                        cls_features = None

                    output = self.vit(
                        input,
                        task_id=self.task_id,
                        cls_features=cls_features,
                        train=True
                    )
                    feat = output['feat'].to(self.device)

                    _, _, _, anchor_feat, _, _, _ = self.model(
                        feat,
                        target,
                        proto_bank=proto_bank,
                        proto_valid_mask=proto_valid,
                        proto_calib_mask=proto_calib_mask
                    )
                    feats.append(anchor_feat.detach().cpu())

            if len(feats) == 0:
                return torch.empty(0, 768)

            return torch.cat(feats, dim=0)

        finally:
            self.model.train(model_was_training)
            self.vit.train(vit_was_training)

            if (
                self.original_model is not None
                and original_model_was_training is not None
            ):
                self.original_model.train(original_model_was_training)

            set_rng_state(rng_backup)

    def _diag_write_classwise_csv(self, stats_pre, stats_post):
        """
        E6a-v2-Diag: 类级诊断 CSV（只输出原始数据，不在代码里宣称因果）
        列: class_id / class_key_drift / class_acc_pre_phase2 /
            class_acc_post_phase2 / class_acc_drop /
            class_old_old_margin_pre / class_old_old_margin_post
        """
        if stats_post is None or self._diag_key_before_phase2 is None \
                or self._diag_key_after_phase2 is None:
            return

        os.makedirs('diagnostics', exist_ok=True)
        path = os.path.join(
            'diagnostics',
            f"E6a_v2_R{self.round}_C{self.id}_T{self.task_id}_classwise.csv"
        )

        kb = F.normalize(self._diag_key_before_phase2.to(self.device), dim=1)
        ka = F.normalize(self._diag_key_after_phase2.to(self.device), dim=1)

        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'class_id', 'class_key_drift',
                'class_acc_pre_phase2', 'class_acc_post_phase2',
                'class_acc_drop',
                'class_old_old_margin_pre', 'class_old_old_margin_post',
            ])
            for c in sorted(self.old_seen_classes):
                drift = (1.0 - (kb[c] * ka[c]).sum()).item()
                acc_pre = oo_pre = acc_post = oo_post = None
                for st, which in ((stats_pre, 'pre'), (stats_post, 'post')):
                    if st is not None and st['per_class_total'][c] > 0:
                        t = st['per_class_total'][c].item()
                        acc = 100.0 * st['per_class_correct'][c].item() / t
                        oo = st['per_class_oo_margin_sum'][c].item() / t
                        if which == 'pre':
                            acc_pre, oo_pre = acc, oo
                        else:
                            acc_post, oo_post = acc, oo
                drop = (acc_pre - acc_post) \
                    if (acc_pre is not None and acc_post is not None) else None
                writer.writerow([
                    c, f"{drift:.6f}",
                    f"{acc_pre:.2f}" if acc_pre is not None else '',
                    f"{acc_post:.2f}" if acc_post is not None else '',
                    f"{drop:+.2f}" if drop is not None else '',
                    f"{oo_pre:.4f}" if oo_pre is not None else '',
                    f"{oo_post:.4f}" if oo_post is not None else '',
                ])
        print(f"[TIDR-ClasswiseDiag] saved: {path}")

    def get_global_proto_and_head(self, proto, head, prompt, round_num):
        """更新全局原型、分类头和提示，然后进行评估"""
        self.global_protos = deepcopy(proto)
        # 这是 Input Enhancement (vit.head) 聚合后的全局分类头
        # 原始 FedTA 不会将其加载到 Tail Anchor 的 model.head
        self.global_head = deepcopy(head)
        self.prompts = prompt
        self.vit.load_prompts(self.prompts)

        if self.task_id == 0:
            self.evaluate(self.task_id, self.nb_classes)
        else:
            self.evaluate(0, self.nb_classes)
            self.evaluate(self.task_id, self.nb_classes)

    def get_global_proto_and_head_no_test(self, proto, head, prompt, round_num):
        """更新全局原型、分类头和提示，不进行评估"""
        self.global_protos = deepcopy(proto)
        # 这是 Input Enhancement (vit.head) 聚合后的全局分类头
        # 原始 FedTA 不会将其加载到 Tail Anchor 的 model.head
        self.global_head = deepcopy(head)
        self.prompts = prompt
        self.vit.load_prompts(self.prompts)

    def get_head(self, head):
        """更新本地分类头并评估"""
        self.heads[self.task_id] = deepcopy(head)
        if self.task_id == 0:
            self.evaluate(self.task_id, self.nb_classes)
        else:
            self.evaluate(0, self.nb_classes)
            self.evaluate(self.task_id, self.nb_classes)

    def train_only_heads(self, round_num, args):
        """仅训练分类头（不训练提示）"""
        task = round_num // self.task_per_global_epoch

        if self.task_id != task:
            if args.data_name in ['cifar100', '5datasets', 'ImageNet-R']:
                self.get_data(task)
            self.task_id = task

        if self.heads[self.task_id] is not None:
            self.head.load_head(self.heads[self.task_id])

        self.head.to(self.device)
        train_loader = DataLoader(self.traindata, batch_size=args.batch_size, num_workers=args.num_workers,
                                  pin_memory=args.pin_mem, shuffle=True)
        print(f'Client {self.id} on Task {self.task_id} is training prompts')

        optimizer = torch.optim.Adam(self.head.parameters(), lr=self.lr, weight_decay=1e-3)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        for epoch in range(self.local_epoch):
            for input, target in train_loader:
                input = Variable(input, requires_grad=False).to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(input)

                output = self.head(output['feat'])
                logits = output

                mask = self.current_class
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                loss = criterion(logits, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.heads[self.task_id] = deepcopy(self.head.get_head())

        if self.task_id == 0:
            self.evaluate(self.task_id, args.nb_classes)
        else:
            self.evaluate(0, args.nb_classes)
            self.evaluate(self.task_id, args.nb_classes)

    def evaluate(self, task=0, nb_classes=None, phase="Server Aggregation",
                 notes_prefix="Global evaluation on task"):
        """使用提示和分类头进行标准评估"""
        self.model.eval()
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data, batch_size=8, shuffle=True)
        correct = 0
        total = 0

        self.model.load_head(self.heads[task])
        self.model.to(self.device)

        # E1-Repair v1: paired hard-eval 诊断（仅评估旧任务时）
        do_hard_diag = (self.use_soft_anchor and task < self.task_id)
        hard_correct = 0

        # TIDR-Diag: 检测旧任务样本被新任务 Key 抢走（new-key invasion）
        # 仅在评估旧任务时统计
        do_invasion_diag = (task < self.task_id)
        old_to_new_collision = 0.0
        old_new_margin_sum = 0.0
        old_new_margin_negative = 0.0
        diag_total = 0
        if do_invasion_diag:
            # 用 new-only 类别：当前任务类减去被评估任务类，
            # 避免任务间共享类别时把共享类误算成 "new-key invasion"
            eval_classes = set(int(c) for c in self.class_mask[task])
            new_only_classes = [
                int(c) for c in self.class_mask[self.task_id]
                if int(c) not in eval_classes
            ]
            if len(new_only_classes) == 0:
                # 极端情况：当前任务类全部 ⊆ 被评估任务类，无可比的新类
                do_invasion_diag = False
            else:
                current_task_classes = torch.tensor(
                    new_only_classes,
                    dtype=torch.long,
                    device=self.device
                )

        # E6a-v3b-Diag: cross-task routing 分组
        # old_to_new_collision 只统计 T_eval → 当前任务，R9（T1 阶段）与
        # R10（T2 阶段）的该指标不可直接比较（统计对象变了）。
        # 新增逐任务组 to_task{k}（k != task）+ any-non-eval 组，
        # 用于研究 cross-task routing accumulation。
        # v3b-Diag r2: use_seen_routing=False 时 hard/soft 可访问全部
        # nb_classes（含未来任务尚未 seen 的 keys），故再增
        # future-unseen 组与 any-non-eval-all（全部类别 - 被评估任务类）组。
        do_cross_task_diag = (task < self.task_id)
        cross_task_groups = {}
        cross_collision = {}
        cross_margin_sum = {}
        cross_margin_negative = {}
        any_non_eval_idx = None
        any_collision = 0.0
        any_margin_sum = 0.0
        any_margin_negative = 0.0
        future_unseen_idx = None
        future_collision = 0.0
        future_margin_sum = 0.0
        future_margin_negative = 0.0
        all_non_eval_idx = None
        all_collision = 0.0
        all_margin_sum = 0.0
        all_margin_negative = 0.0
        if do_cross_task_diag:
            eval_classes_ct = set(int(c) for c in self.class_mask[task])
            for k in range(0, self.task_id + 1):
                if k == task:
                    continue
                cls_k = sorted(
                    int(c) for c in self.class_mask[k]
                    if int(c) not in eval_classes_ct
                )
                if cls_k:
                    cross_task_groups[k] = torch.tensor(
                        cls_k, dtype=torch.long, device=self.device
                    )
                    cross_collision[k] = 0.0
                    cross_margin_sum[k] = 0.0
                    cross_margin_negative[k] = 0.0
            all_seen_ct = set()
            for k in range(0, self.task_id + 1):
                for c in self.class_mask[k]:
                    all_seen_ct.add(int(c))
            any_cls = sorted(all_seen_ct - eval_classes_ct)
            if any_cls:
                any_non_eval_idx = torch.tensor(
                    any_cls, dtype=torch.long, device=self.device
                )
            # v3b-Diag r2: future-unseen / all-non-eval
            all_cls_ct = set(range(self.nb_classes))
            future_cls = sorted(all_cls_ct - all_seen_ct)
            if future_cls:
                future_unseen_idx = torch.tensor(
                    future_cls, dtype=torch.long, device=self.device
                )
            all_noneval_cls = sorted(all_cls_ct - eval_classes_ct)
            if all_noneval_cls:
                all_non_eval_idx = torch.tensor(
                    all_noneval_cls, dtype=torch.long, device=self.device
                )

        # E6a-v2-Diag: Old-Old routing 诊断（--run_tidr_diagnostics 开启时）
        # old-old 类别集合 = 被评估任务自身类别（与 head 掩码一致）
        do_old_old_diag = (self.run_tidr_diagnostics and task < self.task_id)
        old_old_top1_correct = 0.0
        old_old_margin_sum = 0.0
        old_old_margin_negative = 0.0
        eval_class_idx = None
        if do_old_old_diag:
            eval_class_idx = torch.tensor(
                [int(c) for c in self.class_mask[task]],
                dtype=torch.long, device=self.device
            )

        # GPC-DR: proto bank for calibrated evaluation
        proto_bank, proto_valid = self._build_semantic_proto_bank()
        proto_calib_mask = self._build_proto_calib_mask(task)

        for input, target in test_loader:
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    output = output['pre_logits'].requires_grad_(False)
                    output = self.vit(input, task_id=self.task_id, cls_features=output, train=True)
                    feat = output['feat'].to(self.device)

                # 正常 E1 soft inference (TIDR-Diag: 接出 hard_idx)
                pre, _, _, _, _, hard_idx, _ = self.model(
                    feat, target.to(self.device),
                    proto_bank=proto_bank, proto_valid_mask=proto_valid,
                    proto_calib_mask=proto_calib_mask
                )

                # Paired hard inference (同一 batch, 同一 ViT feature, 同一 head)
                if do_hard_diag:
                    original_flag = self.model.soft_anchor
                    try:
                        self.model.soft_anchor = False
                        pre_hard, _, _, _, _, _, _ = self.model(
                                feat, target.to(self.device),
                                proto_bank=proto_bank, proto_valid_mask=proto_valid,
                                proto_calib_mask=proto_calib_mask
                            )
                    finally:
                        self.model.soft_anchor = original_flag

                # TIDR-Diag: 旧任务样本是否被新任务 Key 抢走
                if do_invasion_diag or do_old_old_diag or do_cross_task_diag:
                    similarity = self.model.compute_similarity(feat)
                    diag_total += len(target)

                if do_invasion_diag:
                    # (1) hard routing collision: hard_idx 落在当前任务类索引上
                    old_to_new_collision += torch.isin(
                        hard_idx, current_task_classes
                    ).float().sum().item()

                    # (2) margin: cos(f, K_y) - max_{c in C_new} cos(f, K_c)
                    correct_sim = similarity.gather(
                        1, target.unsqueeze(1)
                    ).squeeze(1)
                    max_new_sim = similarity[
                        :, current_task_classes
                    ].max(dim=1).values
                    margin = correct_sim - max_new_sim

                    old_new_margin_sum += margin.sum().item()
                    old_new_margin_negative += (
                        margin < 0
                    ).float().sum().item()

                # E6a-v3b-Diag: cross-task routing 累积
                # （逐任务 to_task{k} + any-non-eval；correct_sim 与
                #   do_invasion 的定义一致: cos(f, K_y)）
                if do_cross_task_diag:
                    true_score_ct = similarity.gather(
                        1, target.unsqueeze(1)
                    ).squeeze(1)
                    for k, idx_k in cross_task_groups.items():
                        cross_collision[k] += torch.isin(
                            hard_idx, idx_k
                        ).float().sum().item()
                        max_k = similarity[:, idx_k].max(dim=1).values
                        margin_k = true_score_ct - max_k
                        cross_margin_sum[k] += margin_k.sum().item()
                        cross_margin_negative[k] += (
                            margin_k < 0
                        ).float().sum().item()
                    if any_non_eval_idx is not None:
                        any_collision += torch.isin(
                            hard_idx, any_non_eval_idx
                        ).float().sum().item()
                        max_any = similarity[
                            :, any_non_eval_idx
                        ].max(dim=1).values
                        any_margin = true_score_ct - max_any
                        any_margin_sum += any_margin.sum().item()
                        any_margin_negative += (
                            any_margin < 0
                        ).float().sum().item()
                    # v3b-Diag r2: future-unseen / all-non-eval
                    if future_unseen_idx is not None:
                        future_collision += torch.isin(
                            hard_idx, future_unseen_idx
                        ).float().sum().item()
                        max_fut = similarity[
                            :, future_unseen_idx
                        ].max(dim=1).values
                        fut_margin = true_score_ct - max_fut
                        future_margin_sum += fut_margin.sum().item()
                        future_margin_negative += (
                            fut_margin < 0
                        ).float().sum().item()
                    if all_non_eval_idx is not None:
                        all_collision += torch.isin(
                            hard_idx, all_non_eval_idx
                        ).float().sum().item()
                        max_all = similarity[
                            :, all_non_eval_idx
                        ].max(dim=1).values
                        all_margin = true_score_ct - max_all
                        all_margin_sum += all_margin.sum().item()
                        all_margin_negative += (
                            all_margin < 0
                        ).float().sum().item()

                # E6a-v2-Diag: old-old margin
                # true_score - max_{c∈被评估任务其他类} cos(f, K_c)
                if do_old_old_diag:
                    true_score = similarity.gather(
                        1, target.unsqueeze(1)
                    ).squeeze(1)
                    other_sim = similarity.clone()
                    other_sim.scatter_(1, target.unsqueeze(1), float('-inf'))
                    max_other_old = other_sim[
                        :, eval_class_idx
                    ].max(dim=1).values
                    oo_margin = true_score - max_other_old
                    old_old_margin_sum += oo_margin.sum().item()
                    old_old_margin_negative += (
                        oo_margin < 0
                    ).float().sum().item()

                    # old-old top1（旧任务类别集合内部）
                    top1_old = eval_class_idx[
                        similarity[:, eval_class_idx].argmax(dim=1)
                    ]
                    old_old_top1_correct += (
                        top1_old == target
                    ).float().sum().item()

            # Soft accuracy
            logits = pre

            mask = self.class_mask[task]
            not_mask = np.setdiff1d(np.arange(nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            predicts = torch.max(logits, dim=1)[1].cpu()
            correct += (predicts == target.cpu()).sum()
            total += len(target)

            # Hard accuracy (paired diagnostic)
            if do_hard_diag:
                logits_hard = pre_hard.index_fill(dim=1, index=not_mask, value=float('-inf'))
                predicts_hard = torch.max(logits_hard, dim=1)[1].cpu()
                hard_correct += (predicts_hard == target.cpu()).sum()

        acc = 100 * correct / total
        print(f'Soft acc: {acc}')

        if do_hard_diag:
            hard_acc = 100 * hard_correct / total
            print(f"[E1-Diag] Client {self.id}, Task {task}: "
                  f"soft={acc.item():.2f}%, hard={hard_acc.item():.2f}%")

        if self.run_tidr_diagnostics and do_old_old_diag and diag_total > 0:
            # E6a-v2-Diag: 区分本地训练后 / 服务器聚合后，并附加 old-old 指标
            # （E6a-v3b-Diag: old_to_new_collision 更名 to_current_task_collision
            #   ——只统计 T_eval → 当前任务；跨任务碰撞看 to_task{k} /
            #   to_any_non_eval_task_collision）
            scope = 'Global' if ('Aggregation' in phase or 'Global' in phase) else 'Local'
            cross_parts = []
            for k in sorted(cross_collision):
                cross_parts.append(
                    f"to_task{k}_collision="
                    f"{cross_collision[k] / diag_total:.4f} "
                    f"mean_vs_task{k}_margin="
                    f"{cross_margin_sum[k] / diag_total:.4f} "
                    f"neg_vs_task{k}_margin_rate="
                    f"{cross_margin_negative[k] / diag_total:.4f}"
                )
            if any_non_eval_idx is not None:
                cross_parts.append(
                    f"to_any_non_eval_task_collision="
                    f"{any_collision / diag_total:.4f} "
                    f"mean_vs_any_non_eval_margin="
                    f"{any_margin_sum / diag_total:.4f} "
                    f"neg_vs_any_non_eval_margin_rate="
                    f"{any_margin_negative / diag_total:.4f}"
                )
            # v3b-Diag r2: future-unseen / all-non-eval
            if future_unseen_idx is not None:
                cross_parts.append(
                    f"to_future_unseen_collision="
                    f"{future_collision / diag_total:.4f} "
                    f"mean_vs_future_unseen_margin="
                    f"{future_margin_sum / diag_total:.4f} "
                    f"neg_vs_future_unseen_margin_rate="
                    f"{future_margin_negative / diag_total:.4f}"
                )
            if all_non_eval_idx is not None:
                cross_parts.append(
                    f"to_any_non_eval_all_collision="
                    f"{all_collision / diag_total:.4f} "
                    f"mean_vs_any_non_eval_all_margin="
                    f"{all_margin_sum / diag_total:.4f} "
                    f"neg_vs_any_non_eval_all_margin_rate="
                    f"{all_margin_negative / diag_total:.4f}"
                )
            cross_str = (' ' + ' '.join(cross_parts)) if cross_parts else ''
            print(f"[TIDR-{scope}Diag] Client {self.id}, Task {task}: "
                  f"to_current_task_collision={old_to_new_collision / diag_total:.4f} "
                  f"mean_old_new_margin={old_new_margin_sum / diag_total:.4f} "
                  f"negative_margin_rate={old_new_margin_negative / diag_total:.4f} "
                  f"old_old_top1_accuracy={old_old_top1_correct / diag_total:.4f} "
                  f"mean_old_old_margin={old_old_margin_sum / diag_total:.4f} "
                  f"negative_old_old_margin_rate={old_old_margin_negative / diag_total:.4f}"
                  f"{cross_str}")
        elif do_invasion_diag and diag_total > 0:
            old_to_new_rate = old_to_new_collision / diag_total
            mean_margin = old_new_margin_sum / diag_total
            neg_rate = old_new_margin_negative / diag_total
            print(f"[TIDR-Diag] Client {self.id}, Task {task}: "
                  f"to_current_task_collision={old_to_new_rate:.4f} "
                  f"mean_old_new_margin={mean_margin:.4f} "
                  f"negative_margin_rate={neg_rate:.4f}")

        self._log_accuracy(acc.item(), f"{notes_prefix} {task}", phase, task_id=task)

    def evaluate_cosin_similarity(self, task=0, nb_classes=None):
        """使用全局原型的余弦相似度进行评估"""
        self.model.eval()
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=2)
        correct = 0
        total = 0

        for input, target in test_loader:
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    cls_features = output['pre_logits']
                    output = self.vit(input, task_id=self.task_id, cls_features=cls_features, train=True)
                _, output_mix, _, _, _, _, _ = self.model(output['feat'].to(self.device), target.to(self.device))

            for i, label in enumerate(target):
                predicts = CosineSimilarityClassifier(output_mix[i].squeeze(0), self.global_protos, self.class_mask[task])
                if predicts == label:
                    correct += 1
            total += len(target)

        acc = 100 * correct / total
        print(f'Client {self.id} on Task {task} acc is {acc}')

        self._log_accuracy(acc, f"Cosine similarity evaluation on task {task}", "Cosine Similarity", task_id=task)

    def evaluate_only_prompts(self, task=0, nb_classes=None):
        """仅使用提示增强的特征进行评估（不使用分类头）"""
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=2)
        correct = 0
        total = 0

        for input, target in test_loader:
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                    output = output['pre_logits'].requires_grad_(False)
                output = self.vit(input, task_id=self.task_id, cls_features=output, train=True)

            logits = output['logits']

            mask = self.class_mask[task]
            not_mask = np.setdiff1d(np.arange(nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            predicts = torch.max(logits, dim=1)[1].cpu()
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total
        print(f'{acc}')

        self._log_accuracy(acc.item(), f"Prompts only evaluation on task {task}", "Prompts Only", task_id=task)

    def evaluate_only_heads(self, task=0, nb_classes=None):
        """仅使用分类头进行评估（不使用提示）"""
        test_data = self.test_loader[task]
        test_loader = DataLoader(test_data, batch_size=4, shuffle=True, num_workers=2)
        self.vit.load_head(self.heads[task])
        self.vit.to(self.device)
        correct = 0
        total = 0

        for input, target in test_loader:
            input = input.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with torch.no_grad():
                if self.original_model is not None:
                    output = self.original_model(input)
                output = self.head(output['feat'])

            logits = output

            mask = self.class_mask[task]
            not_mask = np.setdiff1d(np.arange(nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            predicts = torch.max(logits, dim=1)[1].cpu()
            correct += (predicts == target.cpu()).sum()
            total += len(target)

        acc = 100 * correct / total
        print(f'{acc}')

        self._log_accuracy(acc.item(), f"Heads only evaluation on task {task}", "Heads Only", task_id=task)

    def _calculate_infonce(self, feature, label, is_last):
        """计算 InfoNCE 对比损失"""
        all_global_protos_keys = np.array(list(self.global_protos.keys()))
        all_protos = [self.global_protos[key] for key in all_global_protos_keys]
        all_protos = np.vstack(all_protos)

        pos_index = np.where(all_global_protos_keys == label)[0]
        neg_index = np.where(all_global_protos_keys != label)[0]

        f_pos = torch.from_numpy(all_protos[pos_index]).to(self.device)
        f_neg = torch.from_numpy(all_protos[neg_index]).to(self.device)
        f_proto = torch.cat((f_pos, f_neg), dim=0)

        l = torch.cosine_similarity(feature, f_proto, dim=1)
        l = l / 0.2

        exp_l = torch.exp(l).view(1, -1)
        pos_mask = torch.tensor([1] * f_pos.shape[0] + [0] * f_neg.shape[0],
                                 dtype=torch.float).to(self.device).view(1, -1)

        pos_l = exp_l * pos_mask
        sum_pos_l = pos_l.sum(1)
        sum_exp_l = exp_l.sum(1)

        if is_last:
            infonce_loss = 1 - torch.log(sum_pos_l)
        else:
            infonce_loss = -torch.log(sum_pos_l / sum_exp_l)

        return infonce_loss

    def train_only_prompts(self, round_num, args):
        """仅训练提示参数（不训练分类头）"""
        self.original_model.eval()

        if self.prompts is not None:
            self.vit.load_prompts(self.prompts)
        else:
            self.vit.init_prompts()

        if self.heads[self.task_id] is not None:
            self.vit.load_head(self.heads[self.task_id])

        task = round_num // self.task_per_global_epoch

        if self.task_id != task:
            if args.data_name in ['cifar100', '5datasets', 'ImageNet-R']:
                self.get_data(task)
            self.task_id = task

        self.model.to(self.device)
        self.vit.to(self.device)
        train_loader = DataLoader(self.traindata, batch_size=args.batch_size, num_workers=args.num_workers,
                                  pin_memory=args.pin_mem, shuffle=True)
        print(f'Client {self.id} on Task {self.task_id} is training prompts')

        optimizer = torch.optim.Adam(self.vit.parameters(), lr=self.lr, weight_decay=1e-3)
        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        for epoch in tqdm(range(self.local_epoch)):
            for input, target in train_loader:
                input = Variable(input, requires_grad=False).to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)

                with torch.no_grad():
                    if self.original_model is not None:
                        output = self.original_model(input)
                        cls_features = output['pre_logits']

                output = self.vit(input, task_id=self.task_id, cls_features=cls_features, train=True)
                logits = output['logits']
                pull_off = output['reduce_sim']

                mask = self.current_class
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(self.device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

                loss = criterion(logits, target) - 0.1 * pull_off
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.heads[self.task_id] = deepcopy(self.vit.head)
        self.prompts = deepcopy(self.vit.get_prompts())

        if self.task_id == 0:
            self.evaluate(self.task_id, args.nb_classes)
        else:
            self.evaluate(0, args.nb_classes)
            self.evaluate(self.task_id, args.nb_classes)

    def get_global_prompt_head(self, head, prompt):
        """更新全局提示和分类头参数"""
        self.heads[self.task_id] = deepcopy(head)
        self.model.head.to(self.device)
        self.prompts = prompt
        self.vit.load_prompts(self.prompts)

        if self.task_id == 0:
            self.evaluate(self.task_id, self.nb_classes)
        else:
            self.evaluate(0, self.nb_classes)
            self.evaluate(self.task_id, self.nb_classes)

    def get_anchor_usage(self):
        """获取 anchor 使用频率统计（FedSMR: 供联邦聚合使用）"""
        if hasattr(self.model, 'get_anchor_usage'):
            return self.model.get_anchor_usage()
        return None

    def get_prompt_usage(self):
        """获取 prompt 使用频率统计（FedSMR: 供联邦聚合使用）"""
        if hasattr(self.vit, 'prompt') and hasattr(self.vit.prompt, 'get_prompt_usage'):
            return self.vit.prompt.get_prompt_usage()
        return None