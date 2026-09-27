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
import os
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
    TIDR_DIAG_ROUND,
    TIDR_DIAG_CLIENT,
    TIDR_DIAG_TASK,
    CKPT_CP2,
    CKPT_CP3,
    get_rng_state,
    set_rng_state,
)


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
        self._diag_key_before_phase2 = None
        self._diag_key_after_phase2 = None
        self._diag_old_key_mask = None
        self._diag_feature_before = None
        self._diag_feature_after = None
        self._diag_feature_samples = None
        self._diag_phase_acc_pre = {}
        self._diag_phase_stats_pre = None
        self._diag_step_stats = None

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
        # E6a-v2-Diag: Phase2 开始钩子（CP2 保存点）
        # 位于 Phase2 全部初始化之后、第一个 batch 之前
        # =============================================================
        if self._tidr_diag_config is not None and not skip_start_hook:
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
        # E6a-v2-Diag: Phase2 结束钩子
        # PhaseDiag post / StepDiag / FeatureDiag / KeyCF Rollback /
        # Classwise CSV / CP3 保存（本地、服务器聚合之前 → Local 范畴）
        # =============================================================
        if self._tidr_diag_config is not None:
            self._tidr_diag_phase2_end(round, args)

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

    def _tidr_diag_phase2_start(self, round, args, global_step, total_steps):
        """
        E6a-v2-Diag: Phase2 开始钩子
        位置: Phase1 结束、Phase2 全部初始化完成、第一个 batch 之前（CP2 保存点）
        """
        cfg = self._tidr_diag_config

        # 1) Key 快照 + old_key_mask
        #    old-key 定义与 TIDR-KeyDiag 完全一致: old_seen_classes
        #    （= 已见类别 - 当前任务类别，任务间共享类别自动排除）
        self._diag_key_before_phase2 = self.model.key.data.clone().cpu()
        old_idx = sorted(self.old_seen_classes)
        old_key_mask = torch.zeros(self.nb_classes, dtype=torch.bool)
        if old_idx:
            old_key_mask[torch.tensor(old_idx, dtype=torch.long)] = True
        self._diag_old_key_mask = old_key_mask

        # 2) PhaseDiag pre: 用当前 Task1-stage 表示评估 Task0 / Task1
        #    （评估协议镜像 evaluate(): vit(task_id=self.task_id), heads[task]）
        acc0_pre, stats0_pre = self._diag_eval_task(0, collect_stats=True)
        acc1_pre, _ = self._diag_eval_task(1, collect_stats=False)
        self._diag_phase_acc_pre = {'task0': acc0_pre, 'task1': acc1_pre}
        self._diag_phase_stats_pre = stats0_pre
        print(f"[TIDR-PhaseDiag] client={self.id} task={self.task_id} "
              f"task0_acc_pre_phase2={acc0_pre:.2f} "
              f"task1_acc_pre_phase2={acc1_pre:.2f}")

        # 3) FeatureDrift: 固定 Task0 测试样本 + Phase2 前特征快照
        #    （实际参与 Key routing 的 query = vit forward_head 的 feat）
        if self._diag_feature_samples is None:
            self._diag_capture_feature_samples()
        self._diag_feature_before = self._diag_extract_features()

        # 4) CP2 保存
        if cfg.get('save'):
            extras = {
                'key_before_phase2': self._diag_key_before_phase2,
                'old_key_mask': self._diag_old_key_mask,
                'phase2_global_step': global_step,
                'total_steps': total_steps,
                'phase_acc_pre': dict(self._diag_phase_acc_pre),
                'phase_stats_pre': self._diag_phase_stats_pre,
                'feature_before': self._diag_feature_before,
                'feature_sample_indices': list(self._diag_feature_samples),
            }
            save_checkpoint(cfg['server'], 'R5_C0_pre_phase2', CKPT_CP2, round, extras)

        if cfg.get('stop_at') == 'R5_C0_pre_phase2':
            raise DiagStopException('R5_C0_pre_phase2')

    def _tidr_diag_phase2_end(self, round, args):
        """
        E6a-v2-Diag: Phase2 结束钩子（本地、服务器聚合之前 → Local 范畴）
        PhaseDiag post / StepDiag / FeatureDiag / KeyCF Rollback /
        Old-Old 诊断（Normal & Rollback）/ Classwise CSV / CP3 保存
        """
        cfg = self._tidr_diag_config
        standalone = bool(cfg.get('standalone'))

        # ---- 1) PhaseDiag post ----
        acc0_post, stats0_post = self._diag_eval_task(0, collect_stats=True)
        acc1_post, _ = self._diag_eval_task(1, collect_stats=False)
        pre = self._diag_phase_acc_pre or {}
        acc0_pre = pre.get('task0', float('nan'))
        acc1_pre = pre.get('task1', float('nan'))
        print(f"[TIDR-PhaseDiag] client={self.id} task={self.task_id} "
              f"task0_acc_pre_phase2={acc0_pre:.2f} "
              f"task0_acc_post_phase2={acc0_post:.2f} "
              f"task0_acc_change={acc0_post - acc0_pre:+.2f} "
              f"task1_acc_pre_phase2={acc1_pre:.2f} "
              f"task1_acc_post_phase2={acc1_post:.2f} "
              f"task1_acc_change={acc1_post - acc1_pre:+.2f}")

        # ---- 2) StepDiag ----
        if self._diag_step_stats is not None:
            s = self._diag_step_stats
            print(f"[TIDR-StepDiag] client={self.id} "
                  f"phase2_optimizer_steps={s['phase2_optimizer_steps']} "
                  f"phase2_num_batches={s['phase2_num_batches']} "
                  f"phase2_num_samples={s['phase2_num_samples']} "
                  f"phase2_mean_batch_size={s['phase2_mean_batch_size']:.2f}")

        # ---- 3) FeatureDiag（同一批固定 Task0 样本、相同 task_id） ----
        if self._diag_feature_samples is None:
            self._diag_capture_feature_samples()
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

        # ---- 4) KeyCF: Old-Key Rollback 反事实实验 ----
        acc_normal = acc0_post
        metrics_normal = self._format_diag_stats(stats0_post) if stats0_post else None
        if metrics_normal:
            self._print_diag_stats('TIDR-Diag-Normal', self.id, 0, metrics_normal)

        key_after = self.model.key.data.clone().cpu()
        self._diag_key_after_phase2 = key_after

        acc_rollback, recovery, restoration_pass = self._diag_key_rollback(acc_normal)
        print(f"[TIDR-KeyCF] client={self.id} task={self.task_id} "
              f"acc_normal={acc_normal:.2f} acc_rollback={acc_rollback:.2f} "
              f"recovery={recovery:+.2f} key_restoration_pass={restoration_pass}")

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
                'old_key_mask': self._diag_old_key_mask,
                'phase_acc_pre': dict(self._diag_phase_acc_pre or {}),
                'phase_acc_post': {'task0': acc0_post, 'task1': acc1_post},
                'phase_stats_pre': self._diag_phase_stats_pre,
                'feature_before': self._diag_feature_before,
                'feature_after': self._diag_feature_after,
                'feature_sample_indices': list(self._diag_feature_samples or []),
            }
            save_checkpoint(cfg['server'], 'R5_C0_post_phase2', CKPT_CP3, round, extras)

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
            # ========================================================
            with torch.no_grad():
                key_before = (
                    self._diag_key_before_phase2
                    .to(
                        device=self.device,
                        dtype=self.model.key.dtype
                    )
                )

                self.model.key[old_idx].copy_(
                    key_before[old_idx]
                )

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

    def _diag_eval_task(self, task, collect_stats=False):
        """
        E6a-v2-Diag: 只读、可恢复的确定性单任务评估。

        关键原则：
        1. 不替换 self.model.head 模块对象（保护 Phase2 optimizer 的参数引用）；
        2. 诊断结束后恢复 model / vit 的 train-eval 状态；
        3. 恢复 RNG，避免诊断改变后续训练数据顺序和随机增强；
        4. DataLoader 使用 shuffle=False；
        5. 评估协议仍保持：vit(task_id=self.task_id, train=True) + heads[task]。

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

        # ============================================================
        # 2. 原位保存当前 head 参数（只保存 state_dict，不替换 head 对象）
        # ============================================================
        head_backup = {
            k: v.detach().clone()
            for k, v in self.model.head.state_dict().items()
        }

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
                    # -----------------------------------------------
                    (
                        pre,
                        _,
                        _,
                        _,
                        _,
                        hard_idx,
                        _
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
                            task
                        )

            acc = (
                100.0 * correct / max(total, 1)
            )

            return acc, stats

        finally:
            # ========================================================
            # 4. 无论诊断成功还是异常，都必须恢复状态
            # ========================================================

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
        return {
            'eval_classes': eval_classes,
            'new_only_classes': new_only_classes,
            'total': 0,
            'wrong_total': 0,
            # old-new（与 evaluate() TIDR-Diag 同定义）
            'old_to_new_collision': 0.0,
            'old_new_margin_sum': 0.0,
            'old_new_margin_negative': 0.0,
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
                               target, task):
        """
        E6a-v2-Diag: 样本级统计累积
        - old-old margin: cos(f, K_y) - max_{c∈旧任务其他类} cos(f, K_c)
        - old-new margin: cos(f, K_y) - max_{c∈new-only 类} cos(f, K_c)
        - 每类一个 Key（Tail_Anchor.key 为 [nb_class, key_size]），
          类别聚合规则与实际推理一致（无需额外聚合）
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
        return {
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

    @staticmethod
    def _print_diag_stats(tag, client_id, task, m):
        """E6a-v2-Diag: 打印完整诊断指标（括号内为占所有分类错误样本的比例）"""
        print(f"[{tag}] client={client_id} task={task} "
              f"old_to_new_collision={m['old_to_new_collision']:.4f} "
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
              f"correct_and_non_true_route={m['correct_and_non_true_route']:.4f}")

    def _diag_capture_feature_samples(self, max_samples=256):
        """
        E6a-v2-Diag: 固定 Task0 测试样本
        取 test_loader[0] Subset 的前 N 个索引（前后两次提取使用完全相同
        的样本集合与顺序，避免 DataLoader shuffle 对齐问题）
        """
        subset = self.test_loader[0]
        indices = list(subset.indices)
        self._diag_feature_samples = indices[:min(max_samples, len(indices))]

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
            # 固定 Task0 样本
            # ========================================================
            sub = Subset(self.train_data[0], self._diag_feature_samples)
            loader = DataLoader(sub, batch_size=8, shuffle=False)

            self.vit.to(self.device)

            if self.original_model is not None:
                self.original_model.to(self.device)
                self.original_model.eval()

            feats = []

            with torch.no_grad():
                for input, target in loader:
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
        path = os.path.join('diagnostics', 'E6a_v2_R5_C0_classwise.csv')

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
                if do_invasion_diag or do_old_old_diag:
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
            # （old-new 三项指标沿用原 TIDR-Diag 定义；若当前任务类全部 ⊆
            #   被评估任务类，old-new 指标输出 0）
            scope = 'Global' if ('Aggregation' in phase or 'Global' in phase) else 'Local'
            print(f"[TIDR-{scope}Diag] Client {self.id}, Task {task}: "
                  f"old_to_new_collision={old_to_new_collision / diag_total:.4f} "
                  f"mean_old_new_margin={old_new_margin_sum / diag_total:.4f} "
                  f"negative_margin_rate={old_new_margin_negative / diag_total:.4f} "
                  f"old_old_top1_accuracy={old_old_top1_correct / diag_total:.4f} "
                  f"mean_old_old_margin={old_old_margin_sum / diag_total:.4f} "
                  f"negative_old_old_margin_rate={old_old_margin_negative / diag_total:.4f}")
        elif do_invasion_diag and diag_total > 0:
            old_to_new_rate = old_to_new_collision / diag_total
            mean_margin = old_new_margin_sum / diag_total
            neg_rate = old_new_margin_negative / diag_total
            print(f"[TIDR-Diag] Client {self.id}, Task {task}: "
                  f"old_to_new_collision={old_to_new_rate:.4f} "
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