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
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from tqdm import tqdm

from Models.Tail_Anchor import Tail_Anchor
from Models.classification_head import Chead
from utils import CosineSimilarityClassifier


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
        self.msp_coherence_coeff = getattr(args, 'msp_coherence_coeff', 0.0)
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
        self.use_gpa = getattr(args, 'use_gpa', False)
        self.lambda_gpa = getattr(args, 'lambda_gpa', 0.2)
        self.lambda_pcr = getattr(args, 'lambda_pcr', 0.05)
        # 已废弃保留
        self.use_soft_prompt = getattr(args, 'use_soft_prompt', False)
        self.use_sparse_softmax = getattr(args, 'use_sparse_softmax', False)
        self.top_k_anchor = getattr(args, 'top_k_anchor', None)
        self.temperature_anneal = getattr(args, 'temperature_anneal', False)
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
                top_k_anchor=self.top_k_anchor,
                temperature_anneal=self.temperature_anneal,
                use_sparse_softmax=self.use_sparse_softmax,
                diversity_margin=self.diversity_margin,
                use_seen_routing=self.use_seen_routing,
                # GPC-DR
                use_proto_calibration=self.use_proto_calibration,
                proto_beta=self.proto_beta,
                proto_temperature=self.proto_temperature,
                adaptive_gamma=self.adaptive_gamma,
                gamma_max=self.gamma_max,
                use_diff_retrieval=self.use_diff_retrieval,
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
        # Phase 2: Train classification head + anchor_pool (Soft Anchor + MSP + InfoNCE)
        # =============================================================
        self.model.train()  # usage 只在训练模式累积
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-3)

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
                    feat_prompt = output['feat']

                # GPC-DR: build proto bank + calib mask
                proto_bank, proto_valid = self._build_semantic_proto_bank()
                proto_calib_mask = self._build_proto_calib_mask(self.task_id)

                pre, output_mixed, pull_off2, anchor_feat, attn_weights, hard_idx, routing_logits = self.model(
                    feat_prompt.to(self.device), target.to(self.device),
                    global_step=global_step, total_steps=total_steps * 2,
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

        # =============================================================
        # Save hard-anchor usage across global rounds
        # =============================================================
        round_usage = task_hard_usage.detach().cpu()

        if self.task_id not in self.task_anchor_usage:
            self.task_anchor_usage[self.task_id] = round_usage.clone()
        else:
            self.task_anchor_usage[self.task_id] += round_usage

        # =============================================================
        # E1 routing diagnostics (sample-weighted summary)
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

                # 正常 E1 soft inference
                pre, _, _, _, _, _, _ = self.model(
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