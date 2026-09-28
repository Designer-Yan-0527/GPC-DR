"""
FedTA 框架服务端模块

该模块实现了 FedTA（联邦任务自适应）框架的服务端逻辑。服务端负责：
- 客户端初始化与协调
- 全局模型聚合（FedAvg）
- 基于知识蒸馏的提示融合
- 原型选择与管理
- 多轮次多任务训练编排

核心功能：
- 客户端-服务端通信协议
- 基于原型的知识共享
- 知识蒸馏提示融合
- 贪婪相似度原型选择
"""

import random
import time
import csv
import os
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from Models.Client_DF import Client_DF
from Models.classification_head import Chead
from checkpoint_utils import (
    load_checkpoint,
    save_checkpoint,
    DiagStopException,
    TIDR_DIAG_ROUND,
    TIDR_DIAG_CLIENT,
)

from utils import accuracy, global_distillation_loss
from torch.nn import functional as F
from sklearn.metrics.pairwise import cosine_similarity


class Server_DF(object):
    """
    FedTA（联邦任务自适应）框架的服务端类
    
    管理联邦学习过程，包括：
    - 客户端初始化
    - 训练协调
    - 模型聚合
    - 原型管理
    - 提示融合
    """

    def __init__(self, id, origin_model, model_name, client_num, task_num, subset, 
                 class_mask, lr, global_epoch, local_epoch, batch_size, device, 
                 method, threshold, surrogate_data, test_data, args, model):
        """
        初始化服务端实例
        
        Args:
            id: 服务端标识符
            origin_model: 预训练基础模型（不含提示）
            model_name: 模型架构名称
            client_num: 联邦中的客户端数量
            task_num: 任务总数
            subset: 每个客户端的数据子集
            class_mask: 每个客户端上每个任务的类别索引
            lr: 学习率
            global_epoch: 每个任务的全局轮次数量
            local_epoch: 每个客户端的本地轮次数量
            batch_size: 训练批次大小
            device: 训练设备（CPU/GPU）
            method: 训练方法标识符
            threshold: 原型选择的相似度阈值
            surrogate_data: 用于提示融合的代理数据集
            test_data: 测试数据集
            args: 命令行参数
            model: 增强提示的模型
        """
        self.id = id
        self.model_name = model_name
        self.origin_model = origin_model
        self.model = model

        self.client_num = client_num
        self.task_num = task_num
        self.clients = []

        # Data and class information for each client
        self.client_data = subset
        self.class_mask = class_mask

        # Training configuration
        self.surrogate_data = surrogate_data
        self.lr = lr
        self.batch_size = batch_size
        self.global_epoch = global_epoch
        self.local_epoch = local_epoch
        self.device = device
        self.method = method
        self.threshold = threshold
        self.test_data = test_data
        self.args = args

        # Task tracking
        self.task_id = -1

        # Knowledge management
        self.existing_class = set()
        self.global_head = Chead(args.nb_classes)
        self.global_protos = None
        self.temp_protos = None
        self.fix_keys = []

        # 统一日志文件初始化
        self.log_file = self._init_log_file()

        # CL 评估协议: continual-learning accuracy matrix R[i][j]
        # （i=训练到第几个 task 后，j=被评估 task；task 末轮评估所有
        #   seen tasks，用于计算标准 ACC / Forgetting F / BWT）
        self.cl_matrix_file = self.log_file.replace('.csv', '_cl_matrix.csv')
        self.cl_matrix = {}  # {client_id: [ [acc_t0, ...], ... ]} 按 after_task 索引
        with open(self.cl_matrix_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['after_task', 'client_id']
                + [f'task{t}' for t in range(self.task_num)])
        print(f"CL matrix 日志文件已创建: {self.cl_matrix_file}")

    def _build_log_filename(self):
        """
        FedSMR-v2 日志命名规则:
        - 基线: FedTA_Baseline_{dataset}.csv
        - A: Anchor_Softmax{T}_gamma{G}_{dataset}.csv
        - A+B: Anchor_Softmax{T}_gamma{G}_MSP{D}_{T}_{dataset}.csv
        - 加后缀: _Route, _Proto, _SeenRoute
        """
        parts = []
        dataset = getattr(self.args, 'data_name', 'unknown')

        use_soft = getattr(self.args, 'use_soft_anchor', False)
        use_msp = getattr(self.args, 'use_msp', False)

        if not use_soft and not use_msp:
            parts.append('FedTA_Baseline')
        elif use_soft and not use_msp:
            temp = getattr(self.args, 'soft_temperature', 0.17)
            gamma = getattr(self.args, 'soft_anchor_ratio', 0.25)
            parts.append(f'Anchor_Softmax{temp}_gamma{gamma}')
        else:
            temp = getattr(self.args, 'soft_temperature', 0.17)
            gamma = getattr(self.args, 'soft_anchor_ratio', 0.25)
            d = getattr(self.args, 'msp_diversity_coeff', 0.03)
            t = getattr(self.args, 'msp_temporal_coeff', 0.1)
            kt = getattr(self.args, 'key_temporal_ratio', 0.5)
            parts.append(f'Anchor_Softmax{temp}_gamma{gamma}_MSP{d}_{t}_kt{kt}')

        # 可选增强后缀
        if getattr(self.args, 'use_route_loss', False):
            parts.append('Route')

        parts.append(dataset)
        return '_'.join(parts) + '.csv'

    def _init_log_file(self):
        """
        初始化统一的日志文件

        格式: {hyperparams}_seed{seed}_{timestamp}.csv
        """
        base_name = self._build_log_filename()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed = getattr(self.args, 'seed', 42)
        stem = base_name.replace('.csv', '')
        filename = f"{stem}_seed{seed}_{timestamp}.csv"
        filepath = os.path.join("logs", filename)

        os.makedirs("logs", exist_ok=True)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Time', 'Round', 'Task_id', 'Client_id', 'Accuracy', 'Notes', 'Phase'])

        print(f"日志文件已创建: {filepath}")
        return filepath

    def init_client(self):
        """使用各自的数据和配置初始化所有客户端"""
        print('Initialize clients')
        for i in range(self.client_num):
            if self.args.data_name in ['cifar100', '5datasets', 'ImageNet-R']:
                self.clients.append(Client_DF(
                    i, self.origin_model, self.model_name, self.global_epoch,
                    self.client_data[i], self.local_epoch, self.batch_size,
                    self.lr, self.device, self.method, self.class_mask[i],
                    self.args, self.model, log_file=self.log_file
                ))
            else:
                self.clients.append(Client_DF(
                    i, self.origin_model, self.model_name, self.global_epoch,
                    None, self.local_epoch, self.batch_size, self.lr,
                    self.device, self.method, None, self.args, log_file=self.log_file
                ))
        print("Initialization completes")

    def fedavg(self, sample_nums):
        """
        标准联邦平均（FedAvg）

        Legacy helper; unused in FedTA/FedSMR main training path.
        主训练路径通过 SIKF + BGPS + vit.head FedAvg 进行服务器聚合，
        不对 Tail Anchor 的 key/anchor/head 做联邦平均。

        Args:
            sample_nums: 每个客户端的样本数量列表

        Returns:
            平均后的模型参数（state_dict）
        """
        training_num = sum(sample_nums)
        averaged_params = self.clients[0].model.cpu().state_dict()

        for k in averaged_params.keys():
            for i in range(self.client_num):
                local_sample_number = sample_nums[i]
                local_model_params = self.clients[i].model.cpu().state_dict()
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        return averaged_params

    def train_clients(self, start_round=0, resume_mid_round=False,
                      resume_extras=None):
        """
        协调所有客户端跨任务和轮次的主训练循环

        该方法编排：
        - 客户端数据更新
        - 每个客户端的本地训练
        - 原型选择和聚合
        - 基于知识蒸馏的提示融合
        - 分类头聚合
        - 向客户端分发全局模型

        E6a-v2-Diag:
        - start_round: 恢复 CP1 后从 Round5 开始（不重复 Round4）
        - resume_mid_round + resume_extras: 恢复 CP2 后 Round5 Client0
          跳过 Phase1 直接进入 Phase2
        """
        stop_at = getattr(self.args, 'stop_at_checkpoint', '') or ''
        run_diag = getattr(self.args, 'run_tidr_diagnostics', False)
        # E6a-v3b-Diag: 完整反事实套件（CF）生效轮（默认 Round5 = Task1 首轮，
        # Task2 诊断传 --diag_cf_round=10）
        diag_cf_round = getattr(self.args, 'diag_cf_round', TIDR_DIAG_ROUND)
        save_task_ckpt = getattr(self.args, 'save_task_checkpoints', False)

        for i in range(start_round, self.task_num * self.global_epoch):
            self.thisclients = list(range(self.client_num))

            cur_task = i // self.global_epoch
            task_first_round = (i % self.global_epoch == 0)

            # ---- E6a-v2-Diag / E6a-v3b-Diag: 诊断轮配置注入（Client0） ----
            # CF 轮（diag_cf_round）: 完整诊断（PhaseDiag + CF 套件 + CP2/CP3）
            # 其他任务首轮（task_first_round 且 task>=1）: 轻量 PhaseDiag
            #   （pre_phase1 / post_phase1 / post_phase2 三点准确率 + drift 指标）
            # mid-round 恢复（CP2 / Task_C0_post_phase1）: Phase2 上下文注入
            #   与诊断开关解耦（--save_task_checkpoints 可独立恢复 post-P1），
            #   且恢复起点轮在诊断开启时强制视为 CF 轮（一次性判定，
            #   不改写 diag_cf_round，保证后续 CF 轮如 Round10 仍生效）
            mid_round_first = resume_mid_round
            if resume_mid_round:
                self._install_phase2_resume_ctx(resume_extras)
                resume_extras = None
                resume_mid_round = False
                is_cf_round = run_diag
            else:
                is_cf_round = run_diag and i == diag_cf_round
            is_phase_diag_round = (
                run_diag and task_first_round and cur_task >= 1
                and not is_cf_round
            )
            if is_cf_round:
                self._install_tidr_diag_config(cf=True)
            elif is_phase_diag_round:
                self._install_tidr_diag_config(cf=False)

            # ---- E6a-v3b-Diag: Task{k}_start Checkpoint（Task 训练开始前） ----
            # mid-round 恢复（CP2）首轮不是 Task 边界，跳过
            if (save_task_ckpt and task_first_round and cur_task >= 1
                    and not mid_round_first):
                save_checkpoint(self, 'Task_start',
                                f'Task{cur_task}_start.pth', i)

            # ---- E6a-v3b-Diag r4: pre-task-switch 同协议基准（Client0） ----
            # 在 update_data()（task switch）之前，用 _diag_eval_task 的
            # 固定 seed 协议测全部旧任务 task0..task{cur_task-1}（此时
            # client.task_id 仍为 cur_task-1，vit 用旧任务 stage——正是
            # "Task{k-1} stage、上轮结束后"状态）。
            # 与 post_task_switch_pre_phase1 / post_phase1 / post_phase2
            # 完全同协议可比，形成 A/B/C/D 四点因果分解：
            #   A(此处) → B 的差 = task switch 本身的因果贡献
            #   B → C = Phase1 优化贡献；C → D = Phase2 优化贡献
            # 测全部旧任务（非仅 Task0）: 若 Task2 边界处 T0 大跌而 T1 几乎
            # 不掉（如 T0: 89→40, T1: 94→93），强烈支持
            # "oldest-task-specific transition failure" 而非所有旧任务一起崩。
            # 注意: 普通 evaluate()（R9 日志的 88.86%）shuffle=True 且无固定
            # seed，与 A 不可做精确差值；A 才是同协议基准。
            # 只读测量：不改模型/算法/训练 RNG（_diag_eval_task 自带
            # RNG 保护、固定诊断 seed 与 train/eval 状态恢复）。
            # mid-round 恢复轮的 client 状态是恢复中段（非 task 边界），跳过。
            if (run_diag and task_first_round and cur_task >= 1
                    and not mid_round_first):
                parts = []
                for t in range(0, cur_task):
                    acc_t = self.clients[TIDR_DIAG_CLIENT] \
                        ._diag_eval_task(t, collect_stats=False)[0]
                    parts.append(f"task{t}_acc_pre_task_switch={acc_t:.2f}")
                print(f"[TIDR-TaskBoundaryDiag] round={i} task={cur_task} "
                      f"client={TIDR_DIAG_CLIENT} " + ' '.join(parts))

            # Update client data for the current round
            for j in range(self.client_num):
                self.clients[j].update_data(round=i, args=self.args)

            # Handle task transition for Office-Home dataset
            if self.task_id != i // self.global_epoch:
                self.task_id = i // self.global_epoch
                if self.args.data_name == 'office_home':
                    datas, mask = self.client_data.random_split(domain=self.task_id)
                    for j in range(self.client_num):
                        self.clients[j].get_data_office_home(self.task_id, datas[j], mask[j])

            print(f"--------round {i}, task number {i//self.global_epoch}-----------")

            # ---- E6a-v3b-Diag: Task Checkpoint 配置注入（Client0 Phase 边界保存） ----
            if save_task_ckpt and task_first_round and cur_task >= 1:
                self.clients[TIDR_DIAG_CLIENT]._task_ckpt_config = {
                    'save': True,
                    'task': cur_task,
                    'server': self,
                }

            # Train each selected client
            for j in self.thisclients:
                self.clients[j].train(round=i, args=self.args)

            # E6a-v2-Diag: 诊断配置仅作用于本轮 Client0 的 train()
            if is_cf_round or is_phase_diag_round:
                self.clients[TIDR_DIAG_CLIENT]._tidr_diag_config = None
            if save_task_ckpt and task_first_round and cur_task >= 1:
                self.clients[TIDR_DIAG_CLIENT]._task_ckpt_config = None

            # Select best prototypes using greedy similarity matching
            self.choose_best_proto_greedy_similarity_fixed_key(
                (i + 2) % self.global_epoch == 0,
                threshold=self.threshold,
                round=i
            )

            # Perform prompt fusion via knowledge distillation
            if i % self.global_epoch != 4:
                self.kd_fusion_prompt(self.thisclients)

            # Aggregate classification heads across clients
            self.fed_avg_head(self.thisclients)

            print('Server aggregation Complete')

            # Distribute global models to clients
            for j in range(self.client_num):
                if j in self.thisclients:
                    self.clients[j].get_global_proto_and_head(
                        self.global_protos, self.global_head, self.prompt, i
                    )
                    print('-------')
                else:
                    self.clients[j].get_global_proto_and_head_no_test(
                        self.global_protos, self.global_head, self.prompt, i
                    )

            # ---- CL 评估协议: Task 最后一轮分发完成后收集 matrix 行 ----
            # （get_global_proto_and_head 内已完成全 seen tasks 评估）
            if (i + 1) % self.global_epoch == 0:
                self._collect_cl_matrix_rows(cur_task)

            # ---- E6a-v2-Diag: CP1（CF 轮前一轮全部完成后：训练+聚合+融合+分发） ----
            # E6a-v3b-Diag: 轮次/文件名跟随 diag_cf_round（默认 5 → R4_complete.pth；
            # Task2 传 10 → R9_complete.pth；resume_point 标签保持 'R4_complete'）
            if i == diag_cf_round - 1 and getattr(self.args, 'save_checkpoints', False):
                save_checkpoint(self, 'R4_complete',
                                f'R{diag_cf_round - 1}_complete.pth', i)
                if stop_at == 'R4_complete':
                    print("[E6a-v2-Diag] Stopped after R4_complete checkpoint")
                    return

            # ---- E6a-v3b-Diag: Task{k}_complete（Task 最后一轮全部完成后） ----
            if save_task_ckpt and (i + 1) % self.global_epoch == 0:
                save_checkpoint(self, 'Task_complete',
                                f'Task{cur_task}_complete.pth', i)

        print("All Process completes")

        # ---- CL 评估协议: 训练结束打印 ACC / F / BWT ----
        self._print_cl_metrics()

    def _collect_cl_matrix_rows(self, cur_task):
        """
        CL 评估协议: 收集各 client 在 Task 末轮生成的 matrix 行

        - 追加写入 cl_matrix CSV（每行: after_task, client_id, task0..task{N-1}，
          未评估的 task 留空）
        - 存入 self.cl_matrix[client_id]（按 after_task 顺序），供训练结束
          计算 ACC / Forgetting F / BWT
        - 打印 [CL-Matrix] 摘要行
        """
        for client in self.clients:
            if client.cl_matrix_row is None:
                continue
            task_id, accs = client.cl_matrix_row
            with open(self.cl_matrix_file, 'a', newline='',
                      encoding='utf-8') as f:
                writer = csv.writer(f)
                row = [f'after_task{task_id}', client.id]
                row += [f'{a:.4f}' for a in accs]
                row += [''] * (self.task_num - len(accs))
                writer.writerow(row)
            self.cl_matrix.setdefault(client.id, []).append(accs)
            acc_str = ' '.join(
                f'task{t}={a:.2f}' for t, a in enumerate(accs))
            print(f"[CL-Matrix] after_task={task_id} client={client.id} "
                  f"{acc_str}")
            client.cl_matrix_row = None

    def _print_cl_metrics(self):
        """
        CL 评估协议: 从 accuracy matrix R[i][j] 计算标准 CL 指标

        - ACC = (1/N) * sum_j R[N-1][j]
        - F_j = max_i R[i][j] - R[N-1][j]（对 j<N-1），F = mean_j F_j
        - BWT = (1/(N-1)) * sum_{j<N-1} (R[N-1][j] - R[j][j])
        """
        if not self.cl_matrix:
            print("[CL-Metrics] 无 matrix 数据（未收集到 Task 末轮评估）")
            return

        print("\n===== CL Metrics (Accuracy Matrix) =====")
        all_acc, all_f, all_bwt = [], [], []
        for client in self.clients:
            rows = self.cl_matrix.get(client.id)
            if not rows:
                continue
            n = len(rows)                      # 已完成的 after_task 数
            last = rows[-1]                    # R[N-1][:]
            num_tasks = len(last)

            # ACC: 最终一行（学完所有任务后）对所有 seen tasks 的均值
            acc = float(np.mean(last))

            # Forgetting / BWT: 只在 matrix 完整（每 task 一行）时计算，
            # 恢复运行产生的部分 matrix 跳过，避免越界/偏差
            if n == num_tasks:
                forgets = [
                    max(rows[i][j] for i in range(n)) - last[j]
                    for j in range(num_tasks - 1)
                ]
                f_mean = float(np.mean(forgets))
                bwt = float(np.mean(
                    [last[j] - rows[j][j] for j in range(num_tasks - 1)]))
            else:
                f_mean = float('nan')
                bwt = float('nan')

            print(f"[CL-Metrics] client={client.id} ACC={acc:.2f} "
                  f"F={f_mean:.2f} BWT={bwt:.2f}")
            all_acc.append(acc)
            if not (f_mean != f_mean):  # nan 检查
                all_f.append(f_mean)
                all_bwt.append(bwt)

        if all_acc:
            msg = (f"[CL-Metrics] MEAN ACC={np.mean(all_acc):.2f}")
            if all_f:
                msg += (f" F={np.mean(all_f):.2f} BWT={np.mean(all_bwt):.2f}")
            print(msg)

    def fuse_protos(self):
        """
        将所有客户端的本地原型聚合为全局原型
        
        计算每个类别在所有客户端上的原型均值
        """
        global_protos = dict()
        temp = dict()
        
        for client in self.clients:
            for label in client.local_protos.keys():
                if label in temp:
                    temp[label].append(client.local_protos[label])
                else:
                    temp[label] = [client.local_protos[label]]
        
        for label in temp.keys():
            global_protos[label] = np.mean(temp[label], axis=0)
            temp[label] = np.vstack(temp[label])
        
        self.global_protos = global_protos
        self.temp_protos = temp

    def choose_best_proto_greedy_similarity_fixed_key(self, is_fix=False, threshold=0.2, round=15):
        """
        使用贪婪相似度匹配为每个类别选择最佳原型
        
        该方法：
        1. 从当前客户端收集原型
        2. 计算余弦相似度矩阵
        3. 选择平均相似度最小的原型（最具判别性）
        4. 固定满足相似度阈值的原型
        
        Args:
            is_fix: 是否固定所有当前原型
            threshold: 原型固定的相似度阈值
            round: 当前训练轮次
        """
        # Initialize global prototypes from previous round or empty dict
        global_protos = self.global_protos if self.global_protos is not None else dict()

        temp = dict()
        this_round = dict()

        # Collect prototypes from current clients (excluding fixed keys)
        for i in self.thisclients:
            for label in self.clients[i].local_protos.keys():
                if label not in self.fix_keys:
                    if label in temp:
                        temp[label].append(self.clients[i].local_protos[label])
                    else:
                        temp[label] = [self.clients[i].local_protos[label]]

                    if not np.isnan(self.clients[i].local_protos[label]).any():
                        if label in this_round:
                            this_round[label].append(self.clients[i].local_protos[label])
                        else:
                            this_round[label] = [self.clients[i].local_protos[label]]

        # 收集每个类别的样本数量（用于 Quality-Weighted Prototype Selection）
        # Determine which keys to process
        if len(this_round.keys()) != 0 and self.fix_keys != []:
            keys = list(this_round.keys())
            keys.extend(self.fix_keys)
        elif len(this_round.keys()) == 0 and self.fix_keys != []:
            keys = self.fix_keys
        else:
            keys = list(this_round.keys())

        # Normalize shapes of global prototypes
        first_shape = None
        for key, value in global_protos.items():
            if first_shape is None:
                first_shape = value.shape
                break
        
        different_shapes = {}
        print(first_shape)
        for key, value in global_protos.items():
            if value.shape != first_shape:
                different_shapes[key] = value.shape
            if len(value.shape) == 3:
                global_protos[key] = global_protos[key].squeeze()

        # Build matrix of all prototypes
        matrix = None
        num = []
        for i in range(len(keys)):
            if keys[i] not in self.fix_keys:
                if matrix is None:
                    matrix = np.array(this_round[keys[i]])
                else:
                    matrix = np.concatenate([matrix, this_round[keys[i]]])

                if keys[i] in global_protos.keys():
                    proto = global_protos[keys[i]]
                    if not isinstance(proto, torch.Tensor):
                        proto = torch.from_numpy(np.array(proto))
                    matrix = np.concatenate([matrix, proto.unsqueeze(0)])
                    num.append(len(this_round[keys[i]]) + 1)
                else:
                    num.append(len(this_round[keys[i]]))
            else:
                proto = self.global_protos[keys[i]]
                if not isinstance(proto, torch.Tensor):
                    proto = torch.from_numpy(np.array(proto))
                if matrix is None:
                    matrix = np.array(proto.unsqueeze(0))
                else:
                    matrix = np.concatenate([matrix, proto.unsqueeze(0)])
                num.append(1)

        # Compute cosine similarity adjacency matrix
        adj_matrix = torch.tensor(cosine_similarity(matrix))
        matrix = torch.Tensor(matrix).to(self.device)

        # Set same-class prototype similarities to 1.0
        low_bound = 0
        for j in num:
            high_bound = low_bound + j
            for h in range(low_bound, high_bound):
                for k in range(low_bound, high_bound):
                    adj_matrix[h][k] = 1.0
            low_bound = high_bound

        # Select best prototype for each class
        low_bound = 0
        map = {}
        for j in range(len(num)):
            if keys[j] not in self.fix_keys:
                high_bound = low_bound + num[j]
                min_sim = 999
                chose = None
                for h in range(low_bound, high_bound):
                    aver_simi = torch.mean(adj_matrix[h])
                    score = aver_simi
                    if score < min_sim:
                        min_sim = aver_simi
                        chose = h
                if min_sim <= threshold:
                    self.fix_keys.append(keys[j])
                map[keys[j]] = min_sim
                new_proto = matrix[chose].cpu()
                global_protos[keys[j]] = new_proto
                low_bound = high_bound
            else:
                high_bound = low_bound + num[j]
                aver_simi = torch.mean(adj_matrix[low_bound])
                map[keys[j]] = aver_simi
                global_protos[keys[j]] = self.global_protos[keys[j]]
                low_bound = high_bound

        self.global_protos = global_protos
        
        # Fix all prototypes if requested
        if is_fix:
            if self.fix_keys != []:
                self.fix_keys.extend(list(global_protos.keys()))
                self.fix_keys = list(set(self.fix_keys))
            else:
                self.fix_keys = list(global_protos.keys())

        print(len(self.fix_keys))

        # Final shape normalization
        first_shape = None
        for key, value in self.global_protos.items():
            if first_shape is None:
                first_shape = value.shape
                break
        
        different_shapes = {}
        for key, value in self.global_protos.items():
            if value.shape != first_shape:
                different_shapes[key] = value.shape
            if len(value.shape) == 3:
                self.global_protos[key] = self.global_protos[key].squeeze()

    def kd_fusion_prompt(self, chosen_clients):
        """
        使用知识蒸馏融合多个客户端的提示
        
        Args:
            chosen_clients: 要融合的客户端索引列表
            
        Returns:
            融合后的提示参数
        """
        if len(chosen_clients) == 1:
            my_result = deepcopy(self.clients[chosen_clients[0]].global_prompt)
        else:
            # Collect all classes from chosen clients
            classes = set()
            for i in chosen_clients:
                temp = set(self.clients[i].current_class)
                classes = set.union(classes, temp)

            self.surrogate_data.getTrainData(list(classes))

            # Initialize with first client's prompts
            my_result = deepcopy(self.clients[chosen_clients[0]].prompts)
            test_loader = DataLoader(
                self.surrogate_data, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=2
            )
            self.model.to(self.device)

            # Fine-tune fused prompts using knowledge distillation
            optimizer = torch.optim.Adam(
                my_result.parameters(), 
                lr=self.lr, 
                weight_decay=1e-03
            )
            
            for h in tqdm(range(10)):
                for iteration, (index, x, y) in enumerate(test_loader):
                    x = Variable(x, requires_grad=True).to(self.device, non_blocking=True)
                    y = y.long().to(self.device)
                    my_result.to(self.device)
                    
                    with torch.no_grad():
                        if self.origin_model is not None:
                            output = self.origin_model(x)
                            cls_features = output['pre_logits']
                        else:
                            cls_features = None

                    # Get output from current fused prompts
                    self.model.load_prompts(my_result)
                    output = self.model.forward_features(
                        x, cls_features=cls_features, train=False
                    )['x']
                    
                    # Get outputs from other clients' prompts for distillation
                    outputs = []
                    with torch.no_grad():
                        for localmodel in chosen_clients[1:]:
                            prompt = self.clients[localmodel].prompts
                            self.model.load_prompts(prompt)
                            my = self.model.forward_features(
                                x, cls_features=cls_features, train=False
                            )['x']
                            outputs.append(my)

                    # Compute distillation loss and update
                    loss = global_distillation_loss(output, outputs)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            self.prompt = my_result
            print('prompt fusion complete')
        
        return my_result

    def start(self):
        """
        启动联邦训练过程
        E6a-v2-Diag: 支持 checkpoint 恢复入口与 --stop_at_checkpoint 停止
        """
        self.init_client()

        # ---- E6a-v2-Diag: checkpoint 恢复入口 ----
        resume_path = getattr(self.args, 'resume_checkpoint', '') or ''
        resume_point = None
        resume_extras = None
        resume_payload = None
        if resume_path:
            resume_point, resume_extras, resume_payload = load_checkpoint(self, resume_path)

        # checkpoint 目录（每次运行独立目录，不覆盖其他实验的 Checkpoint）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._ckpt_dir = os.path.join(
            'checkpoints', 'E6a_v2_diag',
            f"{getattr(self.args, 'data_name', 'unknown')}"
            f"_seed{getattr(self.args, 'seed', 42)}_{timestamp}"
        )

        stop_at = getattr(self.args, 'stop_at_checkpoint', '') or ''
        if stop_at:
            print(f"[E6a-v2-Diag] stop_at_checkpoint={stop_at}")

        try:
            if resume_point == 'R5_C0_post_phase2':
                # CP3: 仅离线诊断（不训练、不聚合、不保存）
                self._run_standalone_diag(
                    resume_extras,
                    round_label=(resume_payload or {}).get(
                        'round', TIDR_DIAG_ROUND))
            elif resume_point == 'Task_C0_post_phase2':
                # E6a-v3b-Diag: Task{k}_C0_post_phase2.pth 离线诊断
                # （与 CP3 同协议；Task 编号在 extras['diag_task']）
                self._run_standalone_diag(
                    resume_extras,
                    round_label=(resume_extras or {}).get('diag_round',
                                                          TIDR_DIAG_ROUND)
                )
            elif resume_point == 'R5_C0_pre_phase2':
                # CP2: CF 轮 Client0 跳过 Phase1 直接进入 Phase2，随后正常继续
                # E6a-v3b-Diag: 恢复起点取自 checkpoint 实际保存轮次
                self.train_clients(
                    start_round=(resume_payload or {}).get(
                        'round', TIDR_DIAG_ROUND),
                    resume_mid_round=True,
                    resume_extras=resume_extras)
            elif resume_point == 'Task_C0_post_phase1':
                # E6a-v3b-Diag: Task{k}_C0_post_phase1.pth mid-round 恢复
                # （Phase1 已完成，跳过 Phase1 直接进入 Phase2，与 CP2 同协议；
                #   不依赖 --run_tidr_diagnostics，与 --save_task_checkpoints 解耦）
                self.train_clients(
                    start_round=(resume_payload or {}).get('round', 0),
                    resume_mid_round=True,
                    resume_extras=resume_extras)
            elif resume_point == 'R4_complete':
                # CP1: 从 CF 轮开始，不重复前一轮
                self.train_clients(
                    start_round=(resume_payload or {}).get(
                        'round', TIDR_DIAG_ROUND - 1) + 1)
            elif resume_point == 'Task_start':
                # E6a-v3b-Diag: Task{k}_start.pth 保存于 Round k*ge 循环体开始、
                # update_data 之前 → 从该 Round 原样继续（数据 split 一致）
                self.train_clients(
                    start_round=(resume_payload or {}).get('round', 0)
                )
            elif resume_point == 'Task_complete':
                # Task{k}_complete.pth 保存于 Task 最后一轮全部完成后
                # → 从下一 Round 继续
                self.train_clients(
                    start_round=(resume_payload or {}).get('round', 0) + 1
                )
            else:
                self.train_clients()
        except DiagStopException as e:
            print(f"[E6a-v2-Diag] Reached stop_at_checkpoint: {e.cp_name}")

    def _install_tidr_diag_config(self, cf=True):
        """
        E6a-v2-Diag / E6a-v3b-Diag: 向 Client0 注入诊断配置
        仅 --run_tidr_diagnostics 开启时生效。
        （mid-round 恢复的 Phase2 上下文注入见 _install_phase2_resume_ctx，
          与本方法解耦，不依赖诊断开关）

        cf=True: 完整反事实套件轮（diag_cf_round，含 CF + CP2/CP3 保存）
        cf=False: 轻量 PhaseDiag 轮（其他任务首轮，仅三点准确率 + drift）
        """
        if not getattr(self.args, 'run_tidr_diagnostics', False):
            return

        client0 = self.clients[TIDR_DIAG_CLIENT]
        client0._tidr_diag_config = {
            'save': getattr(self.args, 'save_checkpoints', False) and cf,
            'stop_at': (getattr(self.args, 'stop_at_checkpoint', '') or '')
                       if cf else '',
            'ckpt_dir': getattr(self, '_ckpt_dir', None),
            'server': self,
            'standalone': False,
            'cf': cf,
        }

    def _install_phase2_resume_ctx(self, resume_extras):
        """
        E6a-v3b-Diag: mid-round 恢复（CP2 / Task{k}_C0_post_phase1）时
        向 Client0 注入 Phase2 上下文与诊断快照。

        与 --run_tidr_diagnostics 解耦：仅 --save_task_checkpoints 保存的
        post-P1 checkpoint 也能独立恢复（跳过 Phase1 直接进入 Phase2）。
        """
        if not resume_extras:
            return
        client0 = self.clients[TIDR_DIAG_CLIENT]
        client0._resume_phase2_ctx = {
            'phase2_global_step': resume_extras['phase2_global_step'],
            'total_steps': resume_extras['total_steps'],
        }
        client0._diag_key_before_phase2 = resume_extras.get('key_before_phase2')
        client0._diag_anchor_before_phase2 = resume_extras.get('anchor_before_phase2')
        client0._diag_old_key_mask = resume_extras.get('old_key_mask')
        client0._diag_feature_before = resume_extras.get('feature_before')
        client0._diag_feature_samples = resume_extras.get('feature_sample_indices')
        client0._diag_phase_acc_pre = resume_extras.get('phase_acc_pre') or {}
        # E6a-v3b-Diag: Phase1 前三点准确率（CP2 恢复时 Phase1 已在原运行完成）
        client0._diag_phase_acc_pre_p1 = (
            resume_extras.get('phase_acc_pre_p1') or {}
        )
        client0._diag_phase_stats_pre = resume_extras.get('phase_stats_pre')
        # E6a-v3b-Diag: 诊断快照（P1 窗口 drift / 固定输入张量）
        client0._diag_anchor_feat_before_phase2 = (
            resume_extras.get('anchor_feat_before_phase2')
        )
        client0._diag_feature_pre_p1 = resume_extras.get('feature_pre_p1')
        client0._diag_anchor_feat_pre_p1 = (
            resume_extras.get('anchor_feat_pre_p1')
        )
        client0._diag_key_pre_p1 = resume_extras.get('key_pre_p1')
        client0._diag_anchor_pre_p1 = resume_extras.get('anchor_pre_p1')
        client0._diag_head_pre_p1 = resume_extras.get('head_pre_p1')
        client0._diag_heads_pre_p1 = resume_extras.get('heads_pre_p1')
        client0._diag_fixed_inputs = resume_extras.get('fixed_inputs')
        client0._diag_fixed_targets = resume_extras.get('fixed_targets')

    def _run_cp3_standalone_diag(self, extras):
        """向后兼容入口（转调 _run_standalone_diag）"""
        self._run_standalone_diag(extras, round_label=TIDR_DIAG_ROUND)

    def _run_standalone_diag(self, extras, round_label=TIDR_DIAG_ROUND):
        """
        E6a-v2-Diag / E6a-v3b-Diag: 离线诊断入口
        （CP3 或 Task{k}_C0_post_phase2.pth）
        直接加载 Phase2 后状态重新评估 Task0（可复现保存时准确率），
        输出全部诊断；不训练、不聚合、不重新保存 Checkpoint
        """
        if not extras:
            print("[E6a-v2-Diag] standalone extras 缺失（key_before/key_after），跳过诊断")
            return

        client0 = self.clients[TIDR_DIAG_CLIENT]

        # ============================================================
        # standalone device protection
        # （Checkpoint 恢复时模块在 CPU 上，诊断前需移回训练设备）
        # ============================================================
        client0.model.to(client0.device)
        client0.vit.to(client0.device)

        if client0.original_model is not None:
            client0.original_model.to(client0.device)
            client0.original_model.eval()

        client0._tidr_diag_config = {
            'save': False,
            'stop_at': '',
            'ckpt_dir': None,
            'server': self,
            'standalone': True,
            'cf': True,
        }
        client0._diag_key_before_phase2 = extras.get('key_before_phase2')
        client0._diag_key_after_phase2 = extras.get('key_after_phase2')
        # 旧 CP3 无此字段 → None → Client_DF 自动 fallback 到 prev_anchor_pool
        # （prev_anchor_pool 在本轮开始 clone 保存，Phase1 不动 Tail Anchor，
        #   等价于 Phase2 前 Anchor，可用于 AnchorCF / AnchorDrift）
        client0._diag_anchor_before_phase2 = extras.get('anchor_before_phase2')
        client0._diag_old_key_mask = extras.get('old_key_mask')
        client0._diag_feature_before = extras.get('feature_before')
        client0._diag_feature_samples = extras.get('feature_sample_indices')
        client0._diag_phase_acc_pre = extras.get('phase_acc_pre') or {}
        client0._diag_phase_stats_pre = extras.get('phase_stats_pre')
        client0._diag_phase_acc_pre_p1 = extras.get('phase_acc_pre_p1') or {}
        # E6a-v3b-Diag: 诊断快照（P1 窗口 drift / 固定输入张量），
        # 使离线复跑能完整复现 P1Drift / HeadDrift / RetrievedAnchorDrift
        client0._diag_anchor_feat_before_phase2 = (
            extras.get('anchor_feat_before_phase2')
        )
        client0._diag_feature_pre_p1 = extras.get('feature_pre_p1')
        client0._diag_anchor_feat_pre_p1 = extras.get('anchor_feat_pre_p1')
        client0._diag_key_pre_p1 = extras.get('key_pre_p1')
        client0._diag_anchor_pre_p1 = extras.get('anchor_pre_p1')
        client0._diag_head_pre_p1 = extras.get('head_pre_p1')
        client0._diag_heads_pre_p1 = extras.get('heads_pre_p1')
        client0._diag_fixed_inputs = extras.get('fixed_inputs')
        client0._diag_fixed_targets = extras.get('fixed_targets')

        print(f"[E6a-v2-Diag] standalone diagnostics "
              f"(client={TIDR_DIAG_CLIENT}, round={round_label}, "
              f"task={extras.get('diag_task', 'n/a')})")
        client0._tidr_diag_phase2_end(round_label, self.args)
        client0._tidr_diag_config = None

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """
        使用 L2 归一化对张量进行归一化
        
        Args:
            x: 输入张量
            dim: 归一化维度
            epsilon: 数值稳定性的小值
            
        Returns:
            L2 归一化后的张量
        """
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def fed_avg_prompt(self, chosen_clients):
        """
        对选定客户端的提示进行平均
        
        Args:
            chosen_clients: 要平均的客户端索引列表
            
        Returns:
            平均后的提示参数
        """
        prompts = [self.clients[i].prompts for i in chosen_clients]

        result_prompt = deepcopy(prompts[0].state_dict())

        for k in result_prompt.keys():
            for i in range(len(prompts)):
                local_model_params = prompts[i].state_dict()
                if i == 0:
                    result_prompt[k] = local_model_params[k]
                else:
                    result_prompt[k] += local_model_params[k]
            result_prompt[k] = result_prompt[k] / len(chosen_clients)

        self.prompt = deepcopy(self.clients[0].prompts)
        self.prompt.load_state_dict(result_prompt)
        return self.prompt

    def fed_avg_head(self, chosen_clients):
        """
        对选定客户端的 Input Enhancement 分类头 (vit.head) 进行联邦平均

        原始 FedTA 聚合的是 vit.head（Stage 1 Input Enhancement 的分类头），
        而非 Tail Anchor 的 model.head（Stage 2 按 task 独立保存）。

        Args:
            chosen_clients: 要平均的客户端索引列表
        """
        heads = [self.clients[i].vit.head for i in chosen_clients]

        result_head = deepcopy(heads[0].state_dict())

        for k in result_head.keys():
            for i in range(len(heads)):
                local_model_params = heads[i].state_dict()
                if i == 0:
                    result_head[k] = local_model_params[k]
                else:
                    result_head[k] += local_model_params[k]
            result_head[k] = result_head[k] / len(chosen_clients)

        self.global_head = deepcopy(self.clients[0].vit.head)
        self.global_head.load_state_dict(result_head)