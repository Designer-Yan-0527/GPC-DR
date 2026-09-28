"""
E6a-v2-Diag: Checkpoint 断点恢复工具

三个 Checkpoint:
  CP1: R4_complete.pth       — Round4 全部完成（客户端训练 + 服务器聚合 +
                               Prompt Fusion + 状态分发）后保存；
                               恢复后直接从 Round5 开始，不重复 Round4
  CP2: R5_C0_pre_phase2.pth  — Round5 Client0 Task1 Phase1 结束、Phase2 全部
                               初始化完成、第一个 Phase2 batch 之前保存；
                               恢复后跳过 Phase1 直接执行 Phase2
  CP3: R5_C0_post_phase2.pth — Round5 Client0 Task1 Phase2 结束、服务器聚合
                               之前保存；可直接加载做 Task0 离线诊断

设计说明:
- 不依赖 Models 包（避免循环导入），只做状态收集/恢复
- 所有客户端共享同一个 vit 对象（Server_DF.model），vit 状态只保存/恢复一次
- global_head / prompt 等模块对象在聚合后架构可能变化
  （global_head 首次 fed_avg_head 后变为 nn.Linear），因此整对象 pickle 而非
  state_dict + 模板重建
- CP1/CP2/CP3 保存点均处于"每 Phase 重建 fresh Adam、无 optimizer/scheduler
  残留状态"的位置，因此无需保存 optimizer/scheduler
- random_split 产生的 train/test Subset 通过 .indices 精确保存重建
- RNG 状态（torch/numpy/python/cuda）在恢复流程的最后一步还原，
  保证后续 DataLoader shuffle / 数据增强与原运行一致
"""

import os
import random
from copy import deepcopy

import numpy as np
import torch

# 诊断目标位置（global_epoch=5 时 round 5 即 Task1 第 1 轮，与现有日志一致）
TIDR_DIAG_ROUND = 5
TIDR_DIAG_CLIENT = 0
TIDR_DIAG_TASK = 1

CKPT_FORMAT = 'E6a-v2-Diag/1'
CKPT_CP1 = 'R4_complete.pth'
CKPT_CP2 = 'R5_C0_pre_phase2.pth'
CKPT_CP3 = 'R5_C0_post_phase2.pth'
CKPT_RESUME_POINTS = (
    'R4_complete', 'R5_C0_pre_phase2', 'R5_C0_post_phase2',
    # E6a-v3b-Diag: Task 边界 Checkpoint（--save_task_checkpoints）
    'Task_C0_post_phase1', 'Task_C0_post_phase2',
)


class DiagStopException(Exception):
    """--stop_at_checkpoint 触发时抛出，由 Server_DF.start() 捕获后正常退出"""

    def __init__(self, cp_name):
        super().__init__(cp_name)
        self.cp_name = cp_name


def _torch_load(path):
    """兼容新旧版本 torch 的加载（weights_only 参数在旧版本不存在）"""
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def get_rng_state():
    """收集全部 RNG 状态（torch / numpy / python / cuda）"""
    state = {
        'torch': torch.get_rng_state(),
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    else:
        state['cuda'] = None
    return state


def set_rng_state(state):
    """还原全部 RNG 状态（必须在所有模型/数据状态恢复完成后最后调用）"""
    torch.set_rng_state(state['torch'].cpu())
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])


def _cpu_state_dict(module):
    """state_dict 全部转到 CPU（恢复时 load_state_dict 支持跨设备 copy）"""
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


# ============================================================================
# Client 状态
# ============================================================================

def collect_client_state(client):
    """收集单个 Client_DF 的完整可恢复状态"""
    test_split_indices = []
    for tl in client.test_loader:
        indices = getattr(tl, 'indices', None)
        test_split_indices.append(list(indices) if indices is not None else None)

    traindata = getattr(client, 'traindata', None)
    train_split_indices = None
    if traindata is not None and hasattr(traindata, 'indices'):
        train_split_indices = list(traindata.indices)

    return {
        'task_id': client.task_id,
        'round': client.round,
        # Tail_Anchor（key / anchor_pool / head / usage buffers / seen mask）
        'model_state': _cpu_state_dict(client.model),
        # per-task head 快照（Chead 模块对象，整对象保存）
        'heads': [deepcopy(h) if h is not None else None for h in client.heads],
        'head': deepcopy(client.head),
        # 该客户端上一轮结束时的 prompt（Global_Prompt 模块对象）
        'prompts_obj': deepcopy(client.prompts) if client.prompts is not None else None,
        # 服务器分发的聚合头（首次 fed_avg_head 后为 nn.Linear，整对象保存）
        'global_head_obj': deepcopy(getattr(client, 'global_head', None)),
        # 类别状态
        'seen_classes': sorted(client.seen_classes),
        'old_seen_classes': sorted(client.old_seen_classes),
        # usage / 上一轮记忆快照
        'task_anchor_usage': {t: v.cpu() for t, v in client.task_anchor_usage.items()},
        'prev_anchor_pool': client.prev_anchor_pool.clone() if client.prev_anchor_pool is not None else None,
        'prev_key_pool': client.prev_key_pool.clone() if client.prev_key_pool is not None else None,
        'prev_anchor_usage': (
            client.prev_anchor_usage.clone()
            if getattr(client, 'prev_anchor_usage', None) is not None else None
        ),
        # 原型
        'local_protos': deepcopy(client.local_protos),
        'global_protos': deepcopy(client.global_protos),
        # 数据 split（random_split 的 Subset.indices，精确重建）
        'test_split_indices': test_split_indices,
        'train_split_indices': train_split_indices,
    }


def restore_client_state(client, cs):
    """恢复单个 Client_DF 的状态（在 server.model/vit 恢复之后调用）"""
    from torch.utils.data import Subset

    client.task_id = cs['task_id']
    client.round = cs['round']
    client.model.load_state_dict(cs['model_state'])
    client.heads = [deepcopy(h) if h is not None else None for h in cs['heads']]
    client.head = deepcopy(cs['head'])
    client.prompts = deepcopy(cs['prompts_obj']) if cs['prompts_obj'] is not None else None
    if cs['global_head_obj'] is not None:
        client.global_head = deepcopy(cs['global_head_obj'])
    client.seen_classes = set(cs['seen_classes'])
    client.old_seen_classes = set(cs['old_seen_classes'])
    client.task_anchor_usage = {t: v.clone() for t, v in cs['task_anchor_usage'].items()}
    client.prev_anchor_pool = cs['prev_anchor_pool'].clone() if cs['prev_anchor_pool'] is not None else None
    client.prev_key_pool = cs['prev_key_pool'].clone() if cs['prev_key_pool'] is not None else None
    if cs['prev_anchor_usage'] is not None:
        client.prev_anchor_usage = cs['prev_anchor_usage'].clone()
    client.local_protos = deepcopy(cs['local_protos'])
    client.global_protos = deepcopy(cs['global_protos'])

    # 重建 test_loader（保持与原运行完全相同的 split）
    client.test_loader = []
    for t, indices in enumerate(cs['test_split_indices']):
        if indices is None:
            break
        client.test_loader.append(Subset(client.train_data[t], indices))

    # 重建当前任务的 traindata / train_dataset / current_class
    if cs['train_split_indices'] is not None and cs['task_id'] >= 0:
        client.traindata = Subset(client.train_data[cs['task_id']],
                                  cs['train_split_indices'])
        client.train_dataset = client.train_data[cs['task_id']]
        client.current_class = client.class_mask[cs['task_id']]


# ============================================================================
# Server 状态
# ============================================================================

def collect_server_state(server):
    """收集 Server_DF 的完整可恢复状态（含共享 vit）"""
    return {
        'task_id': server.task_id,
        'global_protos': deepcopy(server.global_protos),
        'temp_protos': deepcopy(server.temp_protos),
        'fix_keys': list(server.fix_keys),
        'global_head_obj': deepcopy(server.global_head),
        'prompt_obj': deepcopy(getattr(server, 'prompt', None)),
        # 共享 vit（所有客户端的 self.vit 即此对象），保存一次全客户端生效
        'model_state': _cpu_state_dict(server.model),
    }


def restore_server_state(server, ss):
    """恢复 Server_DF 状态（共享 vit 恢复一次即对所有客户端生效）"""
    server.task_id = ss['task_id']
    server.global_protos = deepcopy(ss['global_protos'])
    server.temp_protos = deepcopy(ss['temp_protos'])
    server.fix_keys = list(ss['fix_keys'])
    server.global_head = deepcopy(ss['global_head_obj'])
    if ss['prompt_obj'] is not None:
        server.prompt = deepcopy(ss['prompt_obj'])
    server.model.load_state_dict(ss['model_state'])


# ============================================================================
# 保存 / 加载
# ============================================================================

def save_checkpoint(server, resume_point, filename, round_num, extras=None):
    """
    保存 checkpoint（原子写：tmp + os.replace，避免写一半损坏）

    Args:
        server: Server_DF 实例（需已设置 server._ckpt_dir）
        resume_point: 'R4_complete' | 'R5_C0_pre_phase2' | 'R5_C0_post_phase2'
        filename: 目标文件名
        round_num: 当前全局轮次
        extras: 附加诊断状态（Key 快照 / 特征快照 / Phase2 步数等）
    """
    ckpt_dir = getattr(server, '_ckpt_dir', None)
    if ckpt_dir is None:
        raise RuntimeError('server._ckpt_dir 未初始化（应由 Server_DF.start() 设置）')
    os.makedirs(ckpt_dir, exist_ok=True)

    payload = {
        'format': CKPT_FORMAT,
        'resume_point': resume_point,
        'ckpt_dir': ckpt_dir,
        'round': round_num,
        'rng': get_rng_state(),
        'server': collect_server_state(server),
        'clients': [collect_client_state(c) for c in server.clients],
        'extras': extras or {},
    }
    path = os.path.join(ckpt_dir, filename)
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)
    print(f"[E6a-v2-Diag] Checkpoint saved: {path} (resume_point={resume_point}, round={round_num})")
    return path


def load_checkpoint(server, path):
    """
    加载 checkpoint 并恢复 server + 全部 clients + 共享 vit + RNG

    恢复顺序: server（含共享 vit）→ clients → RNG（最后，避免恢复过程消耗随机数）

    Returns:
        (resume_point, extras, payload)
    """
    payload = _torch_load(path)
    if payload.get('format') != CKPT_FORMAT:
        raise RuntimeError(f"不支持的 checkpoint 格式: {payload.get('format')}")

    restore_server_state(server, payload['server'])
    for client, cs in zip(server.clients, payload['clients']):
        restore_client_state(client, cs)
    set_rng_state(payload['rng'])

    resume_point = payload['resume_point']
    print(f"[E6a-v2-Diag] Checkpoint loaded: {path} "
          f"(resume_point={resume_point}, round={payload['round']})")
    return resume_point, payload.get('extras') or {}, payload
