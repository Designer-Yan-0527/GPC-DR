"""
FedSMR-v2: Tail_Anchor 模型模块 (Residual Soft Anchor + Seen-Only Diversity)
"""

from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.classification_head import Chead


class Tail_Anchor(nn.Module):

    def __init__(self, anchor_size, key_size, nb_class,
                 soft_anchor=True, soft_temperature=0.1,
                 soft_anchor_ratio=0.25,
                 top_k_anchor=None, temperature_anneal=False,
                 use_sparse_softmax=False,
                 diversity_margin=0.2,
                 use_seen_routing=False,
                 # GPC-DR
                 use_proto_calibration=False,
                 proto_beta=0.5,
                 proto_temperature=0.10,
                 adaptive_gamma=True,
                 gamma_max=0.35):
        super(Tail_Anchor, self).__init__()
        self.size = anchor_size
        self.key_size = key_size
        self.nb_class = nb_class
        self.soft_anchor = soft_anchor
        self.soft_temperature = soft_temperature
        self.soft_anchor_ratio = soft_anchor_ratio
        self.top_k_anchor = top_k_anchor
        self.temperature_anneal = temperature_anneal
        self.use_sparse_softmax = use_sparse_softmax
        self.diversity_margin = diversity_margin
        self.use_seen_routing = use_seen_routing

        # GPC-DR
        self.use_proto_calibration = use_proto_calibration
        self.proto_beta = proto_beta
        self.proto_temperature = proto_temperature
        self.adaptive_gamma = adaptive_gamma
        self.gamma_max = gamma_max

        self.temp_min = 0.03
        self.temp_max = max(soft_temperature, 0.5)

        # Hard 使用频率追踪 (argmax-based, 不受 soft attention 污染)
        self.register_buffer('anchor_hard_usage', torch.zeros(nb_class))
        self.register_buffer('anchor_usage', torch.zeros(nb_class))

        # Key Pool
        self.key = nn.Parameter(torch.randn(nb_class, key_size))
        nn.init.uniform_(self.key, -1, 1)

        # P0.3 FIX: nb_class instead of hardcoded 200
        self.head = Chead(nb_class)

        # Anchor Pool (raw, 不归一化 — 分类头使用原始模长)
        self.anchor_pool = nn.Parameter(torch.randn(nb_class, key_size))

        # 已见类别掩码 (seen-only diversity + seen-class routing)
        self.register_buffer('seen_class_mask', torch.zeros(nb_class, dtype=torch.bool))

    def set_seen_classes(self, class_indices):
        for c in class_indices:
            if 0 <= c < self.nb_class:
                self.seen_class_mask[c] = True

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def compute_temperature(self, global_step=None, total_steps=None):
        if self.temperature_anneal and global_step is not None and total_steps is not None:
            progress = min(global_step / total_steps, 1.0)
            return self.temp_min + 0.5 * (self.temp_max - self.temp_min) * (1.0 + np.cos(np.pi * progress))
        return self.soft_temperature

    def _get_routing_similarity(self, similarity):
        """Seen-class routing (controlled by --use_seen_routing)"""
        if not self.use_seen_routing:
            return similarity
        valid_mask = self.seen_class_mask.to(similarity.device)
        if valid_mask.any():
            return similarity.masked_fill(~valid_mask.unsqueeze(0), float('-inf'))
        return similarity

    def forward(self, x, class_mask, global_step=None, total_steps=None,
                proto_bank=None, proto_valid_mask=None):
        """
        GPC-DR: Global Prototype-Calibrated Differentiable Retrieval

        Returns: logits, output_mixed, reduce_sim, anchor_feat, attn_weights, hard_idx, routing_logits

        Key innovations:
        - Proto-calibrated routing: r = s^K/τ_K + β·s^P/τ_P
        - Differentiable soft anchor: q @ A.detach() — CE → routing, anchor protected
        - Confidence-gated adaptive γ: γ_i = γ_max·(1 - H(q_i)/log C)
        """
        # ---- 1. similarity ----
        x_embed_norm = self.l2_normalize(x, dim=1)
        key_norm = self.l2_normalize(self.key.reshape(-1, self.key_size), dim=1).to(x.device)
        similarity = torch.matmul(x_embed_norm, key_norm.t())

        anchor_pool_raw = self.anchor_pool.reshape(-1, self.key_size).to(x.device)

        # ---- 2. Routing logits ----
        routing_sim = self._get_routing_similarity(similarity)

        if self.soft_anchor:
            temp = self.compute_temperature(global_step, total_steps)
            routing_logits = routing_sim / temp

            # GPC: global prototype-calibrated routing
            if self.use_proto_calibration and proto_bank is not None and proto_valid_mask is not None:
                proto_norm = F.normalize(proto_bank, dim=1)
                proto_sim = torch.matmul(x_embed_norm, proto_norm.t().to(x.device))
                routing_logits[:, proto_valid_mask] += (
                    self.proto_beta * proto_sim[:, proto_valid_mask] / self.proto_temperature
                )

            # ---- Hard idx from calibrated routing ----
            hard_idx = routing_logits.argmax(dim=1)
            hard_anchor = anchor_pool_raw[hard_idx]

            # ---- Soft attention ----
            if self.use_sparse_softmax and self.top_k_anchor is not None:
                top_k = min(self.top_k_anchor, routing_logits.shape[1])
                topk_logits, topk_idx = torch.topk(routing_logits, k=top_k, dim=1)
                soft_attn = torch.zeros_like(routing_logits)
                soft_attn.scatter_(1, topk_idx, torch.softmax(topk_logits, dim=1))
                soft_attn = soft_attn / (soft_attn.sum(dim=1, keepdim=True) + 1e-8)
            else:
                soft_attn = torch.softmax(routing_logits, dim=1)

            # ---- Differentiable soft anchor (anchor_pool detached, routing through q) ----
            soft_anchor = torch.matmul(soft_attn, anchor_pool_raw.detach())

            # ---- Confidence-gated adaptive γ ----
            if self.adaptive_gamma:
                entropy = -(soft_attn * torch.log(soft_attn + 1e-8)).sum(dim=1)
                entropy_norm = entropy / np.log(self.nb_class)
                confidence = (1.0 - entropy_norm).detach()
                gamma = (self.gamma_max * confidence).unsqueeze(1)
            else:
                gamma = self.soft_anchor_ratio

            anchor_feat = hard_anchor + gamma * (soft_anchor - hard_anchor)

            if self.training:
                hard_batch_usage = torch.bincount(hard_idx, minlength=self.nb_class).float().detach()
                self.anchor_hard_usage += hard_batch_usage.to(self.anchor_hard_usage.device)
                soft_batch_usage = soft_attn.sum(dim=0).detach()
                self.anchor_usage += soft_batch_usage.to(self.anchor_usage.device)

            attn_weights = soft_attn

        else:
            # Hard Anchor only (FedTA baseline when soft_anchor=False)
            hard_idx = routing_sim.argmax(dim=1)
            hard_anchor = anchor_pool_raw[hard_idx]
            routing_logits = routing_sim  # for PCR loss compatibility

            anchor_feat = hard_anchor

            if self.training:
                hard_batch_usage = torch.bincount(hard_idx, minlength=self.nb_class).float().detach()
                self.anchor_hard_usage += hard_batch_usage.to(self.anchor_hard_usage.device)

            attn_weights = torch.zeros(similarity.shape[0], similarity.shape[1], device=x.device)
            attn_weights.scatter_(1, hard_idx.unsqueeze(1), 1.0)
            if self.training:
                self.anchor_usage += attn_weights.sum(dim=0).detach().to(self.anchor_usage.device)

        # ---- 3. Hard-key pull (FedTA baseline, unchanged) ----
        batched_key_norm = key_norm[hard_idx]
        reduce_sim = torch.sum(x_embed_norm * batched_key_norm.squeeze(1)) / self.key_size

        # ---- 4. classifier ----
        output_mixed = torch.stack((x, anchor_feat), dim=1).view(-1, self.key_size * 2)
        logits = self.head(output_mixed)

        return logits, output_mixed, reduce_sim, anchor_feat, attn_weights, hard_idx, routing_logits

    def compute_similarity(self, x):
        x_norm = self.l2_normalize(x, dim=1)
        key_norm = self.l2_normalize(self.key.reshape(-1, self.key_size), dim=1)
        return torch.matmul(x_norm, key_norm.t().to(x.device))

    def anchor_diversity_loss(self):
        """Seen-Only Diversity with margin — 只约束已见类别，对角线不参与 mean"""
        seen_indices = self.seen_class_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(seen_indices) < 2:
            return torch.tensor(0.0, device=self.anchor_pool.device)

        seen_anchors = self.anchor_pool[seen_indices]
        anchor_norm = self.l2_normalize(seen_anchors.reshape(-1, self.key_size), dim=1)
        sim_matrix = anchor_norm @ anchor_norm.T
        n = sim_matrix.size(0)

        # 建议9: 严格 off-diagonal mean
        off_diag = ~torch.eye(n, dtype=torch.bool, device=sim_matrix.device)
        pair_sim = sim_matrix[off_diag]
        return F.relu(pair_sim - self.diversity_margin).mean()

    def get_anchor_usage(self):
        return self.anchor_hard_usage.clone()

    def reset_anchor_usage(self):
        self.anchor_hard_usage.zero_()
        self.anchor_usage.zero_()

    def load_head(self, head):
        self.head = deepcopy(head)

    def get_head(self):
        return self.head