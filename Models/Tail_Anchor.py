"""
FedSMR-v2: Tail_Anchor 模型模块 (Residual Soft Anchor + Seen-Only Diversity)
"""

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.classification_head import Chead


class Tail_Anchor(nn.Module):

    def __init__(self, anchor_size, key_size, nb_class,
                 soft_anchor=True, soft_temperature=0.1,
                 soft_anchor_ratio=0.25,
                 diversity_margin=0.2):
        super(Tail_Anchor, self).__init__()
        self.size = anchor_size
        self.key_size = key_size
        self.nb_class = nb_class
        self.soft_anchor = soft_anchor
        self.soft_temperature = soft_temperature
        self.soft_anchor_ratio = soft_anchor_ratio
        self.diversity_margin = diversity_margin

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

        # 已见类别掩码 (seen-only diversity)
        self.register_buffer('seen_class_mask', torch.zeros(nb_class, dtype=torch.bool))

    def set_seen_classes(self, class_indices):
        for c in class_indices:
            if 0 <= c < self.nb_class:
                self.seen_class_mask[c] = True

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x, class_mask):
        """
        E3 plain Residual Soft-Anchor forward.

        Returns: logits, output_mixed, reduce_sim, anchor_feat, attn_weights, hard_idx, routing_logits

        Design:
        - Hard route: always local-key similarity (FedTA baseline)
        - Soft route: residual soft-anchor with stop-gradient on residual
          (legacy E1/E2/E3 path)
        """
        # ---- 1. similarity ----
        x_embed_norm = self.l2_normalize(x, dim=1)
        key_norm = self.l2_normalize(self.key.reshape(-1, self.key_size), dim=1).to(x.device)
        similarity = torch.matmul(x_embed_norm, key_norm.t())

        anchor_pool_raw = self.anchor_pool.reshape(-1, self.key_size).to(x.device)

        # ---- 2. Hard route: always local-key similarity (FedTA baseline) ----
        routing_sim = similarity
        hard_idx = routing_sim.argmax(dim=1)
        hard_anchor = anchor_pool_raw[hard_idx]

        if self.soft_anchor:
            routing_logits = routing_sim / self.soft_temperature

            # ---- Soft attention ----
            soft_attn = torch.softmax(
                routing_logits,
                dim=1
            )

            # Legacy E1/E2/E3: stop-gradient on full residual
            # Forward: hard + γ·(soft-hard) ≡ (1-γ)·hard + γ·soft
            # Backward: only hard_anchor receives CE gradient; soft branch detached
            soft_anchor = torch.matmul(soft_attn, anchor_pool_raw)
            soft_residual = (soft_anchor - hard_anchor).detach()
            anchor_feat = hard_anchor + self.soft_anchor_ratio * soft_residual

            if self.training:
                hard_batch_usage = torch.bincount(hard_idx, minlength=self.nb_class).float().detach()
                self.anchor_hard_usage += hard_batch_usage.to(self.anchor_hard_usage.device)
                soft_batch_usage = soft_attn.sum(dim=0).detach()
                self.anchor_usage += soft_batch_usage.to(self.anchor_usage.device)

            attn_weights = soft_attn

        else:
            # Hard Anchor only (FedTA baseline when soft_anchor=False)
            routing_logits = routing_sim

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