"""
CIFAR100 Delay 配置模块

该模块定义了 CIFAR100 延迟实验的命令行参数。

参数分类：
- 训练参数：batch-size, epochs
- 模型参数：model, input-size, pretrained, drop, drop-path
- 优化器参数：opt, opt-eps, opt-betas, clip-grad, momentum, weight-decay
- 学习率调度：sched, lr, warmup-lr, min-lr, decay-epochs
- 数据增强：color-jitter, aa, smoothing, train-interpolation
- 随机擦除：reprob, remode, recount
- 数据参数：data-path, dataset, shuffle, output_dir, device, seed
- 持续学习：train_mask, task_inc
- Prompt 参数：size, length, top_k, initializer, prompt_key
- ViT 参数：global_pool, head_type, freeze
- FedTA 参数：method, client_num, task_num, private_class_num
"""

import argparse


def get_args_parser(subparsers):
    """
    添加 CIFAR100 Delay 实验的命令行参数
    
    Args:
        subparsers: argparse 子解析器
    """
    # 训练参数
    subparsers.add_argument('--batch-size', default=16, type=int, help='每个设备的批次大小')
    subparsers.add_argument('--epochs', default=5, type=int, help='训练轮数')

    # 模型参数
    subparsers.add_argument('--model', default='vit_base_patch16_224', type=str, metavar='MODEL', 
                           help='训练模型名称')
    subparsers.add_argument('--input-size', default=224, type=int, help='图像输入尺寸')
    subparsers.add_argument('--pretrained', default=True, help='是否加载预训练模型')
    subparsers.add_argument('--drop', type=float, default=0.0, metavar='PCT', help='Dropout 率')
    subparsers.add_argument('--drop-path', type=float, default=0.0, metavar='PCT', help='Drop path 率')

    # 优化器参数
    subparsers.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='优化器类型')
    subparsers.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='优化器 epsilon')
    subparsers.add_argument('--opt-betas', default=(0.9, 0.999), type=float, nargs='+', metavar='BETA', 
                           help='优化器 beta 参数')
    subparsers.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM', help='梯度裁剪范数')
    subparsers.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD 动量')
    subparsers.add_argument('--weight-decay', type=float, default=0.0, help='权重衰减')
    subparsers.add_argument('--reinit_optimizer', type=bool, default=True, help='是否重新初始化优化器')

    # 学习率调度参数
    subparsers.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='学习率调度器')
    subparsers.add_argument('--lr', type=float, default=0.001, metavar='LR', help='学习率')
    subparsers.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', 
                           help='学习率噪声开关百分比')
    subparsers.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='学习率噪声限制')
    subparsers.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='学习率噪声标准差')
    subparsers.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='热身学习率')
    subparsers.add_argument('--min-lr', type=float, default=1e-5, metavar='LR', help='最小学习率')
    subparsers.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='学习率衰减间隔')
    subparsers.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='热身轮数')
    subparsers.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='冷却轮数')
    subparsers.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='Plateau 调度器耐心轮数')
    subparsers.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='学习率衰减率')
    subparsers.add_argument('--unscale_lr', type=bool, default=True, help='是否按批次大小缩放学习率')

    # 数据增强参数
    subparsers.add_argument('--color-jitter', type=float, default=None, metavar='PCT', help='颜色抖动因子')
    subparsers.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='使用 AutoAugment 策略')
    subparsers.add_argument('--smoothing', type=float, default=0.1, help='标签平滑')
    subparsers.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='训练插值方法')

    # 随机擦除参数
    subparsers.add_argument('--reprob', type=float, default=0.0, metavar='PCT', help='随机擦除概率')
    subparsers.add_argument('--remode', type=str, default='pixel', help='随机擦除模式')
    subparsers.add_argument('--recount', type=int, default=1, help='随机擦除次数')

    # 数据参数
    subparsers.add_argument('--data-path', default='/local_datasets/', type=str, help='数据集路径')
    subparsers.add_argument('--dataset', default='svhn-mnist', type=str, help='数据集名称')
    subparsers.add_argument('--shuffle', default=False, help='是否打乱数据顺序')
    subparsers.add_argument('--output_dir', default='output/', help='输出路径')
    subparsers.add_argument('--device', default='cuda', help='训练设备')
    subparsers.add_argument('--seed', default=42, type=int, help='随机种子')
    subparsers.add_argument('--eval', action='store_true', help='仅执行评估')
    subparsers.add_argument('--num_workers', default=2, type=int, help='数据加载线程数')
    subparsers.add_argument('--pin-mem', action='store_true',
                        help='固定 CPU 内存以提高传输效率')
    subparsers.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='不固定内存')
    subparsers.set_defaults(pin_mem=True)

    # 持续学习参数
    subparsers.add_argument('--train_mask', default=True, type=bool, help='训练时是否使用类别掩码')
    subparsers.add_argument('--task_inc', default=False, type=bool, help='是否进行任务增量学习')

    # Prompt Pool 参数
    subparsers.add_argument('--size', default=100, type=int, help='Prompt 池大小')
    subparsers.add_argument('--length', default=10, type=int, help='Prompt 长度')
    subparsers.add_argument('--top_k', default=1, type=int, help='选择的 Prompt 数量')
    subparsers.add_argument('--initializer', default='uniform', type=str, help='初始化方法')
    subparsers.add_argument('--prompt_key', default=True, type=bool, help='是否使用 Prompt 键')
    subparsers.add_argument('--prompt_key_init', default='uniform', type=str, help='Prompt 键初始化')
    subparsers.add_argument('--use_prompt_mask', default=False, type=bool, help='是否使用 Prompt 掩码')
    subparsers.add_argument('--shared_prompt_pool', default=False, type=bool, help='是否共享 Prompt 池')
    subparsers.add_argument('--shared_prompt_key', default=False, type=bool, help='是否共享 Prompt 键')
    subparsers.add_argument('--batchwise_prompt', default=True, type=bool, help='是否批次级 Prompt')
    subparsers.add_argument('--embedding_key', default='cls', type=str, help='嵌入键类型')
    subparsers.add_argument('--predefined_key', default='', type=str, help='预定义键')
    subparsers.add_argument('--pull_constraint', default=True, help='是否使用拉约束')
    subparsers.add_argument('--pull_constraint_coeff', default=0.1, type=float, help='拉约束系数')

    # ViT 参数
    subparsers.add_argument('--global_pool', default='token', choices=['token', 'avg'], type=str, 
                           help='全局池化类型')
    subparsers.add_argument('--head_type', default='prompt', choices=['token', 'gap', 'prompt', 'token+prompt'], 
                           type=str, help='分类头输入类型')
    subparsers.add_argument('--freeze', default=['blocks', 'patch_embed', 'cls_token', 'norm', 'pos_embed'], 
                           nargs='*', type=list, help='冻结的模型部分')

    # 杂项参数
    subparsers.add_argument('--print_freq', type=int, default=10, help='打印频率')

    # FedTA 参数
    subparsers.add_argument('--method', type=str, default='fedta', help='Prompt 方法')
    subparsers.add_argument('--e_prompt_layer_idx', default=[2, 3, 4], type=int, nargs="+",
                            help='E-Prompt 层索引')
    subparsers.add_argument('--client_num', default=5, type=int, help='客户端数量')
    subparsers.add_argument('--task_num', default=5, type=int, help='任务数量')
    subparsers.add_argument('--private_class_num', default=15, type=int, help='每个客户端私有类别数')
    subparsers.add_argument('--surrogate_num', default=20, type=int, help='代理数据数量')
    subparsers.add_argument('--global_epoch', default=5, type=int, help='全局训练轮数')
    subparsers.add_argument('--local_epoch', default=30, type=int, help='本地训练轮数')
    subparsers.add_argument('--threshold', default=0.25, type=float, help='BGPS 原型选择相似度阈值（对齐官方硬编码值）')
    subparsers.add_argument('--data_name', default='cifar100', type=str, help='数据集名称')
    subparsers.add_argument('--model_name', default='Tail_Anchor', choices=['AlexNet', 'VGG16', 'ResNet18', 'SimpleCNN', 'Tail_Anchor'],
                            type=str, help='模型名称')

    # ===== FedSMR-v2 超参数 =====
    # -- Residual Soft-Anchor --
    subparsers.add_argument('--use_soft_anchor', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='是否启用 Residual Soft-Anchor')
    subparsers.add_argument('--soft_temperature', default=0.17, type=float,
                            help='Soft Anchor softmax 温度')
    subparsers.add_argument('--soft_anchor_ratio', default=0.25, type=float,
                            help='残差系数 γ')
    subparsers.add_argument('--use_seen_routing', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='是否将 Anchor 路由限制在 seen classes（避免未来类污染）')

    # -- Anchor Routing Loss --
    subparsers.add_argument('--use_route_loss', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='是否启用监督路由损失 L_route')
    subparsers.add_argument('--route_temperature', default=0.1, type=float,
                            help='路由损失的温度系数')
    subparsers.add_argument('--lambda_route', default=0.05, type=float,
                            help='路由损失权重，建议 0.05~0.1')

    # -- Memory Structure Preservation (MSP) --
    subparsers.add_argument('--use_msp', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='是否启用 MSP 正则化')
    subparsers.add_argument('--msp_diversity_coeff', default=0.03, type=float,
                            help='Seen-Only Diversity 系数（v2: 建议0.03）')
    subparsers.add_argument('--diversity_margin', default=0.2, type=float,
                            help='Diversity margin: cos>margin才惩罚')
    subparsers.add_argument('--msp_coherence_coeff', default=0.0, type=float,
                            help='Coherence 系数（v2: 默认关闭）')
    subparsers.add_argument('--msp_temporal_coeff', default=0.1, type=float,
                            help='Key+Anchor Temporal Stability 系数')
    subparsers.add_argument('--key_temporal_ratio', default=0.5, type=float,
                            help='Key temporal 在总 temporal loss 中的权重 η')

    # -- Prototype Head Replay --
    subparsers.add_argument('--use_proto_replay', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='是否启用 Global Prototype Head Replay')
    subparsers.add_argument('--lambda_proto', default=0.2, type=float,
                            help='Proto replay 损失权重，CIFAR-100:0.20, ImageNet-R:0.30')

    # -- Class-Aware Head Aggregation / Head Grad Mask --
    # [已删除] use_class_aware_head_agg 和 use_head_grad_mask
    # 原始 FedTA 设计：vit.head 做全局 FedAvg，Tail Anchor model.head 做 per-task 快照

    # ===== GPC-DR: Global Prototype-Calibrated Differentiable Retrieval =====
    subparsers.add_argument('--use_proto_calibration', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='是否启用全局原型校准路由 (GPC-DR)')
    subparsers.add_argument('--proto_beta', default=0.5, type=float,
                            help='原型路由权重 β')
    subparsers.add_argument('--proto_temperature', default=0.10, type=float,
                            help='原型相似度温度 τ_P')
    subparsers.add_argument('--use_diff_retrieval', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='是否启用 Gradient-Decoupled Differentiable Retrieval')
    subparsers.add_argument('--use_task_isolated_diff', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='TIDR: only current-task keys receive CE gradient through differentiable retrieval')
    subparsers.add_argument('--adaptive_gamma', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='是否启用置信度门控的自适应 γ')
    subparsers.add_argument('--gamma_max', default=0.35, type=float,
                            help='自适应 γ 最大值')
    subparsers.add_argument('--use_gpa', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='是否启用 Phase 1 Global Prototype Alignment')
    subparsers.add_argument('--lambda_gpa', default=0.2, type=float,
                            help='GPA 损失权重')
    subparsers.add_argument('--lambda_pcr', default=0.05, type=float,
                            help='PCR 损失权重 (proto calibration 启用时替代 lambda_route)')

    # -- 已废弃/可选增强 --
    subparsers.add_argument('--use_soft_prompt', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='[已废弃] Soft Prompt Retrieval')
    subparsers.add_argument('--use_sparse_softmax', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='[已废弃] Top-K Sparse Softmax')
    subparsers.add_argument('--temperature_anneal', default=False, type=lambda x: (str(x).lower() == 'true'),
                            help='[已废弃] Temperature Annealing')
