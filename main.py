"""
FedTA (Federated Task Adaptation) Framework - Main Entry Point

This module serves as the main entry point for the FedTA federated learning framework.
It handles:
- Command line argument parsing
- Dataset preparation and splitting
- Model initialization (base model and prompt-enhanced model)
- Server initialization and training orchestration

Key Features:
- Multi-dataset support (CIFAR100, 5datasets, ImageNet-R, SVHN-MNIST, Office-Home)
- Vision Transformer with prompt learning (L2P)
- Federated learning with task adaptation
- Configurable training parameters via command line
"""

import argparse
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader

from Models.Server_DF import Server_DF
from config.cifar100_delay import get_args_parser

from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
import Models.vision_transformer__l2p
from data.iCIFAR100c import iCIFAR100c
from data.imagenet_r_subset_spliter import ImagenetR_spliter

torch.set_printoptions(threshold=float('inf'))

from data.cifar100_subset_spliter import cifar100_Data_Spliter
import warnings
warnings.filterwarnings("ignore")


def main(args):
    """
    Main training function for FedTA framework.
    
    Args:
        args: Parsed command line arguments containing all training configurations
    """
    # Set up training device
    device = torch.device(args.device)

    # Fix random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Configure pretrained model path
    pretrained_cfg = create_model(args.model).default_cfg
    if isinstance(pretrained_cfg, dict):
        pretrained_cfg['file'] = 'pretrain_model/ViT-B_16.npz'
    else:
        pretrained_cfg.file = 'pretrain_model/ViT-B_16.npz'

    cudnn.benchmark = True

    # Prepare dataset based on configuration
    print(args.data_name)
    if args.data_name == 'cifar100':
        client_data, client_mask = cifar100_Data_Spliter(
            client_num=args.client_num,
            task_num=args.task_num,
            private_class_num=args.private_class_num,
            input_size=args.input_size
        ).random_split()
        surro_data, test_data = cifar100_Data_Spliter(
            client_num=args.client_num,
            task_num=args.task_num,
            private_class_num=args.private_class_num,
            input_size=args.input_size
        ).process_testdata(args.surrogate_num)
        surro_data = iCIFAR100c(subset=surro_data)
        args.nb_classes = 100

    elif args.data_name == 'ImageNet-R':
        data_spliter = ImagenetR_spliter(
            client_num=args.client_num,
            task_num=args.task_num,
            private_class_num=args.private_class_num,
            input_size=args.input_size
        )

        client_data, client_mask = data_spliter.random_split()
        args.nb_classes = 200

        surro_data, test_data = ImagenetR_spliter(
            client_num=args.client_num,
            task_num=args.task_num,
            private_class_num=args.private_class_num,
            input_size=args.input_size
        ).process_testdata(args.surrogate_num)
        surro_data = iCIFAR100c(subset=surro_data)

    else:
        raise ValueError(f"Unsupported dataset: {args.data_name}. "
                         f"Supported datasets: cifar100, ImageNet-R")

    # Save experiment configuration for reproducibility
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Initialize original (base) model without prompts
    print(f"Creating original model: {args.model}")
    original_model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    # Initialize prompt-enhanced model
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        prompt_length=args.length,
        embedding_key=args.embedding_key,
        prompt_init=args.prompt_key_init,
        prompt_pool=True,
        prompt_key=args.prompt_key,
        pool_size=args.size,
        top_k=args.top_k,
        batchwise_prompt=args.batchwise_prompt,
        prompt_key_init=args.prompt_key_init,
        head_type=args.head_type,
    )

    # Load pretrained weights
    pretrained_path = 'pretrain_model/ViT-B_16.npz'
    print(f"Loading pretrained weights from {pretrained_path}")
    original_model.load_pretrained(pretrained_path)
    model.load_pretrained(pretrained_path)

    # Move models to device
    original_model.to(device)
    model.to(device)

    # Freeze specified model parameters if configured
    if args.freeze:
        for n, p in model.named_parameters():
            if n.startswith(tuple(args.freeze)):
                p.requires_grad = False

    # Print configuration and trainable parameters
    print(args)
    for n, p in model.named_parameters():
        if p.requires_grad == True:
            print(n)

    # Initialize server and start training
    if args.data_name in ['cifar100', 'ImageNet-R']:
        myServer = Server_DF(
            id='Server',
            origin_model=original_model,
            model_name=args.model_name,
            client_num=args.client_num,
            task_num=args.task_num,
            subset=client_data,
            class_mask=client_mask,
            lr=args.lr,
            global_epoch=args.global_epoch,
            local_epoch=args.local_epoch,
            batch_size=args.batch_size,
            device=device,
            method=args.method,
            threshold=args.threshold,
            surrogate_data=surro_data,
            test_data=None,
            args=args,
            model=model
        )
    else:
        myServer = Server_DF(
            id='Server',
            origin_model=original_model,
            model=model,
            client_num=args.client_num,
            task_num=args.task_num,
            subset=data_spliter,
            class_mask=None,
            lr=args.lr,
            global_epoch=args.global_epoch,
            local_epoch=args.local_epoch,
            batch_size=args.batch_size,
            device=device,
            method=args.method,
            threshold=args.threshold,
            surrogate_data=None,
            test_data=None,
            args=args
        )

    myServer.start()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser('FedTA training and evaluation configs')

    # Add subparsers for different dataset configurations
    subparser = parser.add_subparsers(dest='subparser_name')

    # CIFAR100 delay configuration
    from config.cifar100_delay import get_args_parser as cifar100_parser
    cifar100_config_parser = subparser.add_parser(
        'cifar100_delay',
        help='Split-CIFAR100 Delay configs'
    )
    cifar100_parser(cifar100_config_parser)

    # ImageNet-R delay configuration
    from config.imagenet_r_delay import get_args_parser as imagenet_r_parser
    imagenet_r_config_parser = subparser.add_parser(
        'imagenet_r_delay',
        help='ImageNet-R delay configs'
    )
    imagenet_r_parser(imagenet_r_config_parser)

    args = parser.parse_args()

    # Create output directory if specified
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Start main training loop
    main(args)