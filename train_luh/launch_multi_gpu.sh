#!/bin/bash

# Multi-GPU training launch script
# Usage: bash launch_multi_gpu.sh [config_file]

# Set the number of GPUs you want to use
NGPUS=2  # Change this to the number of GPUs you have

# Set the config file
CONFIG_FILE=${1:-"config.yaml"}  # Default to config.yaml if not provided

# Set environment variables for NCCL (if needed)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=lo  # or your network interface
export CUDA_VISIBLE_DEVICES=0,1  # Specify which GPUs to use

# Launch with torchrun (recommended for PyTorch >= 1.10)
torchrun \
    --nproc_per_node=$NGPUS \
    --master_port=29500 \
    run_train_luh.py \
    --config-path="." \
    --config-name="$CONFIG_FILE"

# Alternative: Launch with python -m torch.distributed.launch (older method)
# python -m torch.distributed.launch \
#     --nproc_per_node=$NGPUS \
#     --master_port=29500 \
#     run_train_luh.py \
#     --config-path="." \
#     --config-name="$CONFIG_FILE" 