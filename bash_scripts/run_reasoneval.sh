#!/bin/bash

# Activate the virtual environment
cd ..
source venv/bin/activate

# Insert logging setup
LOG_DIR=logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_ds_$(date +%Y%m%d_%H%M%S).log"
echo "Logging output to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# Number of GPUs to select (default = 2)
NUM_GPUS=${1:-1}

# Memory usage threshold in MiB (adjust if needed)
MEM_THRESHOLD=17000

# Function to check GPU availability
check_gpus() {
    available_gpus=$(nvidia-smi --query-gpu=index,memory.used \
        --format=csv,noheader,nounits \
        | awk -F', ' -v thresh="$MEM_THRESHOLD" '$2 < thresh' \
        | sort -t',' -k2 -n \
        | awk -F',' '{print $1}' \
        | head -n "$NUM_GPUS")
    
    gpu_array=($available_gpus)
}


python eval_reasoneval.py \
    --hf-manager-path awsuineg/ue_manager_plan_Qwen3-8B  \
    --base-model-path Qwen/Qwen3-8B \
    --reasoneval-model-path GAIR/ReasonEval-7B \
    --prompt-file configs/qwen3_prompt.txt \
    --device auto