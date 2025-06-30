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

# Loop until the desired number of GPUs are available
while true; do
    check_gpus
    if [ ${#gpu_array[@]} -eq "$NUM_GPUS" ]; then
        export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${gpu_array[*]}")
        echo "Using GPUs: $CUDA_VISIBLE_DEVICES"
        break
    else
        echo "Waiting for $NUM_GPUS GPUs with memory usage below ${MEM_THRESHOLD} MiB..."
        echo "Current eligible GPUs: ${gpu_array[@]}"
        sleep 30
    fi
done
PYTHONPATH=./ \
WANDB_PROJECT=ue-reasoning \
DEEPSEEK_API_KEY=$(<configs/deepseek_api_key.txt) \
HYDRA_CONFIG=configs/polygraph_eval_claim_reasoning.yaml \
    python eval_uhead.py \
    model.path=Qwen/Qwen3-8B \
    dataset=test_plan_Qwen3-8B_texts\
    load_from_disk=True \
    stat_calculators.2.cfg.uq_head_path=checkpoints/uhead_claim_Qwen3-8B_natural_plan_actual_combined/model \
    subsample_eval_dataset=720 \
    +hf_save_path=awsuineg/ue_manager_plan_Qwen3-8B