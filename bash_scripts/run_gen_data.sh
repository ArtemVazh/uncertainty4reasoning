#!/bin/bash

# Activate the virtual environment
cd ..
source venv/bin/activate

# Set multiprocessing start method to avoid CUDA fork issues
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONMULTIPROCESSING_START_METHOD=spawn

# Insert logging setup
LOG_DIR=logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_ds_$(date +%Y%m%d_%H%M%S).log"
echo "Logging output to $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# Number of GPUs to select (default = 2)
NUM_GPUS=${1:-1}

# Memory usage threshold in MiB (adjust if needed)
MEM_THRESHOLD=25000

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

# Use the detected GPU (first one from the available list)
DEVICE_ID="cuda:0"

python -m synthetic_dataset_generation.run_generate_texts \
    --dataset-path planning_datasets/meeting_train.jsonl \
    --n-samples 800 \
    --model-path Qwen/Qwen3-8B \
    --device $DEVICE_ID \
    --prompt-file configs/qwen3_prompt_jingwei_new.txt \
    --save-path train_data/train_plan_Qwen3-8B_texts_new_samples \
    --question-col prompt_0shot \
    --answer-col golden_plan \
    --n-samples-per-input 3 \
    --temperature 1.0 \
    --top-k 50 \
    --top-p 0.95 \
    --vllm
