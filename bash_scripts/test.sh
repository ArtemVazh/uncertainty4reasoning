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
DEEPSEEK_API_KEY=$(<configs/deepseek_api_key.txt) python add_stepfactcheck.py \
    --load_path ./workdir/cache/bio/Qwen/Qwen3-8B/test_plan_Qwen3-8B_texts/2025-06-25/20-18-44 \
    --save_path ./workdir/cache/bio/Qwen/Qwen3-8B/test_plan_Qwen3-8B_texts/2025-06-25/20-18-44_with_stepfactcheck \
    --n_threads 32 \
    --progress_bar