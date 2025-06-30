#!/bin/bash

# Two-part evaluation script adapted from run_test.sh
# This shows how to run your original command in two parts for better performance

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

echo "============================================================================="
echo "TWO-PART EVALUATION: Based on your original run_test.sh"
echo "============================================================================="

# Part 1: Run everything except StepFactCheck (~1 hour)
echo "PART 1: Running evaluation without StepFactCheck (faster)"
echo "============================================================================="

PYTHONPATH=./ \
WANDB_PROJECT=ue-reasoning \
DEEPSEEK_API_KEY=$(<configs/deepseek_api_key.txt) \
    python run_part1.py \
    model.path=Qwen/Qwen3-8B \
    dataset=test_plan_Qwen3-8B_texts \
    load_from_disk=True \
    stat_calculators.2.cfg.uq_head_path=checkpoints/uhead_claim_Qwen3-8B_natural_plan_actual_combined/model \
    subsample_eval_dataset=3 \
    +hf_save_path=awsuineg/ue_manager_plan_Qwen3-8B_part1

# Check if Part 1 succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================================="
    echo "PART 1 COMPLETED! Now find the output directory..."
    echo "============================================================================="
    
    # Find the most recent output directory
    # Adjust this path based on where your results are saved
    CACHE_DIR="./workdir/cache"
    LATEST_DIR=$(find "$CACHE_DIR" -name "ue_manager.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2- | xargs dirname)
    
    if [ -n "$LATEST_DIR" ] && [ -f "$LATEST_DIR/ue_manager.pth" ]; then
        echo "Found Part 1 results at: $LATEST_DIR"
        
        # Part 2: Add StepFactCheck annotation with high parallelization
        echo ""
        echo "============================================================================="
        echo "PART 2: Adding StepFactCheck annotation with high parallelization"
        echo "============================================================================="
        
        DEEPSEEK_API_KEY=$(<configs/deepseek_api_key.txt) \
            python add_stepfactcheck.py \
            --load_path "$LATEST_DIR" \
            --save_path "${LATEST_DIR}_with_stepfactcheck" \
            --n_threads 32 \
            --progress_bar
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "============================================================================="
            echo "TWO-PART EVALUATION COMPLETED SUCCESSFULLY!"
            echo "Final results with StepFactCheck: ${LATEST_DIR}_with_stepfactcheck"
            echo "============================================================================="
        else
            echo "Part 2 failed! But Part 1 results are still available at: $LATEST_DIR"
            exit 1
        fi
    else
        echo "Could not find Part 1 results automatically."
        echo "Please check the output above for the save path and run Part 2 manually:"
        echo "  python add_stepfactcheck.py --load_path /path/to/part1/results --save_path /path/to/final/results --n_threads 32"
        exit 1
    fi
else
    echo "Part 1 failed! Check the error messages above."
    exit 1
fi 