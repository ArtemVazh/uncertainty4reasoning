#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7
#SBATCH --mem-per-cpu=16G
#SBATCH --job-name=extract_claim_qwen3_1.7b_thinking_%a
#SBATCH --output=extract_claim_qwen3_1.7b_thinking_%a.out
#SBATCH --error=extract_claim_qwen3_1.7b_thinking_%a.err

module load eth_proxy

ROOT="/cluster/project/sachan/njingwei/uncertainty4reasoning"
MODEL_PATH="Qwen/Qwen3-1.7B"
DATASET_PATH="JingweiNi/train_prm800k_qwen3_1.7b_thinking_texts"
SAMPLE_SIZE=5000
PARTITION_NUM=8
PARTITION_IDX=${SLURM_ARRAY_TASK_ID}

export HF_HOME="/cluster/work/lawecon/Work/jingwei/transformer_models"
export HUGGINGFACE_TOKEN="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"


ANNOTATION_CMD="PYTHONPATH=${ROOT} python -m synthetic_dataset_generation.run_extract_claims_thinking \
  --dataset-path $DATASET_PATH \
  --model-path $MODEL_PATH \
  --prompt-file $ROOT/configs/qwen3_prompt_thinking.txt \
  --save-path $ROOT/gen_data/train_prm800k_qwen3_1.7b_thinking_${SAMPLE_SIZE}_extracted_${PARTITION_IDX} \
  --hf-save-path JingweiNi/train_prm800k_qwen3_1.7b_thinking_${SAMPLE_SIZE}_extracted_${PARTITION_IDX} \
  --start-idx 0 \
  --subset $SAMPLE_SIZE \
  --partition-num $PARTITION_NUM \
  --partition-idx $PARTITION_IDX"

eval "$ANNOTATION_CMD"

