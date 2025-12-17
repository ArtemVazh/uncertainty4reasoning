#!/bin/bash -l

set -euo pipefail

ROOT="/iopsstor/scratch/cscs/jni/uncertainty4reasoning"
MODEL_PATH="Qwen/Qwen3-8B"
DATASET_PATH="JingweiNi/train_prm800k_qwen3_8b_thinking_texts"
ANNOTATOR_MODEL="gpt-5-mini-2025-08-07"
ANNOTATOR_NAME="gpt-5-mini"
SAMPLE=2

export HF_HOME="/iopsstor/scratch/cscs/jni/hf_home"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TIKTOKEN_ENCODINGS_BASE="/iopsstor/scratch/cscs/jni/tiktoken_encodings"
huggingface-cli login --token $HUGGINGFACE_TOKEN

ANNOTATION_CMD="PYTHONPATH=${ROOT} python -m synthetic_dataset_generation.run_extract_verify_claims_thinking \
  --dataset-path $DATASET_PATH \
  --model-path $MODEL_PATH \
  --anno-model $ANNOTATOR_MODEL \
  --prompt-file $ROOT/configs/qwen3_prompt_thinking.txt \
  --save-path $ROOT/gen_data/train_prm800k_${ANNOTATOR_NAME}_finished_annotate_qwen3_8b_thinking_${SAMPLE} \
  --hf-save-path JingweiNi/train_prm800k_${ANNOTATOR_NAME}_finished_annotate_qwen3_8b_thinking_${SAMPLE} \
  --api-cache $ROOT/gen_data/${ANNOTATOR_NAME}_annotate_qwen3_8b_thinking \
  --n-threads 16 \
  --sample $SAMPLE \
  --unique-questions \
  --debug \
  --fact-check-api-key ${OPENAI_API_KEY}"

eval "$ANNOTATION_CMD"

