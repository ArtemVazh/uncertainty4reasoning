#!/bin/bash -l

set -euo pipefail

ROOT="/iopsstor/scratch/cscs/jni/uncertainty4reasoning"
MODEL_PATH="swiss-ai/Apertus-8B-Instruct-2509"
DATASET_PATH="JingweiNi/train_akimbio_apertus_8b_factuality_texts_cleaned"
ANNOTATOR_MODEL="gpt-5.1-2025-11-13"
ANNOTATOR_NAME="gpt-5.1"
SAMPLE=3

export HF_HOME="/iopsstor/scratch/cscs/jni/hf_home"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TIKTOKEN_ENCODINGS_BASE="/iopsstor/scratch/cscs/jni/tiktoken_encodings"
huggingface-cli login --token $HUGGINGFACE_TOKEN

ANNOTATION_CMD="PYTHONPATH=${ROOT} python -m synthetic_dataset_generation.run_extract_verify_claims_factuality \
  --dataset-path $DATASET_PATH \
  --model-path $MODEL_PATH \
  --anno-model $ANNOTATOR_MODEL \
  --prompt-file $ROOT/configs/apertus_8b_prompt.txt \
  --fact-check-prompt $ROOT/synthetic_dataset_generation/utils/sent_fact_check_prompt.txt \
  --save-path $ROOT/gen_data/train_akimbio_apertus_8b_factuality_${ANNOTATOR_NAME}_annotated_${SAMPLE} \
  --hf-save-path JingweiNi/train_akimbio_apertus_8b_factuality_${ANNOTATOR_NAME}_annotated_${SAMPLE} \
  --api-cache $ROOT/gen_data/akimbio_apertus_8b_${ANNOTATOR_NAME}_annotated_factuality \
  --n-threads 16 \
  --sample $SAMPLE \
  --debug \
  --fact-check-base-url None \
  --fact-check-api-key ${OPENAI_API_KEY}"


eval "$ANNOTATION_CMD"


