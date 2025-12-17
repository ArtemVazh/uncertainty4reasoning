#!/bin/bash -l

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=a-infra01-1
#SBATCH --job-name=gpt_oss_annotate_qwen3_1.7b_thinking
#SBATCH --output=gpt_oss_annotate_qwen3_1.7b_thinking.out
#SBATCH --error=gpt_oss_annotate_qwen3_1.7b_thinking.err


ROOT="/iopsstor/scratch/cscs/jni/uncertainty4reasoning"
MODEL_PATH="Qwen/Qwen3-1.7B"
CLAIMS_PREFIX="JingweiNi/train_prm800k_qwen3_1.7b_thinking_5000_extracted"
ANNOTATOR_MODEL="openai/gpt-oss-120b"
ANNOTATOR_NAME="gpt-oss-120b"
VLLM_LOG="vllm_gpt_oss_120b.log"
SHARDS="4 5 6 7"
SHARDS_TAG="4_7"
DEBUG=0
TIMEOUT=600

export HF_HOME="/iopsstor/scratch/cscs/jni/hf_home"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TIKTOKEN_ENCODINGS_BASE="/iopsstor/scratch/cscs/jni/tiktoken_encodings"
huggingface-cli login --token $HUGGINGFACE_TOKEN

VLLM_CMD="vllm serve $ANNOTATOR_MODEL --tensor-parallel-size 4"
ANNOTATION_CMD="PYTHONPATH=${ROOT} python -m synthetic_dataset_generation.run_verify_claims_thinking \
  --prompt-file $ROOT/configs/qwen3_prompt_thinking.txt \
  --save-path $ROOT/gen_data/train_prm800k_${ANNOTATOR_NAME}_annotated_qwen3_1.7b_thinking_5000_shards_${SHARDS_TAG} \
  --hf-save-path JingweiNi/train_prm800k_${ANNOTATOR_NAME}_annotated_qwen3_1.7b_thinking_5000_shards_${SHARDS_TAG} \
  --extracted-claims-path-prefix $CLAIMS_PREFIX \
  --shards-to-load $SHARDS \
  --anno-model $ANNOTATOR_MODEL \
  --api-cache $ROOT/gen_data/${ANNOTATOR_NAME}_annotate_qwen3_1.7b_thinking_5000 \
  --n-threads 64 \
  --fact-check-base-url http://localhost:8000/v1 \
  --debug"

ENV_CMD="pip install parse nltk sentence-transformers rouge_score  && pip uninstall -y numpy && pip install --no-cache-dir 'numpy==1.26.4'"


echo "vLLM command: $VLLM_CMD"
echo "Annotation command: $ANNOTATION_CMD"

if [ $DEBUG -eq 0 ]; then
    LAUNCHING="srun --container-writable --environment=qwen3_next"
else
    LAUNCHING=""
fi

$LAUNCHING bash -lc "
  set -euo pipefail

  eval \"$ENV_CMD\"

  # 1) start vLLM in background
  $VLLM_CMD > $ROOT/$VLLM_LOG 2>&1 &
  VLLM_PID=\$!
  echo \"[INFO] vLLM PID=\$VLLM_PID\"

  # Ensure cleanup on any exit
  trap 'echo \"[CLEANUP] Stopping vLLM (\$VLLM_PID)\"; kill -TERM \$VLLM_PID 2>/dev/null || true; wait \$VLLM_PID 2>/dev/null || true' EXIT

  # 2) wait for readiness (timeout ${TIMEOUT}s)
  echo \"[WAIT] Watching $VLLM_LOG for readiness...\"
  if ! timeout $TIMEOUT bash -c '( tail -n0 -f $ROOT/$VLLM_LOG & ) | grep -q -- \"Application startup complete.\"'; then
    echo \"[ERROR] vLLM did not become ready within ${TIMEOUT}s\"
    exit 1
  fi
  echo \"[READY] vLLM is ready.\"

  # 3) run env setup + annotation
  echo \"[RUN] Launching annotation...\" 
  
  eval \"$ANNOTATION_CMD\"
  
  kill -TERM \$VLLM_PID 2>/dev/null || true
  wait \$VLLM_PID 2>/dev/null || true
  trap - EXIT
  exit 0
"
