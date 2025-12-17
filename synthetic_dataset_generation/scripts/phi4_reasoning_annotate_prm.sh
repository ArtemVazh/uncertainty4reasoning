#!/bin/bash -l

#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=a-infra01-1
#SBATCH --job-name=phi4_annotate_prm
#SBATCH --output=phi4_annotate_prm.out
#SBATCH --error=phi4_annotate_prm.err


ROOT="/iopsstor/scratch/cscs/jni/uncertainty4reasoning"
MODEL_PATH="microsoft/phi-4"
DATASET_PATH="JingweiNi/train_prm800k_phi4_texts_finished"
ANNOTATOR_MODEL="microsoft/Phi-4-reasoning"
ANNOTATOR_NAME="Phi-4-reasoning"
VLLM_LOG="phi4_reasoning_vllm.log"

DEBUG=1


VLLM_CMD="vllm serve $ANNOTATOR_MODEL --enable-reasoning --reasoning-parser deepseek_r1 --pipeline-parallel-size 2 --tensor-parallel-size 2"
ANNOTATION_CMD="PYTHONPATH=${ROOT} python -m synthetic_dataset_generation.run_extract_verify_claims_self \
  --dataset-path $DATASET_PATH \
  --model-path $MODEL_PATH \
  --anno-model $ANNOTATOR_MODEL \
  --prompt-file $ROOT/configs/phi4_prompt_general.txt \
  --save-path $ROOT/gen_data/train_prm800k_${ANNOTATOR_NAME}_finished_self_annotate \
  --hf-save-path JingweiNi/train_prm800k_${ANNOTATOR_NAME}_finished_self_annotate \
  --api-cache $ROOT/gen_data/${ANNOTATOR_NAME}_self_annotate \
  --n-threads 128"

ENV_CMD="pip install parse nltk sentence-transformers rouge_score"

if [ $DEBUG -eq 0 ]; then
    LAUNCHING="srun --container-writable --environment=apertus_alignment"
else
    LAUNCHING=""
fi

$LAUNCHING bash -lc "
  set -euo pipefail

  # 1) start vLLM in background
  $VLLM_CMD > $ROOT/$VLLM_LOG 2>&1 &
  VLLM_PID=\$!
  echo \"[INFO] vLLM PID=\$VLLM_PID\"

  # Ensure cleanup on any exit
  trap 'echo \"[CLEANUP] Stopping vLLM (\$VLLM_PID)\"; kill -TERM \$VLLM_PID 2>/dev/null || true; wait \$VLLM_PID 2>/dev/null || true' EXIT

  # 2) wait for readiness (timeout 600s)
  echo \"[WAIT] Watching $VLLM_LOG for readiness...\"
  if ! timeout 600 bash -c '( tail -n0 -f $ROOT/$VLLM_LOG & ) | grep -q -- \"Application startup complete.\"'; then
    echo \"[ERROR] vLLM did not become ready within 600s\"
    exit 1
  fi
  echo \"[READY] vLLM is ready.\"

  # 3) run env setup + annotation
  echo \"[RUN] Launching annotation...\"
  if $ENV_CMD && $ANNOTATION_CMD; then
    echo \"[DONE] Annotation finished successfully. Shutting down vLLM...\"
    kill -TERM \$VLLM_PID 2>/dev/null || true
    wait \$VLLM_PID 2>/dev/null || true
    trap - EXIT
    exit 0
  else
    echo \"[ERROR] Annotation failed. Shutting down vLLM...\"
    kill -TERM \$VLLM_PID 2>/dev/null || true
    wait \$VLLM_PID 2>/dev/null || true
    trap - EXIT
    exit 2
  fi
"

