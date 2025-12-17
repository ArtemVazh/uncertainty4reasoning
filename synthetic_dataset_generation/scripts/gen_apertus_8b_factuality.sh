#!/bin/bash -l

#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --account=a-infra01-1
#SBATCH --job-name=gen_apertus_8b_factuality
#SBATCH --output=gen_apertus_8b_factuality.out
#SBATCH --error=gen_apertus_8b_factuality.err


ROOT="/iopsstor/scratch/cscs/jni"

export HF_HOME="${ROOT}/hf_home"
export CUDA_VISIBLE_DEVICES=0,1,2,3
huggingface-cli login --token $HUGGINGFACE_TOKEN

MODEL_PATH="swiss-ai/Apertus-8B-Instruct-2509"
# DATASET_PATH="JingweiNi/train_prm800k_for_uhead"
DATASET_PATH="llm-uncertainty-head/train_akimbio_gemma2"
DATASET_SPLIT="train"
QUESTION_COL="question"
ANSWER_COL="answer"
PROMPT_FILE="${ROOT}/uncertainty4reasoning/configs/apertus_8b_prompt.txt"
SAVE_PATH="${ROOT}/uncertainty4reasoning/gen_data/train_akimbio_apertus_8b_factuality_texts"
HF_CACHE="${ROOT}/hf_home"
N_SAMPLES_PER_INPUT=1
TENSOR_PARALLEL_SIZE=4
MAX_NEW_TOKENS=30000
N_SAMPLES=50


DEBUG=0

if [ $DEBUG -eq 0 ]; then
    LAUNCHING="srun --container-writable --environment=qwen3_next"
else
    LAUNCHING=""
fi

ENV_COMMAND="pip install parse nltk sentence-transformers rouge_score  && pip uninstall -y numpy && pip install --no-cache-dir 'numpy==1.26.4'"

SCRIPT_COMMAND="PYTHONPATH=${ROOT}/uncertainty4reasoning:$PYTHONPATH python -m synthetic_dataset_generation.run_generate_texts_vllm \
    --dataset-path ${DATASET_PATH} \
    --dataset-split ${DATASET_SPLIT} \
    --question-col ${QUESTION_COL} \
    --answer-col ${ANSWER_COL} \
    --final-answers \
    --n-samples-per-input ${N_SAMPLES_PER_INPUT} \
    --prompt-file ${PROMPT_FILE} \
    --model-path ${MODEL_PATH} \
    --save-path ${SAVE_PATH} \
    --hf-cache ${HF_CACHE} \
    --vllm \
    --temperature 1.0 \
    --top-p 0.95 \
    --top-k 50 \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --n-samples ${N_SAMPLES}"

$LAUNCHING bash -lc "$ENV_COMMAND && $SCRIPT_COMMAND"

