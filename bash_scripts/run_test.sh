#!/bin/bash

# Activate the virtual environment
cd ..
source venv/bin/activate

PYTHONPATH=./ \
WANDB_PROJECT=ue-reasoning \
DEEPSEEK_API_KEY=$(<configs/deepseek_api_key.txt) \
HYDRA_CONFIG=configs/polygraph_eval_claim_reasoning.yaml \
    python eval_uhead.py \
    model.path=Qwen/Qwen3-8B \
    dataset=test_plan_Qwen3-8B_texts\
    load_from_disk=True \
    stat_calculators.2.cfg.uq_head_path=checkpoints/uhead_claim_Qwen3-8B_natural_plan_actual_2_trial \
    subsample_eval_dataset=10 \
    +hf_save_path=awsuineg/ue_manager_plan_Qwen3-8B