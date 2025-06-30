# Overview

Main steps to run whole benchmark on new dataset/model (there are examples for each step in later sections):

1. Generate training dataset
    - Generate texts: `synthetic_dataset_generation/run_generate_texts.py`
    - Split texts to steps and annotate with DeepSeek: `synthetic_dataset_generation/run_extract_verify_claims.py`
2. Generate test dataset: `synthetic_dataset_generation/run_create_test_dataset.py`
3. Train UHead: `train_luh/run_train_luh.py`
4. Test UHead: `eval_uhead.py`
5. Evaluate baselines
    - PRM: `eval_prm.py`
    - ReasonEval: `eval_reasoneval.py`
6. Plot resulting tables: `plot_results.ipynb`

Before usage: paste your deekseek API key in `configs/deepseek_api_key.txt`.

# Example usage

For the following commands, change `/cluster/...` paths to your local paths to save models/datasets in. Also change `rediska0123/...` to your huggingface path.

## 1. Generate training dataset

Example commands to generate training dataset:

1. Generate annotation dataset (on GPU). Runs for ~30 mins.
```bash
python -m synthetic_dataset_generation.run_generate_texts \
  --dataset-path openai/gsm8k,main --n-samples 819 \
  --model-path Qwen/Qwen3-1.7B \
  --device cuda:0 \
  --prompt-file configs/qwen3_prompt.txt \
  --save-path /cluster/project/sachan/ekaterina/.cache/train_gsm8k_Qwen3-1.7B_texts
# 819 samples: all except last 500 for test
```

2. Verify claims with DeepSeek (no GPU required). DeepSeek answers are cached. Can run for 30min-1h with enough `n-threads`.
```bash
python -m synthetic_dataset_generation.run_extract_verify_claims \
  --dataset-path /cluster/project/sachan/ekaterina/.cache/train_gsm8k_Qwen3-1.7B_texts \
  --model-path Qwen/Qwen3-1.7B \
  --prompt-file configs/qwen3_prompt.txt \
  --save-path /cluster/project/sachan/ekaterina/.cache/train_gsm8k_Qwen3-1.7B \
  --hf-save-path rediska0123/train_gsm8k_Qwen3-1.7B \
  --api-key-file configs/deepseek_api_key.txt \
  --n-threads 16
```

## 2. Generate test dataset

Apply prompt to create test dataset. No GPU required. Runs super fast.
```bash
python -m synthetic_dataset_generation.run_create_test_dataset \
  --dataset-path openai/gsm8k,main --dataset-split test --start-index 819 \
  --save-path /cluster/project/sachan/ekaterina/.cache/test_gsm8k_Qwen3-1.7B \
  --hf-save-path rediska0123/test_gsm8k_Qwen3-1.7B \
  --hf-cache /cluster/project/sachan/ekaterina/.cache \
  --prompt-file configs/gsm8k_3shot_prompt.txt
# start from 819'th text until the end of the dataset
```

## 3. Train UHead

Train with the following script. Runs several hours. Recommended to experiment with different number of epochs.

Before running, create your wandb project and replace `WANDB_PROJECT` variable, or alternatively switch off wandb logging by setting `report_to: none` in YAML config.

```bash
PYTHONPATH=./ WANDB_PROJECT=ue-reasoning \
HYDRA_CONFIG=../configs/train_uhead_claim.yaml \
python train_luh/run_train_luh.py \
  model.pretrained_model_name_or_path=Qwen/Qwen3-1.7B \
  dataset.path=hf:rediska0123/train_gsm8k_Qwen3-1.7B \
  dataset.prompt_path=configs/qwen3_prompt.txt \
  training_arguments.num_train_epochs=30 \
  +save_dir=/cluster/project/sachan/ekaterina/.cache/uhead_Qwen3-1.7B_gsm8k \
  +hf_save_path=rediska0123/uhead_Qwen3-1.7B_gsm8k
```

## 4. Test UHead

Example to test your UHead along with other UE baselines (MaxProb, Perplexity, Entropy, CCP). Replace `WANDB_PROJECT` with your wandb project.
```bash
PYTHONPATH=./ \
WANDB_PROJECT=ue-reasoning \
DEEPSEEK_API_KEY=$(<configs/deepseek_api_key.txt) \
HYDRA_CONFIG=configs/polygraph_eval_claim_reasoning.yaml \
    python eval_uhead.py \
    model.path=Qwen/Qwen3-1.7B \
    dataset=rediska0123/test_gsm8k_Qwen3-1.7B \
    stat_calculators.2.cfg.uq_head_path=rediska0123/uhead_Qwen3-1.7B_gsm8k \
    +hf_save_path=rediska0123/ue_manager_gsm8k_Qwen3-1.7B
```

## 5. Evaluate baselines

### Process-Reward Model baseline
Runs super fast, updates manager in `hf-manager-path` with PRM reward values.
```bash
python eval_prm.py \
    --hf-manager-path rediska0123/ue_manager_gsm8k_Qwen3-1.7B \
    --base-model-path Qwen/Qwen3-1.7B \
    --prm-model-path Qwen/Qwen2.5-Math-7B-PRM800K \
    --prompt-file configs/qwen3_prompt.txt \
    --device auto
```

### ReasonEval baseline
Runs super fast, updates manager in `hf-manager-path` with PRM reward values.
```bash
python eval_reasoneval.py \
    --hf-manager-path rediska0123/ue_manager_gsm8k_Qwen3-1.7B \
    --base-model-path Qwen/Qwen3-1.7B \
    --reasoneval-model-path GAIR/ReasonEval-7B \
    --prompt-file configs/qwen3_prompt.txt \
    --device auto
```

## 6. Plot results

Use `plot_results.ipynb` to get results table.