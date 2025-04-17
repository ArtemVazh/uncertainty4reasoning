Before usage: paste your deekseek API key in `configs/deepseek_api_key.txt`.

# Generate dataset for training and testing UHead

Already generated datasets:
- `rediska0123/train_gsm8k_Qwen2.5-Math-7B`
- `rediska0123/train_gsm8k_Qwen2.5-Math-1.5B`
- `rediska0123/test_gsm8k`

Example commands to generate new datasets:

#### Train dataset creation

1. Generate annotation dataset (on GPU). Runs for ~30 mins.
```bash
python -m synthetic_dataset_generation.run_generate_texts \
  --dataset-path openai/gsm8k,main --dataset-split test --n-samples 819 \ 
  --model-path Qwen/Qwen2.5-Math-7B \
  --device auto \
  --prompt-file configs/gsm8k_3shot_prompt.txt \
  --save-path /cluster/project/sachan/ekaterina/.cache/train_gsm8k_Qwen2.5-Math-7B-texts
# 819 samples: all except last 500 for test
```

2. Verify claims with DeepSeek (no GPU required). DeepSeek answers are cached. Can run for 30min-1h with enough `n-threads`.
```bash
python -m synthetic_dataset_generation.run_extract_verify_claims \
    --dataset-path /cluster/project/sachan/ekaterina/.cache/train_gsm8k_Qwen2.5-Math-7B-texts \
    --model-path Qwen/Qwen2.5-Math-7B \
    --prompt-file configs/gsm8k_3shot_prompt.txt \
    --save-path /cluster/project/sachan/ekaterina/.cache/train_gsm8k_Qwen2.5-Math-7B \
    --hf-save-path rediska0123/train_gsm8k_Qwen2.5-Math-7B \
    --api-key-file configs/deepseek_api_key.txt \
    --n-threads 16
```

Resulting train dataset is saved at `rediska0123/train_gsm8k_Qwen2.5-Math-7B`.

#### Test dataset
Apply prompt to create test dataset (model-agnostic, just prompts). No GPU required. Runs super fast.
```bash
python -m synthetic_dataset_generation.run_create_test_dataset \
    --dataset-path openai/gsm8k,main --dataset-split test --start-index 819 \
    --save-path /cluster/project/sachan/ekaterina/.cache/test_gsm8k \
    --prompt-file configs/gsm8k_3shot_prompt.txt \
    --hf-save-path rediska0123/test_gsm8k
```
Resulting test dataset is saved at `rediska0123/test_gsm8k`.

# Train UHead

Already trained models:
- `rediska0123/uhead_Qwen2.5-Math-7B`
- `rediska0123/uhead_Qwen2.5-Math-1.5B`

Example commands to train new model: TODO

# Test UHead
Already saved test results:
- `rediska0123/ue_manager_gsm8k_Qwen2.5-Math-1.5B`

Example to test your UHead along with other UE baselines (MaxProb, Perplexity, Entropy, CCP):
```bash
PYTHONPATH=./ \
WANDB_PROJECT=YOUR_WANDB_PROJECT \
DEEPSEEK_API_KEY=$(<configs/deepseek_api_key.txt) \
HYDRA_CONFIG=configs/polygraph_eval_claim_reasoning.yaml \
    python eval_uhead.py \
    model.path=Qwen/Qwen2.5-Math-7B \
    dataset=rediska0123/test_gsm8k_Qwen2.5-Math-7B \
    stat_calculators.2.cfg.uq_head_path=rediska0123/uhead_Qwen2.5-Math-7B \
    +hf_save_path=rediska0123/ue_manager_gsm8k_Qwen2.5-Math-1.5B
```

# Test baselines

#### Qwen/Qwen2.5-Math-7B-PRM800K
Runs super fast, saves reward values to `save-path`.
```bash
python eval_qwen_prm.py \
    --hf-manager-path rediska0123/ue_manager_gsm8k_Qwen2.5-Math-1.5B \
    --base-model-path Qwen/Qwen2.5-Math-7B \
    --save-path /cluster/project/sachan/ekaterina/.cache/scores_prm_gsm8k_qwen1.5B.json \
    --prm-model-path Qwen/Qwen2.5-Math-7B-PRM800K \
    --prompt-file configs/gsm8k_3shot_prompt.txt \
    --device auto
```

#### ReasonEval
Runs super fast, saves reward values to `save-path`.
```bash
python eval_reasoneval.py \
    --hf-manager-path rediska0123/ue_manager_gsm8k_Qwen2.5-Math-1.5B \
    --base-model-path Qwen/Qwen2.5-Math-7B \
    --save-path /cluster/project/sachan/ekaterina/.cache/scores_reasoneval_gsm8k_qwen1.5B.json \
    --reasoneval-model-path GAIR/ReasonEval-7B \
    --prompt-file configs/gsm8k_3shot_prompt.txt \
    --device auto
```

# Plot results

Use `plot_results.ipynb` to get results table.