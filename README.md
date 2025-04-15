Generate annotation dataset:
```bash
python -m synthetic_dataset_generation.run_extract_verify_claims \
  --model-path Qwen/Qwen2.5-Math-7B \
  --hf-cache /cluster/project/sachan/ekaterina/.cache \
  --prompt-file configs/gsm8k_3shot_prompt.txt \
  --save-path /cluster/project/sachan/ekaterina/.cache/train_gsm8k_Qwen2.5-Math-7B \
  --num-samples 819  # all except last 500 for test
```

Verify claims with DeepSeek:
```bash
python -m synthetic_dataset_generation.run_extract_verify_claims \
    --dataset-path /cluster/project/sachan/ekaterina/.cache/train_gsm8k_Qwen2.5-Math-7B \
    --model-path Qwen/Qwen2.5-Math-7B \
    --hf-cache /cluster/project/sachan/ekaterina/.cache \
    --prompt-file configs/gsm8k_3shot_prompt.txt \
    --api-key-file configs/deepseek_api_key.txt \
    --save-path /cluster/project/sachan/ekaterina/.cache/train_gsm8k_Qwen2.5-Math-7B \
    --hub-repo rediska0123/train_gsm8k_Qwen2.5-Math-7B  # optionally save to huggingface
```

Test UHead (already pre-trained in `rediska0123/uhead_Qwen2.5-Math-7B`) and other UE baselines (CCP, MaxProb, ...):
```bash
PYTHONPATH=./ \
WANDB_PROJECT=YOUR_WANDB_PROJECT \
DEEPSEEK_API_KEY=$(configs/deepseek_api_key.txt) \
HYDRA_CONFIG=configs/polygraph_eval_claim_reasoning.yaml \
    python eval_uhead.py \
    model.path=Qwen/Qwen2.5-Math-7B \
    dataset=rediska0123/test_gsm8k_Qwen2.5-Math-7B \
    stat_calculators.2.cfg.uq_head_path=rediska0123/uhead_Qwen2.5-Math-7B
```