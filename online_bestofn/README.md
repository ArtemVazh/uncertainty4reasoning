# Direct Online Best-of-N Implementation

This directory contains a streamlined implementation of online best-of-n evaluation using direct UHead scoring, I forgone the stat calculator pipeline for step control.

## Usage
To run eval on uhead
```bash
python online_bestofn/run_direct_online_bestofn.py \
    --dataset-path rediska0123/test_gsm8k_large_Qwen3-8B \
    --dataset-split train \
    --save-path online_bon_results/uhead_claim_Qwen3-8B_prm12k_cr_1.0_1000_clariden_4e_b32_last_gsm8k_large.torch \
    --model-path Qwen/Qwen3-8B \
    --uhead-path JingweiNi/uhead_claim_Qwen3-8B_prm12k_cr_1.0_1000_clariden_4e_b32_last \
    --prompt-file configs/qwen3_prompt_general.txt \
    --n 10 --max-new-tokens 512  --seed 42 --verbose --temperature 1.5
```

To run eval on PRM
```bash
python online_bestofn/run_direct_online_bestofn_prm.py \
    --dataset-path rediska0123/test_gsm8k_large_Qwen3-8B \
    --dataset-split train \
    --save-path online_bon_results/test_prm_qwen2_5_7b_prm800k_gsm8k_large.torch \
    --model-path Qwen/Qwen3-8B \
    --prm-path Qwen/Qwen2.5-Math-7B-PRM800K \
    --prompt-file configs/qwen3_prompt_general.txt \
    --n 10 --max-new-tokens 512 --seed 42 --verbose --temperature 1.5
```
