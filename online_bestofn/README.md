# Direct Online Best-of-N Implementation

This directory contains a streamlined implementation of online best-of-n evaluation using direct UHead scoring, I forgone the stat calculator pipeline for step control.

## Usage

### Running with UHead
```bash
python online_bestofn/run_direct_online_bestofn.py \
    --dataset-path rediska0123/test_gsm8k_large_Qwen3-8B \
    --dataset-split train \
    --save-dir online_bon_results \
    --model-path Qwen/Qwen3-8B \
    --uhead-path JingweiNi/uhead_claim_Qwen3-8B_prm12k_cr_1.0_1000_clariden_4e_b32_last \
    --prompt-file configs/qwen3_prompt_general.txt \
    --n 10 --max-new-tokens 512  --seed 42 --verbose --temperature 1.5
```

This will save results to: `online_bon_results/test_gsm8k_large_Qwen3-8B/uhead_claim_Qwen3-8B_prm12k_cr_1.0_1000_clariden_4e_b32_last.pt`

### Running with PRM
```bash
python online_bestofn/run_direct_online_bestofn_prm.py \
    --dataset-path rediska0123/test_gsm8k_large_Qwen3-8B \
    --dataset-split train \
    --save-dir online_bon_results \
    --model-path Qwen/Qwen3-8B \
    --prm-path Qwen/Qwen2.5-Math-7B-PRM800K \
    --prompt-file configs/qwen3_prompt_general.txt \
    --n 10 --max-new-tokens 512 --seed 42 --verbose --temperature 1.5
```

This will save results to: `online_bon_results/test_gsm8k_large_Qwen3-8B/Qwen2.5-Math-7B-PRM800K.pt`

### Running with ReasonEval

ReasonEval supports multiple evaluation criteria:
- **validity**: Select candidates with highest validity scores (better reasoning quality)
- **redundancy**: Select candidates with lowest redundancy scores (less repetitive)
- **both**: Select candidates with best combined score (validity - redundancy)
- **run_all**: Run all three criteria evaluations sequentially

#### Basic usage (single criterion):
```bash
python online_bestofn/run_direct_online_bestofn_reasoneval_separate.py \
    --dataset-path rediska0123/test_gsm8k_large_Qwen3-8B \
    --dataset-split train \
    --save-dir online_bon_results/reasoneval_results \
    --model-path Qwen/Qwen3-8B \
    --reasoneval-path GAIR/ReasonEval-7B \
    --prompt-file configs/qwen3_prompt_general.txt \
    --criterion validity \
    --n 10 --max-new-tokens 512 --seed 42 --verbose --temperature 1.5
```

#### Running all criteria:
```bash
python online_bestofn/run_direct_online_bestofn_reasoneval_separate.py \
    --dataset-path rediska0123/test_gsm8k_large_Qwen3-8B \
    --dataset-split train \
    --save-dir online_bon_results/reasoneval_results \
    --model-path Qwen/Qwen3-8B \
    --reasoneval-path GAIR/ReasonEval-7B \
    --prompt-file configs/qwen3_prompt_general.txt \
    --criterion run_all \
    --n 10 --max-new-tokens 512 --seed 42 --verbose --temperature 1.5
```

This will generate three result files in `online_bon_results/test_gsm8k_large_Qwen3-8B/`:
- `ReasonEval-7B_validity.pt`: Results using validity-based selection
- `ReasonEval-7B_redundancy.pt`: Results using redundancy-based selection  
- `ReasonEval-7B_both.pt`: Results using combined criterion (validity - redundancy)

#### Additional options:
- `--correctness-mode`: Choose between `exact_match` (default) or `deepseek` for answer verification
- `--resume`: Resume from existing save files
- `--n-threads`: Number of threads for DeepSeek verification (default: 1)
- `--annotation-prompt-type`: Type of annotation prompt for DeepSeek (`unique` or `non_unique`, default: non_unique), if your task has unique answers, use unique.
- `--subset`: Run only on a subset for debug usage.
