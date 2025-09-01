# Direct Online Best-of-N Implementation

This directory contains a streamlined implementation of online best-of-n evaluation with multiple scoring methods:
- **UHead**: Uncertainty-based scoring using learned uncertainty heads
- **PRM**: Process Reward Model scoring for step-by-step evaluation
- **ReasonEval**: Validity and redundancy scoring for reasoning quality

All implementations support:
- Two-phase evaluation: trajectory generation followed by correctness checking
- Resume capability with dataset validation
- Memory optimization through batch generation
- Multi-GPU support for large models
- Consistent directory-based result organization

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

**Memory optimization options** (to avoid OOM):
```bash
# Sequential generation (slowest but most memory efficient)
python online_bestofn/run_direct_online_bestofn.py \
    --sequential-generation \
    # ... other arguments

# Batch generation with smaller batch size
python online_bestofn/run_direct_online_bestofn.py \
    --batch-size 2 \
    --n 10 \
    # ... other arguments

# Control UHead feature extraction batch size (default: 1)
python online_bestofn/run_direct_online_bestofn.py \
    --feature-batch-size 1 \
    --n 10 \
    # ... other arguments
```

The `--feature-batch-size` parameter controls how many candidates are processed at once during UHead feature extraction. Lower values use less memory but may be slightly slower. Default is 1 for maximum memory efficiency.

**Automatic OOM Fallback**: If the UHead feature extraction runs out of memory, it will progressively reduce the batch size:
1. First attempt: Use the specified `--feature-batch-size`
2. On OOM: Try half the current batch size (minimum 2)
3. Still OOM: Try batch_size=2
4. Still OOM: Try batch_size=1
5. If still OOM with batch_size=1, the error is raised

This progressive fallback ensures the evaluation can adapt to available GPU memory while maintaining the best possible performance.

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

**Multi-GPU support**: By default, the base model uses `cuda:0` and PRM uses `cuda:1`. To customize:
```bash
python online_bestofn/run_direct_online_bestofn_prm.py \
    --device cuda:2 \
    --prm-device cuda:3 \
    # ... other arguments
```

**Memory optimization options** (to avoid OOM):
```bash
# Sequential generation (slowest but most memory efficient)
python online_bestofn/run_direct_online_bestofn_prm.py \
    --sequential-generation \
    # ... other arguments

# Batch generation with smaller batch size
python online_bestofn/run_direct_online_bestofn_prm.py \
    --batch-size 2 \
    --n 10 \
    # ... other arguments
```

When using `--batch-size`, candidates will be generated in smaller batches to avoid OOM. For example, with `--n 10 --batch-size 2`, it will generate 5 batches of 2 candidates each.

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
    --save-dir online_bon_results \
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
    --save-dir online_bon_results \
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
- `--resume` / `--no-resume`: Resume from existing save files (default: resume enabled)
- `--n-threads`: Number of threads for DeepSeek verification (default: 1)
- `--annotation-prompt-type`: Type of annotation prompt for DeepSeek (`unique` or `non_unique`, default: non_unique), if your task has unique answers, use unique.
- `--subset`: Run only on a subset for debug usage
- `--max-steps`: Maximum number of reasoning steps (default: 30)

**Memory optimization options** (to avoid OOM):
```bash
# Sequential generation (slowest but most memory efficient)
python online_bestofn/run_direct_online_bestofn_reasoneval_separate.py \
    --sequential-generation \
    # ... other arguments

# Batch generation with smaller batch size
python online_bestofn/run_direct_online_bestofn_reasoneval_separate.py \
    --batch-size 2 \
    --n 10 \
    # ... other arguments
```

**Multi-GPU support**: By default, the base model uses `cuda:0` and ReasonEval uses `cuda:1`. To customize:
```bash
python online_bestofn/run_direct_online_bestofn_reasoneval_separate.py \
    --device cuda:2 \
    --reasoneval-device cuda:3 \
    # ... other arguments
```

**Note on GPU assignment**: When running with bash scripts that set `CUDA_VISIBLE_DEVICES`, the device indices are remapped. For example, if `CUDA_VISIBLE_DEVICES=4,0`, then `cuda:0` refers to physical GPU 4 and `cuda:1` refers to physical GPU 0.

## Viewing Results

### Print Results Table
Use the `print_results.py` script to view a formatted table of all results for a dataset:

```bash
# Basic usage
python online_bestofn/print_results.py online_bon_results/test_gsm8k_large_Qwen3-8B

# Sort by accuracy (highest first)
python online_bestofn/print_results.py online_bon_results/test_gsm8k_large_Qwen3-8B --sort-by accuracy

# Sort by model type
python online_bestofn/print_results.py online_bon_results/test_gsm8k_large_Qwen3-8B --sort-by type
```

The table displays:
- Model name and type (UHead, PRM, ReasonEval)
- Total samples, completed, and errors
- Accuracy (overall and for completed samples only)
- Average number of reasoning steps
- Correctness verification mode used

Example output:
```
================================================================================
Results for dataset: test_gsm8k_large_Qwen3-8B
================================================================================

+----------------------------------------+--------------------+-------+-----------+--------+----------+-----------------+-----------+-------------+
| Model                                  | Type               | Total | Completed | Errors | Accuracy | Acc (Completed) | Avg Steps | Correctness |
+========================================+====================+=======+===========+========+==========+=================+===========+=============+
| ReasonEval-7B_validity                 | ReasonEval (validity) | 1019  | 1019      | 0      | 78.2%    | 78.2%           | 12.3      | exact_match |
| ReasonEval-7B_redundancy               | ReasonEval (redundancy) | 1019  | 1019      | 0      | 76.5%    | 76.5%           | 11.8      | exact_match |
| ReasonEval-7B_both                     | ReasonEval (both)     | 1019  | 1019      | 0      | 79.1%    | 79.1%           | 12.1      | exact_match |
| Qwen2.5-Math-7B-PRM800K                | PRM                   | 1019  | 1019      | 0      | 80.3%    | 80.3%           | 10.5      | exact_match |
| uhead_claim_Qwen3-8B_prm12k            | UHead                 | 1019  | 1018      | 1      | 77.8%    | 77.9%           | 11.2      | exact_match |
+----------------------------------------+--------------------+-------+-----------+--------+----------+-----------------+-----------+-------------+
```
