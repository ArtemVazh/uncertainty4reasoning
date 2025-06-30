# Two-Part Evaluation System

This system splits the evaluation into two parts to optimize processing time:

## Part 1: Fast Evaluation (~1 hour)
Runs all uncertainty estimators and metrics except the expensive StepFactCheck (DeepSeek annotation).

## Part 2: Add StepFactCheck Annotation
Loads results from Part 1 and adds the StepFactCheck annotation with high parallelization.

## Usage

### Method 1: Using wrapper scripts

1. **Run Part 1:**
   ```bash
   python run_part1.py
   ```

2. **Run Part 2:**
   ```bash
   python add_stepfactcheck.py \
     --load_path /path/to/part1/results \
     --save_path /path/to/final/results \
     --n_threads 32
   ```

### Method 2: Using hydra directly

1. **Run Part 1:**
   ```bash
   python eval_uhead.py --config-path=configs --config-name=polygraph_eval_claim_reasoning_part1
   ```

2. **Run Part 2:**
   ```bash
   python add_stepfactcheck.py \
     --load_path ./workdir/cache/bio/Qwen/Qwen2.5-Math-7B/rediska0123-test_gsm8k/YYYY-MM-DD/HH-MM-SS \
     --save_path ./workdir/cache/bio/Qwen/Qwen2.5-Math-7B/rediska0123-test_gsm8k/YYYY-MM-DD/HH-MM-SS-with-stepfactcheck \
     --n_threads 32 \
     --progress_bar
   ```

## Key Differences from Original

### Part 1 Changes:
- **Removed**: `StepFactCheck` from `generation_metrics` in the config
- **Kept**: All UE estimators (RandomBaselineClaim, MaximumClaimProbability, etc.)
- **Kept**: All stat calculators (ClaimExtractor, Luh, etc.)

### Part 2 Features:
- **Loads**: Existing UEManager results from Part 1
- **Adds**: StepFactCheck with configurable `n_threads` (default: 32)
- **Recalculates**: UE metrics for the new generation metric
- **Saves**: Updated results with all metrics included

## Benefits

1. **Time Optimization**: Part 1 runs much faster without the expensive DeepSeek API calls
2. **Parallelization**: Part 2 can use high thread counts for StepFactCheck
3. **Fault Tolerance**: If Part 2 fails, you don't need to re-run Part 1
4. **Resource Management**: Can run Part 1 and Part 2 on different machines/times

## Arguments for Part 2

- `--load_path`: Path to directory containing `ue_manager.pth` from Part 1
- `--save_path`: Where to save the final results
- `--n_threads`: Number of threads for StepFactCheck (increase for faster processing)
- `--prompt_file`: Path to prompt file (default: `configs/qwen3_prompt.txt`)
- `--model`: Model for StepFactCheck (default: `deepseek-reasoner`)
- `--cache_path`: Cache directory (default: `./workdir/cache`)
- `--progress_bar`: Show progress bar during annotation 