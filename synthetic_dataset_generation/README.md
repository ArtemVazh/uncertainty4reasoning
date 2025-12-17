# Claim Extraction and Verification

Two scripts turn model traces into claim-level datasets:
- `run_extract_claims_thinking.py`: align generated tokens with questions and pull out claim spans.
- `run_extract_verify_claims.py`: (re-)extract claims and score each with correctness and informativeness labels.

## 1) Extract thinking claims
Input: HF dataset or disk path with `question`, `input_ids` (prompt + reply), and `reply` columns. Output: dataset saved to disk and (optionally) pushed to the Hub with an added `claims` column and `original_index` when sharded.

Key flags:
- `--dataset-path`, `--model-path`, `--prompt-file`: source data, tokenizer, and question template (`{q}` placeholder).
- `--save-path`, `--hf-save-path`: where to write the sharded result locally/HF.
- `--start-idx`, `--subset`, `--partition-num`, `--partition-idx`: slice/shard control for array jobs.
- `--hf-cache`: override cache dir for the tokenizer.

Example (matches `synthetic_dataset_generation/extract_claim_qwen3_1.7b_thinking.sh`):
```bash
PYTHONPATH=. python -m synthetic_dataset_generation.run_extract_claims_thinking \
  --dataset-path JingweiNi/train_prm800k_qwen3_1.7b_thinking_texts \
  --model-path Qwen/Qwen3-1.7B \
  --prompt-file configs/qwen3_prompt_thinking.txt \
  --save-path gen_data/train_prm800k_qwen3_1.7b_thinking_5000_extracted_0 \
  --hf-save-path JingweiNi/train_prm800k_qwen3_1.7b_thinking_5000_extracted_0 \
  --subset 5000 --partition-num 8 --partition-idx 0
```

## 2) Extract and verify claims
Input: dataset with `question`, `reply`, `input_ids`, `answer` (and optional task-specific fields noted in the code). The script extracts steps, calls a fact-checker, and writes targets aligned to each token.

Key flags:
- `--dataset-path`, `--model-path`, `--prompt-file`: same meaning as above (model only required when extracting, not when passing `--extracted-claims-paths`).
- `--api-key-file`, `--anno-model`, `--n-threads`, `--api-cache`, `--fact-check-base-url`: fact-checker settings (DeepSeek/OpenAI or local vLLM).
- `--save-path`, `--hf-save-path`: output dataset locations.
- `--start-idx`, `--subset`, `--partition-num`, `--partition-idx`, `--sample`, `--unique-questions`: sampling controls.

Minimal run:
```bash
PYTHONPATH=. python -m synthetic_dataset_generation.run_extract_verify_claims \
  --dataset-path /path/to/text_dataset \
  --model-path Qwen/Qwen3-1.7B \
  --prompt-file configs/qwen3_prompt.txt \
  --api-key-file configs/deepseek_api_key.txt \
  --save-path gen_data/train_gsm8k_qwen3_1.7b_verified \
  --hf-save-path your_hf_namespace/train_gsm8k_qwen3_1.7b_verified \
  --n-threads 16
```

For a full Slurm + vLLM setup, see `synthetic_dataset_generation/gpt_oss_annotate_qwen3_1.7b_thinking.sh`; adapt the Python module name to `run_extract_verify_claims.py` and point `--fact-check-base-url`/`--fact-check-api-key` at your deployment.
