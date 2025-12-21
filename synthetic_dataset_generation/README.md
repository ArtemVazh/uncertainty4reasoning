# Claim Extraction and Verification

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


## 2) Verify thinking claims
After extracting thinking-mode claims, `run_verify_claims_thinking.py` re-scores them with a fact-checker (e.g., local vLLM) without re-generating claims. Key flags: `--extracted-claims-path-prefix`, `--shards-to-load`, `--prompt-file`, `--anno-model`, `--fact-check-base-url`, `--api-cache`, `--save-path`, `--hf-save-path`.

Example (see `synthetic_dataset_generation/scripts/gpt_oss_annotate_qwen3_1.7b_thinking_0_3.sh`):
```bash
PYTHONPATH=. python -m synthetic_dataset_generation.run_verify_claims_thinking \
  --prompt-file configs/qwen3_prompt_thinking.txt \
  --extracted-claims-path-prefix /path/to/extracted_shards_prefix \
  --shards-to-load 0 1 2 3 \
  --anno-model openai/gpt-oss-120b \
  --fact-check-base-url http://localhost:8000/v1 \
  --save-path gen_data/train_prm800k_gpt-oss-annotated_thinking_shards_0_3 \
  --hf-save-path your_hf_namespace/train_prm800k_gpt-oss-annotated_thinking_shards_0_3
```
