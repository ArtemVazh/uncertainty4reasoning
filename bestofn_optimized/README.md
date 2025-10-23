## Optimized offline Best-of-N evaluation

1. Generate texts using `synthetic_dataset_generation/run_generate_texts.py`:

```bash
python -m synthetic_dataset_generation.run_generate_texts \
  --dataset-path rediska0123/test_math_no_prm800k_Qwen3-8B --dataset-split train \
  --model-path Qwen/Qwen3-8B \
  --save-path ./test_math_no_prm800k_Qwen3-8B_texts \
  --n-samples-per-input 32 --batch-size 8 --max-new-tokens 256 
```

2. Score texts with UHead and unsupervised baselines:

```bash
python -m bestofn_optimized.run_uhead \
  --dataset-path local,./test_math_no_prm800k_Qwen3-8B_texts \
  --model-path Qwen/Qwen3-8B \
  --uhead-path JingweiNi/uhead_claim_Qwen3-8B_fixed_prm_layer1_dim512_head16_e5_lr5e-4_pos3 \
  --save-path ./test_math_no_prm800k_Qwen3-8B_texts_uhead \
  --save-every 50 --batch-size 10
```

3. Score texts with all PRMs:

```bash
python -m bestofn_optimized.run_prm \
  --dataset-path local,./test_math_no_prm800k_Qwen3-8B_texts_uhead \
  --model-path Qwen/Qwen3-8B \
  --save-path ./test_math_no_prm800k_Qwen3-8B_texts_uhead_prm \
  --save-every 50
```

4. DeepSeek annotation

```bash
python -m bestofn_optimized.deepseek_annotation \
  --dataset-path local,./test_math_no_prm800k_Qwen3-8B_texts_uhead_prm \
  --save-path ./test_math_no_prm800k_Qwen3-8B_texts_uhead_prm_annotated \
  --prompt-file configs/qwen3_prompt_general.txt --n-threads 16
```

5. Visualize results: `bestofn_optimized/plot_bon_results.ipynb`
