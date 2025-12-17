#!/usr/bin/env python3
"""
Debugging script for StepsExtractorThinking.
Tests the step extraction on the example from test.txt.
"""

import sys
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
from argparse import Namespace

# Add the utils directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from synthetic_dataset_generation.utils.steps_extractor_thinking import StepsExtractorThinking
from lm_polygraph.stat_calculators.extract_claims import Claim
from synthetic_dataset_generation.run_extract_verify_claims_thinking import extract_tokens_of_reply




def create_tokenizer(model_path: str, cache_dir: str = None):
    """Create Qwen/Qwen3-8b tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        print(f"Using {model_path} tokenizer from transformers")
        return tokenizer
    except Exception as e:
        print(f"Error loading {model_path} tokenizer: {e}")
        raise


def debug_extraction(
    dataset_path: str = "JingweiNi/train_prm800k_qwen3_8b_thinking_texts",
    model_path: str = "Qwen/Qwen3-8B",
    prompt_file: str = "configs/qwen3_prompt_thinking.txt",
    sample_idx: int = 0,
    hf_cache: str = None,
):
    """Main debugging function."""
    # Load dataset from HuggingFace (same as run_extract_verify_claims_thinking.py)
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)['train']
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    
    # Select a single sample for debugging
    if sample_idx >= len(dataset):
        print(f"Warning: sample_idx {sample_idx} is out of range. Using index 0.")
        sample_idx = 0
    
    # Create a single-item dataset for processing
    single_sample_dataset = dataset.select([sample_idx])
    print(f"Using sample at index {sample_idx}")
    print(f"Question: {single_sample_dataset[0]['question'][:200]}...")
    print(f"Reply length: {len(single_sample_dataset[0]['reply'])} characters")
    print()
    
    # Create tokenizer (same as run_extract_verify_claims_thinking.py)
    tokenizer = create_tokenizer(model_path, cache_dir=hf_cache)
    
    # Create model (same as run_extract_verify_claims_thinking.py - uses Namespace)
    model = Namespace(tokenizer=tokenizer)
    
    # Load prompt file (same as run_extract_verify_claims_thinking.py)
    prompt_path = Path(__file__).parent.parent / prompt_file
    if not prompt_path.exists():
        prompt_path = Path(prompt_file)
    prompt = open(prompt_path, 'r').read()
    print(f"Loaded prompt from: {prompt_path}")
    print()
    
    # Extract tokens of reply (same as run_extract_verify_claims_thinking.py)
    print("Extracting reply tokens...")
    greedy_tokens_list = extract_tokens_of_reply(single_sample_dataset, tokenizer, prompt)
    greedy_tokens = greedy_tokens_list[0]  # Get tokens for the single sample
    print(f"Extracted {len(greedy_tokens)} tokens")
    
    # Get reply text from dataset (same as run_extract_verify_claims_thinking.py)
    greedy_text = tokenizer.decode(greedy_tokens)
    
    # Create extractor
    extractor = StepsExtractorThinking(
        thinking_prefix="<think>",
        thinking_suffix="</think>",
        progress_bar=False,
        min_chars_per_step=20,
    )
    
    # Prepare inputs (same as run_extract_verify_claims_thinking.py)
    texts = single_sample_dataset["question"]  # Use question from dataset
    dependencies = {
        "greedy_texts": [greedy_text],  # Reply text from dataset
        "greedy_tokens": [greedy_tokens],  # Extracted reply tokens
    }
    
    print(f"Input question: {texts[0][:200]}...")
    print()
    
    # Extract steps
    print("Extracting steps...")
    print("=" * 80)
    try:
        result = extractor(dependencies, texts, model)
        
        claims = result["claims"][0]  # Get claims for first (and only) text
        claim_texts = result["claim_texts_concatenated"]
        claim_input_texts = result["claim_input_texts_concatenated"]
        
        print(f"\nExtracted {len(claims)} steps")
        print("=" * 80)
        print()
        
        # Display each step
        for i, claim in enumerate(claims, 1):
            print(f"\n{'='*80}")
            print(f"STEP {i} (Token IDs: {len(claim.aligned_token_ids)} tokens)")
            print(f"{'='*80}")
            print(rf"Claim Text: {claim.claim_text}")
            print(f"\nToken IDs: {claim.aligned_token_ids}")
            print(f"\nTokens: {[greedy_tokens[idx] for idx in claim.aligned_token_ids]}")
            print(f"\nDecoded text: {tokenizer.decode([greedy_tokens[idx] for idx in claim.aligned_token_ids])}")
            print()
        
        # Summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Total steps extracted: {len(claims)}")
        print(f"Average tokens per step: {sum(len(c.aligned_token_ids) for c in claims) / len(claims) if claims else 0:.2f}")
        print(f"Min tokens per step: {min(len(c.aligned_token_ids) for c in claims) if claims else 0}")
        print(f"Max tokens per step: {max(len(c.aligned_token_ids) for c in claims) if claims else 0}")
        print(f"Steps with < {extractor.min_chars_per_step} characters: {sum(1 for c in claims if len(c.claim_text) < extractor.min_chars_per_step)}")
        print()
        
        # Check for thinking prefix/suffix handling
        prefix_count = sum(1 for c in claims if "<think>" in c.claim_text)
        suffix_count = sum(1 for c in claims if "</think>" in c.claim_text)
        print(f"Steps containing thinking_prefix: {prefix_count}")
        print(f"Steps containing thinking_suffix: {suffix_count}")
        print()
        
        # Show claim_texts_concatenated
        print(f"Total claim texts concatenated: {len(claim_texts)}")
        print(f"Sample claim texts (first 3):")
        for i, ct in enumerate(claim_texts[:3], 1):
            print(f"  {i}. {ct[:100]}{'...' if len(ct) > 100 else ''}")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 80)
        print("Attempting direct split_to_steps call for debugging...")
        print("=" * 80)
        try:
            # Try calling split_to_steps directly
            steps = extractor.split_to_steps(greedy_text, greedy_tokens, tokenizer)
            print(f"\nDirect split_to_steps extracted {len(steps)} steps")
            for i, step in enumerate(steps[:5], 1):  # Show first 5 steps
                print(f"\nStep {i}: {step.claim_text[:100]}...")
        except Exception as e2:
            print(f"Direct call also failed: {e2}")
            traceback.print_exc()


if __name__ == "__main__":
    debug_extraction()

