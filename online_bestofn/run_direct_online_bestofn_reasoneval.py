#!/usr/bin/env python3
"""
Run direct online best-of-n evaluation with ReasonEval
"""

import argparse
import os
import logging
import random
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from lm_polygraph import WhiteboxModel
import traceback
from online_bestofn.direct_online_bestofn_reasoneval import DirectOnlineBestOfNReasonEval
from online_bestofn.direct_online_bestofn import _is_correct_answer
from utils import parse_ans

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("direct_online_bon_reasoneval")

from transformers import AutoTokenizer, AutoModelForCausalLM


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = None
    tokenizer.padding_side = 'left'  # Fix padding side for decoder-only models
    return tokenizer


def load_model(model_path: str, device_map: str):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, trust_remote_code=True)
    return model


def get_parser():
    """Command line arguments"""
    parser = argparse.ArgumentParser(description="Direct online best-of-n with ReasonEval")
    
    # Dataset arguments
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Dataset to evaluate on (HuggingFace name or local path)")
    parser.add_argument("--dataset-split", type=str, default="test", 
                        help="Dataset split to use")
    parser.add_argument("--subset", type=int, default=None,
                        help="Only process first N samples from dataset")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-1.7B",
                        help="Base model for generation")
    parser.add_argument("--reasoneval-path", type=str, default="GAIR/ReasonEval-7B",
                        help="Path to ReasonEval model (7B or 34B)")
    
    # Generation arguments
    parser.add_argument("--n", type=int, default=10,
                        help="Number of candidates per step")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    parser.add_argument("--max-new-tokens", type=int, default=250,
                        help="Max tokens per step")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="Maximum number of reasoning steps")
    
    # ReasonEval-specific arguments
    parser.add_argument("--aggregation", type=str, default="default",
                        choices=["default", "validity", "redundancy"],
                        help="How to aggregate validity/redundancy scores")
    
    # Output arguments
    parser.add_argument("--save-path", type=str, required=True,
                        help="Path to save results")
    parser.add_argument("--prompt-file", type=str, default=None,
                        help="Path to prompt template file (optional)")
    
    # System arguments
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--hf-cache", type=str, default=None,
                        help="HuggingFace cache directory")
    
    return parser


def load_prompt_template(prompt_file: str) -> str:
    """Load prompt template from file"""
    if prompt_file and os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            return f.read().strip()
    else:
        # Default prompt template for ReasonEval
        return "Question: {question}\n\nLet's solve this step by step.\n\n"


def prepare_dataset_with_prompts(dataset, prompt_template: str):
    """Add prompts to dataset questions"""
    
    def add_prompt(example):
        # Format prompt with question
        if "{question}" in prompt_template:
            example["question"] = prompt_template.format(question=example["question"])
        else:
            example["question"] = prompt_template + example["question"]
        return example
    
    return dataset.map(add_prompt)


def main(args):
    """Main evaluation function"""
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    log.info(f"Set random seed to {args.seed}")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Load dataset
    log.info(f"Loading dataset: {args.dataset_path} ({args.dataset_split})")
    dataset = load_dataset(
        args.dataset_path, 
        split=args.dataset_split,
        cache_dir=args.hf_cache
    )
    
    # Add prompts if provided
    if args.prompt_file:
        prompt_template = load_prompt_template(args.prompt_file)
        dataset = prepare_dataset_with_prompts(dataset, prompt_template)
        log.info(f"Added prompts from {args.prompt_file}")
    
    # Load model
    log.info(f"Loading model: {args.model_path}")
    tokenizer = load_tokenizer(args.model_path)
    base_model = load_model(args.model_path, args.device)
    base_model.eval()
    model = WhiteboxModel(base_model, tokenizer)
    
    # Run direct online best-of-n with ReasonEval
    log.info(f"Starting direct online best-of-n evaluation with ReasonEval")
    log.info(f"  - ReasonEval model: {args.reasoneval_path}")
    log.info(f"  - Aggregation: {args.aggregation}")
    log.info(f"  - Candidates per step: {args.n}")
    log.info(f"  - Temperature: {args.temperature}")
    log.info(f"  - Max tokens per step: {args.max_new_tokens}")
    log.info(f"  - Max steps: {args.max_steps}")
    
    # Create generator with ReasonEval
    generator = DirectOnlineBestOfNReasonEval(
        model=model,
        reasoneval_model_path=args.reasoneval_path,
        candidates_per_step=args.n,
        max_steps=args.max_steps,
        temperature=args.temperature,
        device=args.device,
        aggregation=args.aggregation,
        verbose=args.verbose
    )
    
    # Process dataset
    results = []
    subset_size = min(args.subset, len(dataset)) if args.subset else len(dataset)
    
    for i in tqdm(range(subset_size), desc="Processing samples"):
        sample = dataset[i]
        
        if args.verbose:
            log.info(f"\n{'='*60}")
            log.info(f"Sample {i+1}/{subset_size}")
            log.info(f"Question: {sample['question'][:200]}...")
            log.info(f"Gold Answer: {sample['answer']}")
        
        try:
            # Generate trajectory
            result = generator.generate_trajectory(sample["question"])
            
            # Extract generated answer
            generated_text = result["trajectory"]
            if sample["question"] in generated_text:
                generated_text = generated_text.replace(sample["question"], "").strip()
            
            # Check correctness
            is_correct = _is_correct_answer(generated_text, sample["answer"])
            
            # Store result
            results.append({
                "index": i,
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "generated_trajectory": result["trajectory"],
                "generated_answer": generated_text,
                "steps": result["steps"],
                "step_scores": result["step_scores"],  # These are uncertainties
                "is_correct": is_correct,
                "completed": result["completed"]
            })
            
            if args.verbose:
                log.info(f"Generated: {generated_text}")
                log.info(f"Generated answer: {parse_ans(generated_text)}")
                log.info(f"Gold answer: {parse_ans(sample['answer'])}")
                log.info(f"Correct: {is_correct}")
                log.info(f"Num steps: {len(result['steps'])}")
                if result['step_scores']:
                    log.info(f"Avg step uncertainty: {np.mean(result['step_scores']):.3f}")
            
        except Exception as e:
            log.error(f"Error processing sample {i}: {e}")
            traceback.print_exc()
            
            results.append({
                "index": i,
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "error": str(e),
                "is_correct": False,
                "completed": False
            })
        
        # Save periodically
        if (i + 1) % 10 == 0:
            torch.save(results, args.save_path)
            log.info(f"Saved {len(results)} results to {args.save_path}")
    
    # Final save
    torch.save(results, args.save_path)
    log.info(f"Final save: {len(results)} results to {args.save_path}")
    
    # Cleanup
    generator.cleanup()
    
    # Print summary statistics
    correct = sum(r.get("is_correct", False) for r in results)
    completed = sum(r.get("completed", False) for r in results)
    errors = sum("error" in r for r in results)
    
    log.info(f"\n{'='*60}")
    log.info(f"Evaluation Summary:")
    log.info(f"  - Total samples: {len(results)}")
    log.info(f"  - Completed: {completed} ({completed/len(results):.1%})")
    log.info(f"  - Correct: {correct} ({correct/len(results):.1%})")
    log.info(f"  - Errors: {errors}")
    
    if completed > 0:
        log.info(f"  - Accuracy (of completed): {correct/completed:.1%}")
    
    # Average statistics for uncertainties
    all_uncertainties = []
    all_steps = []
    for r in results:
        if "step_scores" in r and r["step_scores"]:
            all_uncertainties.extend(r["step_scores"])
            all_steps.append(len(r["steps"]))
    
    if all_uncertainties:
        log.info(f"\nStep Statistics:")
        log.info(f"  - Avg steps per trajectory: {np.mean(all_steps):.1f}")
        log.info(f"  - Avg step uncertainty: {np.mean(all_uncertainties):.3f}")
        log.info(f"  - Min step uncertainty: {np.min(all_uncertainties):.3f}")
        log.info(f"  - Max step uncertainty: {np.max(all_uncertainties):.3f}")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)