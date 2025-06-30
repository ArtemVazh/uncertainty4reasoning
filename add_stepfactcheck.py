#!/usr/bin/env python3
"""
Script to add StepFactCheck annotation to an existing UEManager with high parallelization.
This is the second part of the two-part evaluation process.
"""

import argparse
import os
import sys
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("add_stepfactcheck")

# Import required modules
from lm_polygraph.utils.manager import UEManager
from lm_polygraph.ue_metrics.ue_metric import get_random_scores, normalize_metric
from synthetic_dataset_generation.utils.step_fact_check import StepFactCheck
from lm_polygraph.ue_metrics import ROCAUC, PRAUC

def load_ue_manager(load_path: str) -> UEManager:
    """Load UEManager from the specified path."""
    load_path = Path(load_path)
    if load_path.is_dir():
        load_path = load_path / "ue_manager.pth"
    
    if not load_path.exists():
        raise FileNotFoundError(f"UEManager file not found at {load_path}")
    
    log.info(f"Loading UEManager from {load_path}")
    return UEManager.load(str(load_path))

def add_stepfactcheck_annotation(
    manager: UEManager,
    n_threads: int = 32,
    progress_bar: bool = True,
    prompt_file: str = "configs/qwen3_prompt.txt",
    model: str = "deepseek-reasoner"
) -> UEManager:
    """
    Add StepFactCheck annotation to the loaded UEManager.
    
    Args:
        manager: Loaded UEManager instance
        n_threads: Number of threads for parallel processing
        progress_bar: Whether to show progress bar
        prompt_file: Path to prompt file for StepFactCheck
        model: Model name for StepFactCheck
    
    Returns:
        Updated UEManager with StepFactCheck annotations
    """
    
    # Get the existing data from the manager
    if "greedy_texts" not in manager.stats:
        raise ValueError("No greedy_texts found in UEManager. Make sure the first part was completed successfully.")
    
    if "target_texts" not in manager.stats:
        raise ValueError("No target_texts found in UEManager. Make sure the first part was completed successfully.")
    
    if "input_texts" not in manager.stats:
        raise ValueError("No input_texts found in UEManager. Make sure the first part was completed successfully.")
    
    greedy_texts = manager.stats["greedy_texts"]
    target_texts = manager.stats["target_texts"]
    input_texts = manager.stats["input_texts"]
    
    log.info(f"Found {len(greedy_texts)} generated texts to annotate")
    log.info(f"Available stats in UEManager: {list(manager.stats.keys())}")
    
    # First, we need to extract claims if they don't exist
    batch_stats = {
        "greedy_texts": greedy_texts,
        "target_texts": target_texts,
        "input_texts": input_texts
    }
    
    # Add greedy_tokens if available (required by some stat calculators)
    if "greedy_tokens" in manager.stats:
        batch_stats["greedy_tokens"] = manager.stats["greedy_tokens"]
    
    # Check if claims are already available and add them to batch_stats
    if "claims" in manager.stats:
        log.info("Using existing claims from UEManager")
        batch_stats["claims"] = manager.stats["claims"]
        if "claim_texts_concatenated" in manager.stats:
            batch_stats["claim_texts_concatenated"] = manager.stats["claim_texts_concatenated"]
        if "claim_input_texts_concatenated" in manager.stats:
            batch_stats["claim_input_texts_concatenated"] = manager.stats["claim_input_texts_concatenated"]
    else:
        log.info("Claims not found in UEManager. StepFactCheck will try to work without them or extract them internally.")
    
    log.info(f"Creating StepFactCheck with {n_threads} threads")
    
    # Create StepFactCheck instance with high parallelization
    stepfactcheck = StepFactCheck(
        prompt_file=prompt_file,
        model=model,
        n_threads=n_threads,
        progress_bar=progress_bar
    )
    
    log.info("Running StepFactCheck annotation...")
    
    # Run StepFactCheck on all the data
    stepfactcheck_results = stepfactcheck(batch_stats, target_texts=target_texts)
    
    if not isinstance(stepfactcheck_results, list):
        stepfactcheck_results = stepfactcheck_results.tolist()
    
    log.info(f"StepFactCheck completed. Generated {len(stepfactcheck_results)} annotations")
    
    # Add StepFactCheck results to generation metrics
    stepfactcheck_level = stepfactcheck.level  # Should be 'sequence'
    stepfactcheck_name = str(stepfactcheck)
    
    manager.gen_metrics[(stepfactcheck_level, stepfactcheck_name)] = stepfactcheck_results
    
    # Add ROC AUC metrics if not already present since we now have binary labels from StepFactCheck
    if not manager.ue_metrics:
        # If no UE metrics, initialize with basic set
        from lm_polygraph.ue_metrics import PredictionRejectionArea
        manager.ue_metrics = [
            PredictionRejectionArea(),
            PredictionRejectionArea(max_rejection=0.5),
            ROCAUC(),
            PRAUC(),
        ]
    else:
        # Add ROC metrics if not already present
        existing_metric_names = [str(metric) for metric in manager.ue_metrics]
        if "ROCAUC" not in existing_metric_names:
            manager.ue_metrics.append(ROCAUC())
        if "PRAUC" not in existing_metric_names:
            manager.ue_metrics.append(PRAUC())
    
    # Recalculate UE metrics with the new generation metric
    log.info("Recalculating UE metrics with StepFactCheck...")
    
    for ue_metric in manager.ue_metrics:
        log.info(f"Processing UE metric: {ue_metric}")
        
        generation_metric = stepfactcheck_results
        gen_level = stepfactcheck_level
        gen_name = stepfactcheck_name
        
        oracle_score_all = ue_metric(-np.array(generation_metric), np.array(generation_metric))
        random_score_all = get_random_scores(ue_metric, np.array(generation_metric))
        
        for (e_level, e_name), estimator_values in manager.estimations.items():
            if gen_level != e_level:
                continue
            
            if len(estimator_values) != len(generation_metric):
                log.warning(
                    f"Length mismatch for {e_name} and {gen_name}: "
                    f"{len(estimator_values)} vs {len(generation_metric)}. Skipping."
                )
                continue
            
            # Handle NaNs
            def _delete_nans(ue, metric):
                metric = np.asarray(metric)
                clipped_ue = np.nan_to_num(ue, nan=-1e7, neginf=-1e7, posinf=1e7)
                is_nan_metric_mask = np.isnan(metric)
                clipped_ue = clipped_ue[~is_nan_metric_mask]
                new_metric = metric[~is_nan_metric_mask]
                return clipped_ue, new_metric
            
            n_nans = np.sum(~np.isfinite(estimator_values))
            if n_nans > 0:
                log.warning(f"Found {n_nans} NaNs in {e_name} estimator.")
            
            n_nans = np.sum(~np.isfinite(generation_metric))
            if n_nans > 0:
                log.warning(f"Found {n_nans} NaNs in {gen_name} generation metric.")
            
            ue, metric = _delete_nans(estimator_values, generation_metric)
            
            if len(ue) == 0:
                manager.metrics[e_level, e_name, gen_name, str(ue_metric)] = np.nan
            else:
                if len(ue) != len(estimator_values):
                    oracle_score = ue_metric(-metric, metric)
                    random_score = get_random_scores(ue_metric, metric)
                else:
                    oracle_score = oracle_score_all
                    random_score = random_score_all
                
                ue_metric_val = ue_metric(ue, metric)
                manager.metrics[e_level, e_name, gen_name, str(ue_metric)] = ue_metric_val
                manager.metrics[e_level, e_name, gen_name, str(ue_metric) + "_normalized"] = \
                    normalize_metric(ue_metric_val, oracle_score, random_score)
    
    log.info("UE metrics recalculation completed")
    return manager

def main():
    parser = argparse.ArgumentParser(
        description="Add StepFactCheck annotation to existing UEManager results"
    )
    parser.add_argument(
        "--load_path", 
        type=str, 
        required=True,
        help="Path to directory containing ue_manager.pth or direct path to ue_manager.pth"
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        required=True,
        help="Path to save the updated UEManager"
    )
    parser.add_argument(
        "--n_threads", 
        type=int, 
        default=32,
        help="Number of threads for StepFactCheck parallelization (default: 32)"
    )
    parser.add_argument(
        "--progress_bar", 
        action="store_true",
        help="Show progress bar during StepFactCheck processing"
    )
    parser.add_argument(
        "--prompt_file", 
        type=str, 
        default="configs/qwen3_prompt.txt",
        help="Path to prompt file for StepFactCheck (default: configs/qwen3_prompt.txt)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="deepseek-reasoner",
        help="Model name for StepFactCheck (default: deepseek-reasoner)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.load_path):
        log.error(f"Load path does not exist: {args.load_path}")
        sys.exit(1)
    
    # Load existing UEManager
    try:
        manager = load_ue_manager(args.load_path)
    except Exception as e:
        log.error(f"Failed to load UEManager: {e}")
        sys.exit(1)
    
    # Add StepFactCheck annotation
    try:
        updated_manager = add_stepfactcheck_annotation(
            manager=manager,
            n_threads=args.n_threads,
            progress_bar=args.progress_bar,
            prompt_file=args.prompt_file,
            model=args.model
        )
    except Exception as e:
        log.error(f"Failed to add StepFactCheck annotation: {e}")
        sys.exit(1)
    
    # Save updated UEManager
    try:
        log.info(f"Saving updated UEManager to {args.save_path}")
        updated_manager.save(args.save_path)
        log.info("Successfully saved updated UEManager with StepFactCheck annotation")
    except Exception as e:
        log.error(f"Failed to save UEManager: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 