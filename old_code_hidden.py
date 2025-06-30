import hydra
import os
import transformers
from pathlib import Path
from omegaconf import OmegaConf
import uuid
from hydra.core.hydra_config import HydraConfig
import argparse
import sys
import numpy as np

import logging
import os
os.environ["TRANSFORMERS_NO_SAFE_LOAD"] = "1" 
log = logging.getLogger("lm_polygraph")
logging.getLogger("httpx").setLevel(logging.WARNING)

from lm_polygraph.utils.manager import UEManager
from lm_polygraph.utils.dataset import Dataset
from lm_polygraph.utils.model import WhiteboxModel, BlackboxModel
from lm_polygraph.utils.processor import Logger
from lm_polygraph.generation_metrics import *
from lm_polygraph.ue_metrics import *
from lm_polygraph.utils.common import load_external_module
from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.defaults.register_default_stat_calculators import (
    register_default_stat_calculators,
)
from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentStatCalculator,
)
from lm_polygraph.utils.factory_estimator import FactoryEstimator
from lm_polygraph.utils.factory_stat_calculator import StatCalculatorContainer
from synthetic_dataset_generation.utils.step_fact_check import StepFactCheck
from lm_polygraph.stat_calculators.extract_claims import ExtractClaims
from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']


def main():
    parser = argparse.ArgumentParser(description="Add StepFactCheck annotation to existing UEManager results")
    parser.add_argument("--load_path", type=str, required=True, 
                       help="Path to the directory containing ue_manager.pth from the first script")
    parser.add_argument("--save_path", type=str, required=True,
                       help="Path to save the updated results")
    parser.add_argument("--prompt_file", type=str, default="configs/qwen3_prompt.txt",
                       help="Path to the prompt file for StepFactCheck")
    parser.add_argument("--model", type=str, default="deepseek-reasoner",
                       help="Model to use for StepFactCheck")
    parser.add_argument("--n_threads", type=int, default=32,
                       help="Number of threads for StepFactCheck (increase for faster processing)")
    parser.add_argument("--cache_path", type=str, default="./workdir/cache",
                       help="Cache path for datasets")
    parser.add_argument("--progress_bar", action="store_true", default=False,
                       help="Show progress bar during StepFactCheck")
    
    args = parser.parse_args()
    
    log.info(f"Loading existing UEManager from {args.load_path}")
    
    # Load the existing UEManager
    manager_path = Path(args.load_path) / "ue_manager.pth"
    if not manager_path.exists():
        raise FileNotFoundError(f"UEManager file not found at {manager_path}")
    
    # Load the manager
    manager = UEManager.load(str(manager_path))
    
    log.info("Loaded existing UEManager successfully")
    log.info(f"Existing stats keys: {list(manager.stats.keys())}")
    log.info(f"Existing gen_metrics keys: {list(manager.gen_metrics.keys())}")
    
    # Check if we have the required stats for StepFactCheck
    required_stats = ["greedy_texts", "input_texts"]
    missing_stats = [stat for stat in required_stats if stat not in manager.stats or not manager.stats[stat]]
    
    if missing_stats:
        raise ValueError(f"Missing required stats for StepFactCheck: {missing_stats}. "
                        f"Make sure the first script generated greedy_texts and input_texts.")
    
    # Check for claims - if not available, we'll need to extract them
    if "claims" not in manager.stats or not manager.stats["claims"]:
        log.warning("Claims not found in stats. Attempting to extract claims from greedy_texts...")
        
        # Extract claims using the same extractor from the pipeline
        claim_extractor = ExtractClaims()
        
        # Prepare stats for claim extraction
        extract_stats = {
            "greedy_texts": manager.stats["greedy_texts"],
            "greedy_tokens": manager.stats.get("greedy_tokens", [])
        }
        
        log.info("Extracting claims from texts...")
        try:
            claims = claim_extractor(extract_stats)
            log.info(f"Successfully extracted {len(claims)} claim sets")
            
            # Add claims to the batch_stats
            batch_stats = {
                "input_texts": manager.stats["input_texts"],
                "greedy_texts": manager.stats["greedy_texts"],
                "claims": claims
            }
        except Exception as e:
            log.error(f"Failed to extract claims: {e}")
            raise ValueError("Could not extract claims from greedy_texts. Make sure the texts contain parseable reasoning steps.")
    else:
        log.info("Claims found in stats. Proceeding with existing claims...")
        batch_stats = {
            "input_texts": manager.stats["input_texts"],
            "greedy_texts": manager.stats["greedy_texts"],
            "claims": manager.stats["claims"]
        }
    
    log.info("All required stats are available. Proceeding with StepFactCheck annotation...")
    
    # Create StepFactCheck metric with high n_threads
    step_fact_check = StepFactCheck(
        prompt_file=args.prompt_file,
        model=args.model,
        n_threads=args.n_threads,
        progress_bar=args.progress_bar,
        cache_path=args.cache_path
    )
    
    log.info(f"Created StepFactCheck with {args.n_threads} threads")
    
    log.info("Starting StepFactCheck annotation...")
    
    # Apply StepFactCheck
    try:
        # StepFactCheck expects target_texts, we can use an empty list since it works with claims
        target_texts = [""] * len(manager.stats["greedy_texts"])
        
        step_fact_check_results = step_fact_check(batch_stats, target_texts=target_texts)
        
        if not isinstance(step_fact_check_results, list):
            step_fact_check_results = step_fact_check_results.tolist()
        
        log.info(f"StepFactCheck completed. Got {len(step_fact_check_results)} results")
        
        # Add the results to the manager
        key = (step_fact_check.level, str(step_fact_check))
        manager.gen_metrics[key] = step_fact_check_results
        
        log.info(f"Added StepFactCheck results with key: {key}")
        
        # Recalculate UE metrics with the new generation metric
        log.info("Recalculating UE metrics with new StepFactCheck results...")
        
        # We need to recreate ue_metrics for this
        from lm_polygraph.ue_metrics import PredictionRejectionArea, ROCAUC, PRAUC
        from lm_polygraph.ue_metrics.ue_metric import get_random_scores, normalize_metric
        from lm_polygraph.utils.manager import _delete_nans
        
        ue_metrics = [
            PredictionRejectionArea(),
            PredictionRejectionArea(max_rejection=0.5),
            ROCAUC(),
            PRAUC(),
        ]
        
        # Recalculate metrics
        for (gen_level, gen_name), generation_metric in manager.gen_metrics.items():
            if gen_name == str(step_fact_check):  # Only for our new metric
                for ue_metric in ue_metrics:
                    log.info(f"Calculating {ue_metric} for {gen_name}")
                    
                    oracle_score_all = ue_metric(
                        -np.array(generation_metric), np.array(generation_metric)
                    )
                    random_score_all = get_random_scores(
                        ue_metric, np.array(generation_metric)
                    )
                    
                    for (e_level, e_name), estimator_values in manager.estimations.items():
                        if gen_level != e_level:
                            continue
                        if len(estimator_values) != len(generation_metric):
                            log.warning(
                                f"Length mismatch between {e_name} ({len(estimator_values)}) "
                                f"and {gen_name} ({len(generation_metric)}). Skipping."
                            )
                            continue
                        
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
                            manager.metrics[
                                e_level, e_name, gen_name, str(ue_metric) + "_normalized"
                            ] = normalize_metric(ue_metric_val, oracle_score, random_score)
        
    except Exception as e:
        log.error(f"Error during StepFactCheck annotation: {e}")
        raise e
    
    # Save the updated manager
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Saving updated UEManager to {save_path}")
    manager.save(str(save_path))
    
    log.info("StepFactCheck annotation completed successfully!")
    log.info(f"Updated results saved to {save_path}")
    
    # Print summary
    log.info(f"Final gen_metrics keys: {list(manager.gen_metrics.keys())}")
    step_fact_check_keys = [key for key in manager.gen_metrics.keys() if "StepFactCheck" in key[1]]
    log.info(f"StepFactCheck keys: {step_fact_check_keys}")


if __name__ == "__main__":
    main() 