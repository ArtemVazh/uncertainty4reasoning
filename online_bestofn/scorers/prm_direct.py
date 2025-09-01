"""
Direct PRM scorer that bypasses the stat calculator pipeline for efficient stepwise scoring
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from transformers import AutoTokenizer, AutoModel

from lm_polygraph import WhiteboxModel
from synthetic_dataset_generation.utils.steps_extractor import StepsExtractor
from .base import RewardBasedScorer

log = logging.getLogger(__name__)


class DirectPRMScorer(RewardBasedScorer):
    """
    Direct PRM scorer that applies Process Reward Model without stat calculator pipeline.
    
    This implementation:
    1. Extracts claims/steps from candidates
    2. Formats them for PRM evaluation
    3. Computes step rewards directly
    4. Returns reward scores (higher = better)
    
    Much cleaner and more efficient than going through the full pipeline.
    """
    
    def __init__(
        self,
        model: WhiteboxModel,
        prm_model_path: str = "Qwen/Qwen2.5-Math-7B-PRM800K",
        device: str = "cuda",
        batch_size: int = 8,
        prompt_template: str = None
    ):
        super().__init__("DirectPRM")
        self.model = model
        self.prm_model_path = prm_model_path
        self.device = device
        self.batch_size = batch_size
        self.prompt_template = prompt_template or "Question: {q}\n\nLet's solve this step by step.\n\n"
        self.prm_model = None
        self.prm_tokenizer = None
        self.steps_extractor = StepsExtractor(progress_bar=False)
        
    def prepare_model(self):
        """Load PRM model and tokenizer"""
        if self.prm_model is None:
            log.info(f"Loading PRM model from {self.prm_model_path}")
            self.prm_tokenizer = AutoTokenizer.from_pretrained(
                self.prm_model_path, 
                trust_remote_code=True
            )
            self.prm_model = AutoModel.from_pretrained(
                self.prm_model_path,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).eval()
            
    def cleanup(self):
        """Free PRM model memory"""
        if self.prm_model is not None:
            del self.prm_model
            self.prm_model = None
            del self.prm_tokenizer
            self.prm_tokenizer = None
            torch.cuda.empty_cache()
    
    def compute_claim_rewards(
        self,
        trajectory: str,
        candidates: List[str],
        **kwargs
    ) -> List[List[float]]:
        """
        Compute reward scores for claims in each candidate.
        
        Args:
            trajectory: Current trajectory text
            candidates: List of candidate next steps
            
        Returns:
            List of claim reward lists (one per candidate)
        """
        self.prepare_model()
        
        if not candidates:
            return []
        
        # Extract question from trajectory
        question = self._extract_question(trajectory)
        
        # Score all candidates
        all_rewards = []
        
        for candidate in candidates:
            try:
                rewards = self._score_single_candidate(question, trajectory, candidate)
                all_rewards.append(rewards)
            except Exception as e:
                log.warning(f"Failed to score candidate: {e}")
                all_rewards.append([0.0])  # Neutral reward
            
            # Clean up memory after each candidate
            torch.cuda.empty_cache()
        
        return all_rewards
    
    def _extract_question(self, trajectory: str) -> str:
        """Extract the original question from the trajectory"""
        # Look for common patterns that indicate end of question
        end_patterns = [
            "Reasoning Steps:",
            "Solution:",
            "Answer:",
            "\n\n",
            "- Step"
        ]
        
        question = trajectory
        for pattern in end_patterns:
            if pattern in trajectory:
                parts = trajectory.split(pattern)
                if parts[0].strip():
                    question = parts[0].strip()
                    break
        
        # Remove any system prompts if present
        if "<|im_start|>" in question:
            # Extract content between user tags
            start = question.find("<|im_start|>user")
            end = question.find("<|im_end|>", start)
            if start != -1 and end != -1:
                question = question[start+len("<|im_start|>user"):end].strip()
        
        return question
    
    def _score_single_candidate(
        self, 
        question: str, 
        trajectory: str,
        candidate: str
    ) -> List[float]:
        """Score a single candidate using PRM"""
        
        # Extract claims from candidate
        try:
            candidate_tokens = self.model.tokenize([candidate])
            if candidate_tokens is None or 'input_ids' not in candidate_tokens:
                log.warning(f"Failed to tokenize candidate: {candidate[:50]}...")
                return [0.0]
                
            claims = self.steps_extractor.split_to_steps(
                candidate,
                candidate_tokens['input_ids'][0],
                self.model.tokenizer
            )
            
            if not claims:
                log.debug(f"No claims extracted from candidate: {candidate[:50]}...")
                return [0.0]
                
        except Exception as e:
            log.warning(f"Error extracting claims: {e}")
            return [0.0]
        
        # Get PRM rewards
        try:
            rewards = self._compute_prm_rewards(question, claims)
            return rewards if rewards else [0.0]
        except Exception as e:
            log.warning(f"Error computing PRM rewards: {e}")
            return [0.0]
    
    def _compute_prm_rewards(self, question: str, claims: List[Any]) -> List[float]:
        """Compute PRM rewards for claims"""
        
        if not claims:
            return []
        
        # Format conversation for PRM
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "<extra_0>".join([c.claim_text for c in claims]) + "<extra_0>"},
        ]
        
        conversation_str = self.prm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        
        input_ids = self.prm_tokenizer.encode(
            conversation_str, 
            return_tensors="pt"
        ).to(self.prm_model.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.prm_model(input_ids=input_ids)
        
        # Extract step rewards
        step_sep_id = self.prm_tokenizer.encode("<extra_0>")[0]
        token_masks = (input_ids == step_sep_id)
        
        # Compute rewards
        rewards = self._extract_step_rewards(outputs[0], token_masks)
        
        return rewards[0] if rewards else []
    
    def _extract_step_rewards(self, logits, token_masks):
        """Extract reward scores from PRM logits"""
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)
        
        all_scores = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]
            # Get positive class probabilities where mask is non-zero
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
            scores = positive_probs.cpu().tolist()
            all_scores.append(scores)
            
        return all_scores


class DirectPRMScorerOptimized(DirectPRMScorer):
    """
    Optimized version with better batching for multiple candidates.
    
    Additional optimizations:
    1. Batch multiple candidates together when possible
    2. Cache question extraction
    3. Reuse tokenization results
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.question_cache = {}
        self.max_cache_size = 100
        
    def compute_claim_rewards(
        self,
        trajectory: str,
        candidates: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Compute rewards with optimized batching"""
        
        self.prepare_model()
        
        if not candidates:
            return []
        
        # Get question (with caching)
        trajectory_hash = hash(trajectory[:200])  # Hash prefix for stability
        if trajectory_hash in self.question_cache:
            question = self.question_cache[trajectory_hash]
        else:
            question = self._extract_question(trajectory)
            # Cache with size limit
            if len(self.question_cache) >= self.max_cache_size:
                self.question_cache.pop(next(iter(self.question_cache)))
            self.question_cache[trajectory_hash] = question
        
        # Process in batches for efficiency
        all_rewards = []
        for i in range(0, len(candidates), self.batch_size):
            batch_candidates = candidates[i:i + self.batch_size]
            batch_rewards = self._score_batch(question, trajectory, batch_candidates)
            all_rewards.extend(batch_rewards)
        
        return all_rewards
    
    def _score_batch(
        self,
        question: str,
        trajectory: str,
        candidates: List[str]
    ) -> List[List[float]]:
        """Score a batch of candidates"""
        # For now, fall back to individual scoring
        # (PRM batching would require careful handling of different claim counts)
        rewards = []
        for candidate in candidates:
            rewards.append(self._score_single_candidate(question, trajectory, candidate))
        return rewards
    
    def cleanup(self):
        """Clean up resources including cache"""
        self.question_cache.clear()
        super().cleanup()