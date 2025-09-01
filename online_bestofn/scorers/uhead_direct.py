"""
Direct UHead scorer that bypasses the stat calculator pipeline for efficient stepwise scoring
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.special import expit
import logging

from lm_polygraph import WhiteboxModel
from luh import AutoUncertaintyHead
from synthetic_dataset_generation.utils.steps_extractor import StepsExtractor
from .base import UncertaintyBasedScorer

log = logging.getLogger(__name__)


class DirectUHeadScorer(UncertaintyBasedScorer):
    """
    Direct UHead scorer that applies uncertainty head without stat calculator pipeline.
    
    This implementation:
    1. Tokenizes and processes candidates in batch
    2. Extracts UHead features directly
    3. Applies uncertainty head to claims
    4. Returns uncertainty scores
    
    Much cleaner and more efficient than going through the full pipeline.
    """
    
    def __init__(
        self,
        model: WhiteboxModel,
        uhead_path: str,
        device: str = "cuda",
        batch_size: int = 8,
        feature_batch_size: int = 1
    ):
        super().__init__("DirectUHead")
        self.model = model
        self.uhead_path = uhead_path
        self.device = device
        self.batch_size = batch_size
        self.feature_batch_size = feature_batch_size  # For memory-efficient feature extraction
        self.uhead = None
        self.steps_extractor = StepsExtractor(progress_bar=False)
        
    def prepare_model(self):
        """Load uncertainty head model"""
        if self.uhead is None:
            log.info(f"Loading UHead from {self.uhead_path}")
            self.uhead = AutoUncertaintyHead.from_pretrained(
                self.uhead_path,
                self.model.model
            )
            self.uhead.to(self.device)
            self.uhead.eval()
            
    def cleanup(self):
        """Free UHead model memory"""
        # import pdb; pdb.set_trace()
        if self.uhead is not None:
            del self.uhead
            self.uhead = None
            torch.cuda.empty_cache()
    
    def compute_claim_uncertainties(
        self,
        trajectory: str,
        candidates: List[str],
        **kwargs
    ) -> List[List[float]]:
        """
        Compute uncertainty scores for claims in each candidate.
        
        Args:
            trajectory: Current trajectory text
            candidates: List of candidate next steps
            
        Returns:
            List of claim uncertainty lists (one per candidate)
        """
        self.prepare_model()
        
        if not candidates:
            return []
        
        # Process candidates in batches for efficiency
        all_uncertainties = []
        # import pdb; pdb.set_trace()
        for i in range(0, len(candidates), self.batch_size):
            batch_candidates = candidates[i:i + self.batch_size]
            batch_uncertainties = self._score_batch(trajectory, batch_candidates)
            all_uncertainties.extend(batch_uncertainties)
            
            # Clean up GPU memory after each batch
            torch.cuda.empty_cache()
            
        return all_uncertainties
    
    def _score_batch(
        self, 
        trajectory: str, 
        candidates: List[str]
    ) -> List[List[float]]:
        """Score a batch of candidates efficiently"""
        
        # Step 1: Prepare full texts
        full_texts = [trajectory + candidate for candidate in candidates]
        
        # Step 1.5: Get trajectory length for context
        trajectory_tokens = self.model.tokenize([trajectory])
        trajectory_length = trajectory_tokens['input_ids'].shape[1]
        
        # Step 2: Tokenize batch
        try:
            inputs = self.model.tokenize(full_texts)
            input_ids = inputs['input_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)
            batch_size, seq_len = input_ids.shape
        except Exception as e:
            log.error(f"Tokenization failed: {e}")
            return [[0.5] for _ in candidates]
        
        # Step 3: Extract UHead features with single forward pass
        try:
            features, generated_seq_length = self._extract_features_batch(input_ids, attention_mask)
            if features is None:
                log.error("Features are None")
                return [[0.5] for _ in candidates]
        except Exception as e:
            log.error(f"Feature extraction failed: {e}")
            return [[0.5] for _ in candidates]
        
        # Step 4: Process each candidate's claims
        batch_uncertainties = []
        
        for idx, candidate in enumerate(candidates):
            try:
                # Extract claims from candidate text
                claims = self._extract_claims(candidate)
                
                if not claims:
                    log.error(f"No claims found for candidate: {candidate}")
                    batch_uncertainties.append([0.5])
                    continue
                
                # Score claims using UHead
                claim_uncertainties = self._score_claims(
                    features[idx:idx+1],  # Single sample features
                    claims,
                    input_ids[idx:idx+1],
                    attention_mask[idx:idx+1],
                    generated_seq_length,
                    trajectory_length  # Pass context length
                )
                
                batch_uncertainties.append(claim_uncertainties)
                
            except Exception as e:
                log.warning(f"Failed to score candidate {idx}: {e}")
                batch_uncertainties.append([0.5])
        
        return batch_uncertainties
    
    def _extract_features_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], int]:
        """Extract UHead features for a batch of inputs with internal batching for memory efficiency"""
        
        batch_size = input_ids.shape[0]
        
        # If batch is small enough, process all at once
        if batch_size <= self.feature_batch_size:
            return self._extract_features_single_batch(input_ids, attention_mask)
        
        # Otherwise, process in smaller chunks
        all_features = []
        generated_length = None
        
        for i in range(0, batch_size, self.feature_batch_size):
            end_idx = min(i + self.feature_batch_size, batch_size)
            
            # Extract sub-batch
            sub_input_ids = input_ids[i:end_idx]
            sub_attention_mask = attention_mask[i:end_idx]
            
            # Process sub-batch
            sub_features, sub_generated_length = self._extract_features_single_batch(
                sub_input_ids, sub_attention_mask
            )
            
            all_features.append(sub_features)
            
            # Verify consistent generation length
            if generated_length is None:
                generated_length = sub_generated_length
            elif generated_length != sub_generated_length:
                log.warning(f"Inconsistent generation lengths: {generated_length} vs {sub_generated_length}")
            
            # Clean up after each sub-batch
            torch.cuda.empty_cache()
        
        # Concatenate all features
        features = torch.cat(all_features, dim=0)
        
        return features, generated_length
    
    def _extract_features_single_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], int]:
        """Extract features for a single batch (original implementation)"""
        
        batch_size = input_ids.shape[0]
        
        # Create batch dict
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'context_lengths': torch.full((batch_size,), input_ids.shape[1], device=self.device)
        }
        
        # Generate minimal output to get proper attention structure
        with torch.no_grad():
            generation_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                output_attentions=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                do_sample=False,
                pad_token_id=self.model.tokenizer.eos_token_id
            )
        
        # Set up outputs for feature extraction
        generated_length = generation_outputs.sequences.shape[1]
        generation_outputs.full_attention_mask = torch.ones(
            (batch_size, generated_length), 
            device=self.device
        )
        generation_outputs.context_lengths = batch['context_lengths']
        
        # Extract features
        with torch.no_grad():
            features = self.uhead.feature_extractor(batch, generation_outputs)
        
        # Clean up memory immediately
        del generation_outputs
        torch.cuda.empty_cache()
            
        return features, generated_length
    
    def _extract_claims(self, candidate_text: str) -> List[Any]:
        """Extract claims from candidate text"""
        
        # Tokenize candidate
        candidate_tokens = self.model.tokenize([candidate_text])
        if candidate_tokens is None or 'input_ids' not in candidate_tokens:
            return []
        
        candidate_token_ids = candidate_tokens['input_ids'][0]
        
        # Extract claims
        claims = self.steps_extractor.split_to_steps(
            candidate_text,
            candidate_token_ids,
            self.model.tokenizer
        )
        
        return claims if claims else []
    
    def _score_claims(
        self,
        features: torch.Tensor,
        claims: List[Any],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        generated_seq_length: int,
        context_length: int
    ) -> List[float]:
        """Score claims using UHead"""
        
        if not claims:
            return [0.5]
        
        # Prepare claim masks using generated sequence length and context
        claim_masks = self._prepare_claim_masks(claims, generated_seq_length, context_length)
        
        if not claim_masks:
            return [0.5]
        
        # Create batch with claims
        batch_with_claims = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'context_lengths': torch.tensor([input_ids.shape[1]], device=self.device),  # Original input length
            'claims': claim_masks
        }
        
        # Apply UHead
        with torch.no_grad():
            feature_seq_len = features.shape[1]
            generated_attention_mask = torch.ones((features.shape[0], feature_seq_len), device=self.device)
            
            uncertainty_logits = self.uhead._compute_tensors(
                batch_with_claims,
                features,
                generated_attention_mask
            )
        
        # Convert logits to probabilities
        uncertainties = []
        for claim_logits in uncertainty_logits.cpu().numpy():
            valid_logits = [l for l in claim_logits if l != -100]
            if valid_logits:
                uncertainties.extend([expit(l) for l in valid_logits])
        
        return uncertainties if uncertainties else [0.5]
    
    def _prepare_claim_masks(
        self, 
        claims: List[Any], 
        seq_len: int,
        context_length: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Prepare claim masks for UHead"""
        
        claim_tensors = []
        
        # If no context length provided, we need to figure it out
        if context_length is None:
            context_length = seq_len - 100  

        for claim in claims:
            if claim is None or not hasattr(claim, 'aligned_token_ids'):
                continue
                
            mask = torch.zeros(seq_len, dtype=int, device=self.device)
            
            try:
                # Map claim tokens to positions in full sequence
                # Claims are from candidate text, so offset by context length
                token_positions = context_length + torch.as_tensor(claim.aligned_token_ids).to(self.device)
                # import pdb; pdb.set_trace()
                # Ensure positions are within bounds
                valid_positions = token_positions[token_positions < seq_len]
                if len(valid_positions) > 0:
                    mask[valid_positions.long()] = 1
                    
            except Exception as e:
                log.warning(f"Error preparing claim mask: {e}")
                continue
                
            claim_tensors.append(mask[1:])  # Skip BOS token
        
        if claim_tensors:
            return [torch.stack(claim_tensors)]
        else:
            return []


class DirectUHeadScorerOptimized(DirectUHeadScorer):
    """
    Optimized version with caching and better batching.
    
    Additional optimizations:
    1. Cache trajectory features across multiple scoring calls
    2. Process all candidates in single batch when possible
    3. Reuse tokenization results
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_cache = {}
        self.max_cache_size = 100
        
    def compute_claim_uncertainties(
        self,
        trajectory: str,
        candidates: List[str],
        **kwargs
    ) -> List[List[float]]:
        """Compute uncertainties with caching"""
        
        self.prepare_model()
        
        if not candidates:
            return []
        
        # Try to score all candidates in one batch if possible
        if len(candidates) <= self.batch_size:
            return self._score_batch_optimized(trajectory, candidates)
        
        # Otherwise fall back to chunked processing
        return super().compute_claim_uncertainties(trajectory, candidates, **kwargs)
    
    def _score_batch_optimized(
        self,
        trajectory: str,
        candidates: List[str]
    ) -> List[List[float]]:
        """Optimized batch scoring with trajectory feature caching"""
        
        # Check cache for trajectory features
        trajectory_hash = hash(trajectory)
        
        if trajectory_hash in self.feature_cache:
            trajectory_features, trajectory_len = self.feature_cache[trajectory_hash]
            log.debug(f"Using cached features for trajectory (hash: {trajectory_hash})")
        else:
            # Extract trajectory features once
            trajectory_features, trajectory_len = self._extract_trajectory_features(trajectory)
            
            # Cache with size limit
            if len(self.feature_cache) >= self.max_cache_size:
                # Remove oldest entry
                self.feature_cache.pop(next(iter(self.feature_cache)))
            
            self.feature_cache[trajectory_hash] = (trajectory_features, trajectory_len)
        
        # Score all candidates using cached trajectory features
        return self._score_with_cached_features(
            trajectory,
            candidates,
            trajectory_features,
            trajectory_len
        )
    
    def _extract_trajectory_features(
        self,
        trajectory: str
    ) -> Tuple[torch.Tensor, int]:
        """Extract and cache trajectory features"""
        
        inputs = self.model.tokenize([trajectory])
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Extract features for trajectory
        features = self._extract_features_batch(input_ids, attention_mask)
        
        return features, input_ids.shape[1]
    
    def _score_with_cached_features(
        self,
        trajectory: str,
        candidates: List[str],
        trajectory_features: torch.Tensor,
        trajectory_len: int
    ) -> List[List[float]]:
        """Score candidates using cached trajectory features"""
        
        # This is a simplified version - in practice you'd need to handle
        # the feature combination more carefully
        uncertainties = []
        
        for candidate in candidates:
            claims = self._extract_claims(candidate)
            
            if not claims:
                uncertainties.append([0.5])
                continue
            
            # Score claims (simplified - would need proper implementation)
            claim_scores = [0.5] * len(claims)  # Placeholder
            uncertainties.append(claim_scores)
        
        return uncertainties
    
    def cleanup(self):
        """Clean up resources including cache"""
        self.feature_cache.clear()
        super().cleanup()