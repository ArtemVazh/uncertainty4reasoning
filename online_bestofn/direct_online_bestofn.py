"""
Direct online best-of-n implementation using direct UHead scorer
"""

import torch
import logging
from typing import List, Dict, Optional
from tqdm import tqdm

from datasets import Dataset
from lm_polygraph import WhiteboxModel

from online_bestofn.step_detection import StepBoundaryDetector
from online_bestofn.step_generation import StepCandidateGenerator
from online_bestofn.scorers.uhead_direct import DirectUHeadScorer
from utils import parse_ans

log = logging.getLogger(__name__)


class DirectOnlineBestOfN:
    """
    Simplified online best-of-n that uses UHead directly without stat calculator pipeline.
    
    Key improvements:
    1. Direct UHead application - no stat calculator overhead
    2. Cleaner separation of scoring from generation
    3. More efficient batch processing
    """
    
    def __init__(
        self,
        model: WhiteboxModel,
        uhead_path: str,
        candidates_per_step: int = 10,
        max_steps: int = 20,
        temperature: float = 0.7,
        device: str = "cuda",
        verbose: bool = True,
        generation_batch_size: int = None
    ):
        self.model = model
        self.candidates_per_step = candidates_per_step
        self.max_steps = max_steps
        self.temperature = temperature
        self.device = device
        self.verbose = verbose
        self.generation_batch_size = generation_batch_size or candidates_per_step
        
        # Initialize components
        self.detector = StepBoundaryDetector(
            step_patterns=["- Step", "<Answer>:", "\n<Answer>:"],
            answer_patterns=["<Answer>:", "\n<Answer>:"],
            max_tokens_per_step=350
        )
        
        self.step_generator = StepCandidateGenerator(
            model=model,
            detector=self.detector,
            candidates_per_step=candidates_per_step,
            temperature=temperature,
            max_new_tokens=350,
            device=device
        )
        
        self.scorer = DirectUHeadScorer(
            model=model,
            uhead_path=uhead_path,
            device=device,
            batch_size=candidates_per_step  # Process all candidates at once
        )
    
    def generate_trajectory(self, prompt: str) -> Dict[str, any]:
        """
        Generate a trajectory step-by-step using direct UHead scoring.
        
        Args:
            prompt: Initial prompt/question
            
        Returns:
            Dictionary with:
                - trajectory: Final generated trajectory
                - steps: List of selected steps
                - step_scores: Scores for each selected step
                - completed: Whether trajectory reached completion
        """
        
        trajectory = prompt
        selected_steps = []
        step_scores = []
        
        for step_num in range(self.max_steps):
            if self.verbose:
                log.info(f"\n=== Step {step_num} ===")
            
            # Generate candidates
            if self.verbose:
                log.info(f"Generating candidates with temperature={self.temperature}")
            
            # Generate candidates in batches if needed
            if self.generation_batch_size < self.candidates_per_step:
                candidates = self._generate_candidates_in_batches(trajectory)
            else:
                candidates = self.step_generator.generate_candidates(
                    trajectory, 
                    verbose=self.verbose
                )
            
            if not candidates:
                if self.verbose:
                    log.info("No candidates generated, stopping")
                break
            
            # Score candidates directly with UHead
            candidate_claim_uncertainties = self.scorer.compute_claim_uncertainties(
                trajectory,
                [c.text for c in candidates]
            )
            
            # Aggregate claim uncertainties (e.g., mean)
            candidate_scores = []
            for uncertainties in candidate_claim_uncertainties:
                if uncertainties and len(uncertainties) > 0:
                    # Handle both list and numpy array
                    if hasattr(uncertainties, 'mean'):
                        score = float(uncertainties.mean())  # numpy array
                    else:
                        score = float(sum(uncertainties) / len(uncertainties))  # list
                else:
                    score = 0.5  # Neutral
                candidate_scores.append(score)
            
            # Log all candidates
            if self.verbose:
                log.info(f"Generated {len(candidates)} candidates:")
                for i, (candidate, score) in enumerate(zip(candidates, candidate_scores)):
                    log.info(f"  [{i}] Score: {score:.3f} | Text: '{candidate.text}'")
            
            # Select best candidate (lowest uncertainty)
            best_idx = min(range(len(candidate_scores)), key=lambda i: candidate_scores[i])
            selected_candidate = candidates[best_idx]
            selected_score = candidate_scores[best_idx]
            
            if self.verbose:
                log.info(f"Selected candidate {best_idx} (score: {selected_score:.3f})")
                log.info(f"Text: {selected_candidate.text}")
            
            # Update trajectory
            trajectory += selected_candidate.text
            selected_steps.append(selected_candidate.text)
            step_scores.append(selected_score)
            
            # Check if trajectory is complete
            if selected_candidate.is_trajectory_complete:
                if self.verbose:
                    log.info("Answer pattern detected - generating final answer")
                
                # Generate final answer
                final_answer = self._generate_final_answer(trajectory)
                trajectory += final_answer
                selected_steps.append(final_answer)
                break
        
        return {
            "trajectory": trajectory,
            "steps": selected_steps,
            "step_scores": step_scores,
            "completed": len(selected_steps) > 0
        }
    
    def _generate_candidates_in_batches(self, trajectory: str) -> List:
        """Generate candidates in smaller batches to avoid OOM"""
        all_candidates = []
        
        # Calculate number of batches needed
        num_batches = (self.candidates_per_step + self.generation_batch_size - 1) // self.generation_batch_size
        
        # Temporarily store original setting
        original_candidates = self.step_generator.candidates_per_step
        
        try:
            for batch_idx in range(num_batches):
                # Calculate batch size for this iteration
                start_idx = batch_idx * self.generation_batch_size
                end_idx = min((batch_idx + 1) * self.generation_batch_size, self.candidates_per_step)
                batch_size = end_idx - start_idx
                
                if self.verbose:
                    log.info(f"Generating batch {batch_idx+1}/{num_batches} ({batch_size} candidates)")
                
                # Set batch size for this generation
                self.step_generator.candidates_per_step = batch_size
                
                # Generate batch
                batch_candidates = self.step_generator.generate_candidates(
                    trajectory, 
                    verbose=False  # Avoid too much logging
                )
                
                if batch_candidates:
                    all_candidates.extend(batch_candidates)
                    
                # Clear GPU cache after each batch
                torch.cuda.empty_cache()
                
        finally:
            # Always restore original setting
            self.step_generator.candidates_per_step = original_candidates
        
        return all_candidates
    
    def _generate_final_answer(self, trajectory: str) -> str:
        """Generate and select best final answer"""
        
        # Generate answer candidates (without step detection)
        inputs = self.model.tokenize([trajectory])
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        
        # Generate answer candidates in batches if needed
        if self.generation_batch_size < self.candidates_per_step:
            outputs = []
            num_batches = (self.candidates_per_step + self.generation_batch_size - 1) // self.generation_batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.generation_batch_size
                end_idx = min((batch_idx + 1) * self.generation_batch_size, self.candidates_per_step)
                batch_size = end_idx - start_idx
                
                with torch.no_grad():
                    batch_outputs = self.model.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=1024,
                        do_sample=True,
                        temperature=self.temperature,
                        num_return_sequences=batch_size,
                        pad_token_id=self.model.tokenizer.eos_token_id,
                        eos_token_id=self.model.tokenizer.eos_token_id
                    )
                    outputs.extend(batch_outputs)
                    
                # Clear GPU cache after each batch
                torch.cuda.empty_cache()
        else:
            with torch.no_grad():
                # Use the underlying model directly to avoid WhiteboxModel wrapper issues
                outputs = self.model.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=self.temperature,
                    num_return_sequences=self.candidates_per_step,
                    pad_token_id=self.model.tokenizer.eos_token_id,
                    eos_token_id=self.model.tokenizer.eos_token_id
                )
        
        # Extract answer candidates
        answer_candidates = []
        for seq in outputs:
            new_tokens = seq[input_ids.shape[1]:]
            answer_text = self.model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            answer_candidates.append(answer_text)
        
        # Score answer candidates
        answer_uncertainties = self.scorer.compute_claim_uncertainties(
            trajectory,
            answer_candidates
        )
        
        # Aggregate scores
        answer_scores = []
        for uncertainties in answer_uncertainties:
            if uncertainties and len(uncertainties) > 0:
                # Handle both list and numpy array
                if hasattr(uncertainties, 'mean'):
                    score = float(uncertainties.mean())  # numpy array
                else:
                    score = float(sum(uncertainties) / len(uncertainties))  # list
            else:
                score = 0.5
            answer_scores.append(score)
        
        # Select best answer
        best_idx = min(range(len(answer_scores)), key=lambda i: answer_scores[i])
        
        if self.verbose:
            log.info(f"Generated {len(answer_candidates)} answer candidates")
            log.info(f"Selected answer {best_idx} (score: {answer_scores[best_idx]:.3f})")
        
        return answer_candidates[best_idx]
    
    def cleanup(self):
        """Clean up resources"""
        self.scorer.cleanup()


def run_direct_online_bestofn(
    dataset: Dataset,
    model: WhiteboxModel,
    uhead_path: str,
    save_path: str,
    n: int = 10,
    max_new_tokens: int = 250,
    subset: Optional[int] = None,
    verbose: bool = True
):
    """
    Run direct online best-of-n evaluation on a dataset.
    
    Args:
        dataset: Evaluation dataset
        model: Language model
        uhead_path: Path to UHead model
        save_path: Path to save results
        n: Number of candidates per step
        max_new_tokens: Max tokens per step
        subset: Evaluate only first N samples
        verbose: Enable verbose logging
    """
    
    if subset:
        dataset = dataset.select(range(min(subset, len(dataset))))
        log.info(f"Using subset of {len(dataset)} samples")
    
    # Initialize generator
    generator = DirectOnlineBestOfN(
        model=model,
        uhead_path=uhead_path,
        candidates_per_step=n,
        temperature=0.7,
        device=str(model.device()),
        verbose=verbose
    )
    
    results = []
    
    try:
        for i, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            log.info(f"\n{'='*60}")
            log.info(f"Sample {i+1}/{len(dataset)}")
            log.info(f"Question: {sample['question']}")
            log.info(f"Gold Answer: {sample['answer']}")
            
            # Generate trajectory
            result = generator.generate_trajectory(sample["question"])
            
            # Extract answer from trajectory
            generated_text = result["trajectory"]
            if "trajectory" in result:
                generated_text = result["trajectory"].replace(sample["question"], "").strip()
            
            # Check correctness
            is_correct = _is_correct_answer(generated_text, sample["answer"])
            
            # Store result
            results.append({
                "question": sample["question"],
                "gold_answer": sample["answer"],
                "generated_trajectory": result["trajectory"],
                "generated_answer": generated_text,
                "steps": result["steps"],
                "step_scores": result["step_scores"],
                "is_correct": is_correct,
                "completed": result["completed"]
            })
            
            log.info(f"Generated: {generated_text}...")
            log.info(f"Gold: {sample['answer']}")
            log.info(f"Generated answer: {parse_ans(generated_text)}")
            log.info(f"Correct: {is_correct}")
            # import pdb; pdb.set_trace()
            
            # Save periodically
            if (i + 1) % 10 == 0:
                torch.save(results, save_path)
                log.info(f"Saved {len(results)} results to {save_path}")
    
    finally:
        # Final save
        torch.save(results, save_path)
        log.info(f"Final save: {len(results)} results to {save_path}")
        
        # Cleanup
        generator.cleanup()
    
    # Print summary
    correct = sum(r["is_correct"] for r in results)
    log.info(f"\nAccuracy: {correct}/{len(results)} = {correct/len(results):.2%}")
    
    return results


def _is_correct_answer(generated: str, gold: str) -> bool:
    """Check if generated answer matches gold answer"""
    try:
        pred = parse_ans(generated)
        gold_parsed = parse_ans(gold)
        
        if pred is None or gold_parsed is None:
            return False
            
        return pred == gold_parsed
    except:
        return False


if __name__ == "__main__":
    # Example usage
    import argparse
    from datasets import load_dataset
    from configs.load_qwen import load_model as load_qwen_model, load_tokenizer as load_qwen_tokenizer
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--uhead-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Load model
    tokenizer = load_qwen_tokenizer(args.model_path)
    base_model = load_qwen_model(args.model_path, args.device)
    model = WhiteboxModel(base_model, tokenizer)
    
    # Load dataset
    dataset = load_dataset(args.dataset_path, split="test")
    
    # Run evaluation
    run_direct_online_bestofn(
        dataset=dataset,
        model=model,
        uhead_path=args.uhead_path,
        save_path=args.save_path,
        n=args.n,
        subset=args.subset,
        verbose=True
    )