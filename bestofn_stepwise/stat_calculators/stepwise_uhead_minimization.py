import torch
from synthetic_dataset_generation.utils.steps_extractor import Claim
from scipy.special import expit
from bestofn_stepwise.stat_calculators.stepwise_minimization import StepwiseUncertaintyMinimizationCalculatorBase
from lm_polygraph import WhiteboxModel


class StepwiseUheadMinimizationCalculator(StepwiseUncertaintyMinimizationCalculatorBase):
    def __init__(
            self,
            uncertainty_head,
            max_tokens: int = 256,
            max_steps: int = 20,
            candidates_per_step: int = 10,
            temperature: float = 1.0,
            top_p: float = 0.95,
            top_k: int = 50,
            device: str = "cuda",
    ):
        super().__init__(
            "uhead", max_tokens, max_steps,
            candidates_per_step, temperature, top_p, top_k, device,
        )
        self.uncertainty_head = uncertainty_head.to(device)

    @staticmethod
    def meta_info() -> tuple[list[str], list[str]]:
        return [
            "min_uhead_final_texts",
            "min_uhead_all_steps",
            "min_uhead_all_uncertainties"
        ], []

    def compute_uncertainty(
            self,
            llm_inputs,
            llm_outputs,
            input_text: str,
            prev_steps: list[Claim],
            step_candidates: list[Claim],
            model: WhiteboxModel,
    ) -> list[float]:
        # SETUP STATS
        n = len(llm_outputs.sequences)
        llm_inputs = {k: v.repeat(n, 1) for k, v in llm_inputs.items()}
        inp, gen = llm_inputs['input_ids'], llm_outputs.sequences
        # 1. context lengths
        llm_outputs.context_lengths = [inp.shape[-1] for _ in range(n)]
        # 2. full attention mask
        full_attn_mask = torch.zeros_like(gen).bool()
        for i in range(inp.shape[0]):
            idx = inp.shape[-1]
            full_attn_mask[i, :idx] = inp[i]
            length = next(
                iter([j for j in range(gen.shape[-1]) if gen[i, j] == model.tokenizer.eos_token_id]),
                full_attn_mask.shape[-1] - idx,
            )
            full_attn_mask[i][idx: idx + length] = 1
        llm_outputs['full_attention_mask'] = full_attn_mask
        # 3. claims
        llm_inputs['claims'] = [x.to(gen) for x in self.prepare_claims(step_candidates, inp.shape[0], gen.shape[-1])]

        with torch.no_grad():
            uncertainty = self.uncertainty_head(llm_inputs, llm_outputs)
        return [expit(x.item()) for x in uncertainty]
