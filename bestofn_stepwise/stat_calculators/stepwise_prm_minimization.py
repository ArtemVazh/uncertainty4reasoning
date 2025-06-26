from baselines.prm import PRMStatCalculator
from synthetic_dataset_generation.utils.steps_extractor import Claim
from bestofn_stepwise.stat_calculators.stepwise_minimization import StepwiseUncertaintyMinimizationCalculatorBase
from lm_polygraph import WhiteboxModel
from parse import parse


class StepwisePRMMinimizationCalculator(StepwiseUncertaintyMinimizationCalculatorBase):
    def __init__(
            self,
            prompt_path: str,
            max_tokens: int = 256,
            max_steps: int = 20,
            candidates_per_step: int = 10,
            temperature: float = 1.0,
            top_p: float = 0.95,
            top_k: int = 50,
            device: str = "cuda",
    ):
        super().__init__(
            "prm", max_tokens, max_steps,
            candidates_per_step, temperature, top_p, top_k, device,
        )
        self.prompt = open(prompt_path, 'r').read()
        self.prm = PRMStatCalculator()

    @staticmethod
    def meta_info() -> tuple[list[str], list[str]]:
        return [
            "min_prm_final_texts",
            "min_prm_all_steps",
            "min_prm_all_uncertainties"
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
        q = parse(self.prompt, input_text).named['q']
        rewards = []
        for step in step_candidates:
            rewards.append(1 - self.prm.get_rewards(question=q, steps=prev_steps + [step])[-1])
        return rewards
