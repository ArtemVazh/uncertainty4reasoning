from baselines.reasoneval import ReasonEvalStatCalculator
from synthetic_dataset_generation.utils.steps_extractor import Claim
from bestofn_stepwise.stat_calculators.stepwise_minimization import StepwiseUncertaintyMinimizationCalculatorBase
from lm_polygraph import WhiteboxModel
from parse import parse


class StepwiseReasonEvalMinimizationCalculator(StepwiseUncertaintyMinimizationCalculatorBase):
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
            "reasoneval", max_tokens, max_steps,
            candidates_per_step, temperature, top_p, top_k, device,
        )
        self.prompt = open(prompt_path, 'r').read()
        self.reasoneval = ReasonEvalStatCalculator()

    @staticmethod
    def meta_info() -> tuple[list[str], list[str]]:
        return [
            "min_reasoneval_final_texts",
            "min_reasoneval_all_steps",
            "min_reasoneval_all_uncertainties"
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
        scores: list[float] = []
        for step in step_candidates:
            x: dict[str, float] = self.reasoneval.get_step_level_scores(q, prev_steps + [step])[-1]
            scores.append(x['redundancy'] - x['validity'])
        return scores
