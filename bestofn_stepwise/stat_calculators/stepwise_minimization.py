import torch
from tqdm import trange
from typing import List, Dict
from transformers import StoppingCriteria, StoppingCriteriaList

from lm_polygraph import WhiteboxModel
from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from synthetic_dataset_generation.utils.steps_extractor import StepsExtractor, Claim


class StopOnNewline(StoppingCriteria):
    def __init__(self, tokenizer, start_length):
        self.tokenizer = tokenizer
        self.start_length = start_length

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
        generated_ids = input_ids[0][self.start_length:]
        decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return any(x in decoded for x in ["\n- Step ", "\n<Answer>: "])


class StepwiseUncertaintyMinimizationCalculatorBase(StatCalculator):
    def __init__(
            self,
            stats_name: str,
            max_tokens: int = 256,
            max_steps: int = 20,
            candidates_per_step: int = 10,
            temperature: float = 1.0,
            top_p: float = 0.95,
            top_k: int = 50,
            device: str = "cuda",
    ):
        super().__init__()
        self.stats_name = stats_name
        self.steps_extractor = StepsExtractor()
        self.max_tokens = max_tokens
        self.max_steps = max_steps
        self.candidates_per_step = candidates_per_step
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.device = device

    def prepare_claims(self, claims, input_len, full_len):
        all_claim_tensors = []
        for claim in claims:
            mask = torch.zeros((1, full_len), dtype=int)
            mask[0, (input_len + torch.as_tensor(claim.aligned_token_ids)).int()] = 1
            all_claim_tensors.append(mask[:, 1:])  # ignoring <s>
        return all_claim_tensors

    def generate_step_candidates(self, model: WhiteboxModel, prompt_tokens: list[int], input_tokens: list[int]):
        llm_inputs = {
            'input_ids': torch.LongTensor([prompt_tokens]).to(self.device),
            'attention_mask': torch.ones(1, len(prompt_tokens)).bool().to(self.device),
        }
        start_len = llm_inputs["input_ids"].shape[-1]
        stopping_criteria = StoppingCriteriaList([StopOnNewline(model.tokenizer, start_len)])

        llm_outputs = model.generate(
            **llm_inputs,
            max_new_tokens=self.max_tokens - (len(prompt_tokens) - len(input_tokens)),
            do_sample=True,
            top_p=self.top_p,
            top_k=self.top_k,
            temperature=self.temperature,
            num_return_sequences=self.candidates_per_step,
            eos_token_id=model.tokenizer.eos_token_id,
            pad_token_id=model.tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            output_scores=True,
            return_dict_in_generate=True,
            output_attentions=True,
        )

        return llm_inputs, llm_outputs

    def extract_first_step(self, model: WhiteboxModel, new_tokens: torch.Tensor):
        new_text = model.tokenizer.decode(new_tokens, skip_special_tokens=True)
        steps = self.steps_extractor.split_to_steps(new_text, new_tokens, model.tokenizer)
        if len(steps) == 0:
            return Claim("", "", [])
        return steps[0]

    def compute_uncertainty(
            self,
            llm_inputs,
            llm_outputs,
            input_text: str,
            prev_steps: list[Claim],
            step_candidates: list[Claim],
            model: WhiteboxModel,
    ) -> list[float]:
        raise NotImplementedError()

    def __call__(
            self,
            dependencies: Dict[str, object],
            texts: List[str],
            model: WhiteboxModel,
            **kwargs,
    ) -> Dict[str, List]:
        results = []

        for prompt in texts:
            all_steps = []
            all_best_steps = []
            all_uncertainties = []
            input_text = prompt
            input_tokens = model.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()

            for _ in trange(self.max_steps, desc="Step-by-step generation"):
                prompt_tokens = model.tokenizer(prompt, return_tensors="pt")["input_ids"][0].tolist()
                if len(input_tokens) + self.max_tokens <= len(prompt_tokens):
                    break
                print('Generating from: "{}"'.format(model.tokenizer.decode(prompt_tokens).split('</think>\n\n')[-1]))
                try:
                    inputs, outputs = self.generate_step_candidates(model, prompt_tokens, input_tokens)
                except torch.OutOfMemoryError as e:
                    print("CUDA out of memory error caught! Will skip next steps")
                    print(e)
                    torch.cuda.empty_cache()
                    break
                step_candidates: list[Claim] = []
                for o in outputs.sequences[:, inputs['input_ids'].shape[-1]:]:
                    step_candidates.append(self.extract_first_step(model, o))
                print('steps:')
                for t, s in zip(outputs.sequences, step_candidates):
                    print(f'"{s.claim_text}"')
                step_uncertainties = self.compute_uncertainty(
                    inputs, outputs,
                    input_text, all_best_steps,
                    step_candidates, model,
                )
                print('step_uncertainties:', step_uncertainties)
                best_step_idx = torch.argmin(torch.tensor(step_uncertainties)).item()

                prompt += step_candidates[best_step_idx].claim_text + '\n'
                all_steps.append(step_candidates)
                all_best_steps.append(step_candidates[best_step_idx])
                all_uncertainties.append(step_uncertainties)
                if '<Answer>:' in step_candidates[best_step_idx].claim_text:
                    break

            results.append({
                f"min_{self.stats_name}_final_texts": [prompt],
                f"min_{self.stats_name}_all_steps": all_steps,
                f"min_{self.stats_name}_all_uncertainties": all_uncertainties,
            })

        return {
            f"min_{self.stats_name}_final_texts": [r[f"min_{self.stats_name}_final_texts"][0] for r in results],
            f"min_{self.stats_name}_all_steps": [r[f"min_{self.stats_name}_all_steps"] for r in results],
            f"min_{self.stats_name}_all_uncertainties": [r[f"min_{self.stats_name}_all_uncertainties"] for r in results],
        }
