
from concurrent.futures import ThreadPoolExecutor, as_completed
from parse import parse
from openai import OpenAI
import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
import diskcache as dc
import threading

from lm_polygraph.generation_metrics.generation_metric import GenerationMetric
from lm_polygraph.stat_calculators.extract_claims import Claim
from synthetic_dataset_generation.utils.chat import DeepSeekChat, OpenAIChat, LocalChat

log = logging.getLogger()


STEP_CHECKING_SYSTEM_PROMPT = """
You are an expert evaluator who assesses the correctness of individual reasoning steps made by a student while solving a problem.
""".strip(" \n")

STEP_CHECKING_PROMPT = """
You are given a problem, its ground-truth solution, and a student's (incomplete) reasoning process so far.
Your task is to determine whether the **latest step** in the student's reasoning is correct.

Instructions:
- A step is **wrong** if it contains explicit logical or computational errors, or if it contradicts any previous steps.
- Redundant, unnecessary, or non-informative steps are **not** considered wrong.
- If the latest step is correct, output **1**. If it is wrong, output **0**.
- Respond with the number (0 or 1) **only**, with no extra text or explanation.

PROBLEM:
{problem}

GROUND-TRUTH SOLUTION:
{answer}

STUDENT'S REASONING PROCESS SO FAR:
{steps}

THE LATEST STEP TO EVALUATE:
{latest_step}
""".strip(" \n")


class StepFactCheckThinking(GenerationMetric):
    def __init__(
            self,
            prompt_file: str,
            cache_path: str = "~/.cache",
            model: str = None,
            api_key: str = None,
            progress_bar: bool = True,
            n_threads: int = 1,
            wait_times: tuple = (5, 10, 30, 60, 120),
            base_url: str = None,
            debug: bool = False,
    ):
        super().__init__(["input_texts", "claims"], "claim")

        with open(prompt_file, 'r') as f:
            self.prompt = f.read()

        if base_url is not None and 'localhost' in base_url:
            print(f"Using Local {model}")
            self.chat = LocalChat(model=model, base_url=base_url, cache_path=cache_path, system_prompt=STEP_CHECKING_SYSTEM_PROMPT)
        else:
            print(f"Using Remote {model}")
            if 'deepseek' in model:
                self.chat = DeepSeekChat(cache_path,  model=model, api_key=api_key, wait_times=wait_times)
            else:
                self.chat = OpenAIChat(model, cache_path=cache_path, system_prompt=STEP_CHECKING_SYSTEM_PROMPT)
        self.progress_bar = progress_bar
        self.n_threads = n_threads
        self.debug = debug

    def __str__(self):
        return "StepFactCheckThinking"

    def parse_problem(self, input_text: str):
        try:
            return parse(self.prompt, input_text).named['q']
        except Exception:
            # For run_extract_verify_claims.py, input texts are raw questions without prompt
            return input_text

    def format_prompt(self, input_text: str, claims: list[Claim], answer: str) -> str:
        problem = self.parse_problem(input_text)
        steps = ' '.join([cl.sentence for i, cl in enumerate(claims)])
        last_claim = claims[-1].claim_text
        return STEP_CHECKING_PROMPT.format(problem=problem, answer=answer, steps=steps, latest_step=last_claim)

    def parse_reply(self, reply: str, reasoning_content: str) -> tuple[float | None, str]:
        if '1' in reply:
            return 1.0, reasoning_content
        elif '0' in reply:
            return 0.0, reasoning_content
        else:
            return np.nan, reasoning_content

    def _score_single(self, args: tuple[list, str, str, int]) -> tuple[float, str]:
        claims, input_text, answer, step_idx = args
        claims_so_far = claims[:step_idx+1]
        prompt = self.format_prompt(input_text, claims_so_far, answer)
        response_object = self.chat.ask(prompt, json_output=False)
        if isinstance(response_object, tuple):
            reply, reasoning_content = response_object
        else:
            reply, reasoning_content = response_object, ""
            
        parsed_reply, parsed_reason = self.parse_reply(reply, reasoning_content)
        claim = claims[step_idx]
        if len(claim.aligned_token_ids) == 0 or np.isnan(parsed_reply):
            return np.nan, parsed_reason
        # Map: model 0 (wrong) -> 1 (error), model 1 (correct) -> 0 (no error)
        label = 1.0 if parsed_reply == 0.0 else 0.0
        return label, parsed_reason

    def __call__(
            self,
            stats: Dict[str, np.ndarray],
            target_texts: List[str],
            output_path: str,
    ) -> list:

        input_texts = stats["input_texts"]

        if "answers" in stats.keys():
            target_texts = stats["answers"]

        print(f"Using {self.n_threads} threads")

        claim_sets = stats["claims"]
        claim_labels: list[list[float]] = [
            [np.nan for _ in range(len(claims))]
            for claims in claim_sets
        ]
        verdict_reasons: list[list[str]] = [
            ["" for _ in range(len(claims))]
            for claims in claim_sets
        ]

        tasks = []
        for sample_idx, (input_text, claims, answer) in enumerate(zip(input_texts, claim_sets, target_texts)):
            for step_idx in range(len(claims)):
                tasks.append((sample_idx, claims, input_text, answer, step_idx))

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = {
                executor.submit(self._score_single, (claims, input_text, answer, step_idx)): (sample_idx, step_idx)
                for sample_idx, claims, input_text, answer, step_idx in tasks
            }
            for future in tqdm(as_completed(futures), desc="Verifying claims", total=len(futures), disable=not self.progress_bar):
                sample_idx, step_idx = futures[future]
                label, reason = future.result()
                claim_labels[sample_idx][step_idx] = label
                verdict_reasons[sample_idx][step_idx] = reason

        claim_sentences = [
            [cl.sentence for cl in claims]
            for claims in claim_sets
        ]

        print(len(input_texts))
        print(len(claim_labels))
        print(len(verdict_reasons))
        print(len(claim_sentences))

        if output_path is not None and len(input_texts) == len(claim_labels) == len(verdict_reasons) == len(claim_sentences):
            new_df = {
                "input_text": input_texts,
                "claims": claim_sentences,
                "claim_labels": claim_labels,
                "verdict_reasons": verdict_reasons,
            }

            if "answers" in stats.keys():
                new_df["answer"] = target_texts
                print(len(target_texts))
            df = pd.DataFrame(new_df)
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            df.to_csv(output_path, index=False)

        return claim_labels

