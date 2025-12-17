from concurrent.futures import ThreadPoolExecutor, as_completed
from parse import parse
from openai import OpenAI
import os
import logging
import diskcache as dc
import threading
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List

from lm_polygraph.generation_metrics.generation_metric import GenerationMetric
from lm_polygraph.stat_calculators.extract_claims import Claim
from synthetic_dataset_generation.utils.chat import DeepSeekChat, LocalChat, OpenAIChat

log = logging.getLogger()


STEP_FACTUALITY_SYSTEM_PROMPT = """
You are an expert in fact-checking.
""".strip(" \n")


class StepFactCheckFactuality(GenerationMetric):
    """
    Uses an LLM to judge factuality of each step incrementally.
    Prompts the model to output 1 (factually OK) or 0 (contains factual error) for the latest sentence.
    """
    def __init__(
        self,
        prompt_file: str,
        fact_check_prompt: str,
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
            self.prompt_template = f.read()

        with open(fact_check_prompt, 'r') as f:
            self.fact_check_prompt = f.read()

        if base_url is not None and 'localhost' in base_url:
            print(f"Using Local {model}")
            self.chat = LocalChat(
                model=model,
                base_url=base_url,
                cache_path=cache_path,
                system_prompt=STEP_FACTUALITY_SYSTEM_PROMPT,
            )
        else:
            print(f"Using Remote {model}")
            if model and 'deepseek' in model:
                self.chat = DeepSeekChat(cache_path, model=model, api_key=api_key, wait_times=wait_times)
            else:
                self.chat = OpenAIChat(model, cache_path=cache_path, system_prompt=STEP_FACTUALITY_SYSTEM_PROMPT)

        self.progress_bar = progress_bar
        self.n_threads = n_threads
        self.debug = debug

    def __str__(self):
        return "StepFactCheckFactuality"

    def parse_problem(self, input_text: str):
        try:
            return parse(self.prompt_template, input_text).named['q']
        except Exception:
            # When raw questions are provided without wrapping prompt
            return input_text

    def format_prompt(self, input_text: str, claims: list[Claim]) -> str:
        question = self.parse_problem(input_text)
        # prior sentences (excluding the latest)
        prior_sentences = [cl.sentence for cl in claims]
        steps = ' '.join(s for s in prior_sentences if s is not None)
        latest_sentence = claims[-1].sentence if claims and claims[-1].sentence is not None else claims[-1].claim_text
        return self.fact_check_prompt.format(question=question, steps=steps, latest_step=latest_sentence)

    def parse_reply(self, reply: str) -> tuple[float | None, str]:
        if reply is None:
            return np.nan, ""

        txt = reply.strip()
        if not txt:
            return np.nan, ""

        # Remove code fences if the model wrapped the response
        if txt.startswith("```") and txt.endswith("```"):
            txt = txt.strip("`").strip()

        result_value = None
        reason_value = ""
        for line in txt.splitlines():
            cleaned = line.strip()
            if cleaned.lower().startswith("[reason]"):
                _, _, after = cleaned.partition(":")
                reason_value = after.strip()
                continue
            if cleaned.lower().startswith("[result]"):
                _, _, after = cleaned.partition(":")
                if '1' in after:
                    result_value = 1.0
                elif '0' in after:
                    result_value = 0.0

        # Fallback to the previous strict parsing if no explicit [RESULT] line was found
        if result_value is None:
            compact = txt.replace("`", "").replace(" ", "").strip()
            if compact == "1":
                result_value = 1.0
            elif compact == "0":
                result_value = 0.0

        return (result_value if result_value is not None else np.nan, reason_value)

    def _score_single(self, args: tuple[list, str, int]) -> tuple[float, str]:
        claims, input_text, step_idx = args
        claims_prefix = claims[:step_idx + 1]
        prompt = self.format_prompt(input_text, claims_prefix)
        if self.debug:
            print(prompt)
        response = self.chat.ask(prompt)
        if isinstance(response, tuple):
            reply, reasoning_content = response
        else:
            reply, reasoning_content = response, ""
        if self.debug:
            print("=================")
            print(f"Reasoning content Step {step_idx + 1}: {reasoning_content}")
            print(f"Reply: {reply}")
        parsed_score, parsed_reason = self.parse_reply(reply)
        if self.debug and parsed_reason:
            print(f"Reason: {parsed_reason}")

        claim = claims[step_idx]
        if len(claim.aligned_token_ids) == 0 or np.isnan(parsed_score):
            label = np.nan
        else:
            label = 1 if parsed_score == 0.0 else 0

        return label, parsed_reason

    def __call__(
        self,
        stats: Dict[str, np.ndarray],
        target_texts: List[str],
        output_path: str,
    ) -> list:
        assert output_path is not None and output_path.endswith(".csv")
        
        input_texts = stats["input_texts"]

        print(f"Using {self.n_threads} threads")

        claim_sets = stats["claims"]
        claim_labels = [
            [np.nan for _ in range(len(claims))]
            for claims in claim_sets
        ]
        verdict_reasons = [
            ["" for _ in range(len(claims))]
            for claims in claim_sets
        ]

        tasks = []
        for sample_idx, (input_text, claims) in enumerate(zip(input_texts, claim_sets)):
            for step_idx in range(len(claims)):
                tasks.append((sample_idx, claims, input_text, step_idx))

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = {
                executor.submit(self._score_single, (claims, input_text, step_idx)): (sample_idx, step_idx)
                for sample_idx, claims, input_text, step_idx in tasks
            }
            for future in tqdm(as_completed(futures), desc="Verifying factuality", total=len(futures), disable=not self.progress_bar):
                sample_idx, step_idx = futures[future]
                label, reason = future.result()
                claim_labels[sample_idx][step_idx] = label
                verdict_reasons[sample_idx][step_idx] = reason
        
        claims_sents = [[cl.sentence for cl in claims] for claims in claim_sets]

        df = pd.DataFrame({
            "input_text": input_texts,
            "claims": claims_sents,
            "claim_labels": claim_labels,
            "verdict_reasons": verdict_reasons,
        })
        df.to_csv(output_path, index=False)


        return claim_labels


