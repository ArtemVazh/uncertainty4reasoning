
from concurrent.futures import ThreadPoolExecutor, as_completed
from parse import parse
from openai import OpenAI
import os
import logging
import numpy as np
from tqdm import tqdm
from typing import Dict, List
import diskcache as dc
import threading

from lm_polygraph.generation_metrics.generation_metric import GenerationMetric
from lm_polygraph.stat_calculators.extract_claims import Claim
from synthetic_dataset_generation.utils.chat import DeepSeekChat, OpenAIChat

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
- Respond with the number **only**, with no extra text or explanation.

PROBLEM:
{problem}

GROUND-TRUTH SOLUTION:
{answer}

STUDENT'S REASONING PROCESS SO FAR:
{steps}

THE LATEST STEP TO EVALUATE:
{latest_step}
""".strip(" \n")


GENERATION_CONFIGS={
    'openai/gpt-oss-120b': {
        'extra_body': {"reasoning_effort": 'medium'},
    },
    "Qwen/Qwen3-8B": {
            "temperature": 0.6,
            "top_p": 0.95,
            "extra_body": {"enable_thinking": True, "top_k": 20},
    },
    "microsoft/Phi-4-reasoning-plus": {
            "temperature": 0.8,
            "top_p": 0.95,
            "extra_body": {"enable_thinking": True, "top_k": 50},
    },
}


class LocalChat:
    """
    Allows for the implementation of a singleton class to chat with Phi-4 model for dataset marking.
    """
    
    
    def __init__(
        self,
        model: str = "openai/gpt-oss-120b",
        base_url: str = "http://localhost:8000/v1",
        cache_path: str = os.path.expanduser("~") + "/.cache",
        system_prompt: str = None,
        generation_config: dict = None,
    ):
        """
        Parameters
        ----------
        model: str
            the model to use in OpenAI to chat.
        base_url: str
            the base url to access the local model.
        """
        self.cache_path = os.path.join(cache_path, "openai_chat_cache.diskcache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        self.base_url = base_url
        self.model = model
        self.generation_config = GENERATION_CONFIGS[model] if generation_config is None else generation_config
        self.system_prompt = system_prompt
        self.client = OpenAI(base_url=self.base_url, api_key='EMPTY')

        # Initialize cache with proper settings
        cache_settings = dc.DEFAULT_SETTINGS.copy()
        cache_settings["eviction_policy"] = "none"
        cache_settings["size_limit"] = int(1e12)
        cache_settings["cull_limit"] = 0
        self.cache = dc.Cache(self.cache_path, **cache_settings)
        self._lock = threading.Lock()

    def ask(self, message: str, **kwargs) -> str:

        reply, reasoning_content = self.cache.get((self.model, message), ('', ''))
        if reply == '' and reasoning_content == '':
            if self.system_prompt is not None:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": message},
                ]
            else:
                messages = [
                    {"role": "user", "content": message},
                ]
            chat = self._send_request(messages, self.generation_config)
            if chat is None:
                reply, reasoning_content = '', ''
            else:
                reasoning_content = chat.choices[0].message.reasoning_content
                reply = chat.choices[0].message.content

            with self._lock:
                self.cache[(self.model, message)] = (reply, reasoning_content)
        else:
            print("Loaded from cache")

        if "please provide" in reply.lower():
            return ""
        if "to assist you" in reply.lower():
            return ""
        if "as an ai language model" in reply.lower():
            return ""

        return reply, reasoning_content

    def _send_request(self, messages, generation_config):
        try:
            response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            **generation_config,
                        )
        except Exception as e:
            log.info(
                f"Request to OpenAI failed with exception: {e}."
            )
            return None

        return response


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
            self.chat = LocalChat(model=model, base_url=base_url, cache_path=cache_path)
        else:
            print(f"Using Remote {model}")
            if 'deepseek' in model:
                self.chat = DeepSeekChat(cache_path,  model=model, api_key=api_key, wait_times=wait_times)
            else:
                self.chat = OpenAIChat(model, cache_path=cache_path)
        self.progress_bar = progress_bar
        self.n_threads = n_threads
        self.debug = debug

    def __str__(self):
        return "StepFactCheckThinking"

    def parse_problem(self, input_text: str):
        try:
            return parse(self.prompt, input_text).named['q']
        except Exception as e:
            # For run_extract_verify_claims.py, input texts are raw questions without prompt
            return input_text

    def format_prompt(self, input_text: str, claims: list[Claim], answer: str) -> str:
        problem = self.parse_problem(input_text)
        steps = ' '.join([cl.sentence for i, cl in enumerate(claims)])
        last_claim = claims[-1].claim_text
        return STEP_CHECKING_PROMPT.format(problem=problem, answer=answer, steps=steps, latest_step=last_claim)

    def parse_reply(self, reply: str) -> int | float:
        if '1' in reply and '0' in reply:
            return np.nan
        elif '1' in reply:
            return 1.0
        elif '0' in reply:
            return 0.0
        else:
            return np.nan

    def _score_single(self, args: tuple[list, str, str]) -> list:
        claims, input_text, answer = args
        correctness_labels = []
        for i in range(len(claims)):
            claims_so_far = claims[:i+1]
            prompt = self.format_prompt(input_text, claims_so_far, answer)
            if self.debug:
                print(prompt)
            reply, reasoning_content = self.chat.ask(prompt, json_output=False)
            if self.debug:
                print("=================")
                print(f"Reasoning content Step {i+1}: {reasoning_content}")
                print(f"Reply: {reply}")
            parsed_reply = self.parse_reply(reply)
            correctness_labels.append(parsed_reply)
            if parsed_reply == 0:
                break
        
        return [
            (
                np.nan if len(claims[i].aligned_token_ids) == 0 or np.isnan(correctness_labels[i]) else
                1 if correctness_labels[i] == 0 else
                0
            ) for i in range(len(correctness_labels))
        ]

    def __call__(
            self,
            stats: Dict[str, np.ndarray],
            target_texts: List[str],
    ) -> list:
        input_texts = stats["input_texts"]

        if "answers" in stats.keys():
            target_texts = stats["answers"]

        all_inputs = [
            (claims, input_text, answer)
            for input_text, claims, answer in zip(input_texts, stats["claims"], target_texts)
        ]

        print(f"Using {self.n_threads} threads")
        if self.n_threads == 1:
            claim_labels = []
            for item in tqdm(all_inputs, desc="Verifying claims", total=len(all_inputs), disable=not self.progress_bar):
                claim_labels.append(self._score_single(item))
        else:
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                futures = {executor.submit(self._score_single, item): idx
                           for idx, item in enumerate(all_inputs)}
                claim_labels = [None] * len(all_inputs)
                for future in tqdm(as_completed(futures), desc="Verifying claims",
                                   total=len(all_inputs), disable=not self.progress_bar):
                    idx = futures[future]
                    claim_labels[idx] = future.result()

        return claim_labels

