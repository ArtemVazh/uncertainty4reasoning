
from concurrent.futures import ThreadPoolExecutor, as_completed
from parse import parse
import numpy as np
import json
from tqdm import tqdm
from typing import Dict, List
from openai import OpenAI
import os
import logging
import diskcache as dc
import threading

from lm_polygraph.generation_metrics.generation_metric import GenerationMetric
from lm_polygraph.utils.openai_chat import OpenAIChat
from lm_polygraph.stat_calculators.extract_claims import Claim
from synthetic_dataset_generation.utils.chat import DeepSeekChat

log = logging.getLogger()

VERSION = 'correctness_redundancy'

PROMPT1_TEMPLATE = {
    'correctness_redundancy': r'''You are given a problem, a ground-truth solution, and a step-by-step student solution. Your task is to analyze each step in the student’s solution to determine whether it is both correct and informative. 

Correctness: if a step is correct, it contains no mistakes in calculation and logic.
Informative: if a step is informative, it provides new information that is not a paraphrase of existing context and previous steps, and it contributes towards getting closer to the answer.

Instructions:
- Carefully examine each student step for logical/calculation errors or unnecessary/redundant reasoning.
- If all steps are correct and they lead to the same final answer as the ground-truth solution, conclude that there are no errors.
- If any step is incorrect (contains logical or calculation error) or non-informative (redundant or has no contribution to the final answer), identify and report those specific steps with an explanation.

PROBLEM:
{problem}

GROUND-TRUTH SOLUTION:
{answer}

STUDENT'S SOLUTION STEPS:
{steps}

Now, please evaluate whether the student’s steps are correct and logical.''',
    'correctness': r'''You are given a problem, a ground-truth solution, and a step-by-step student solution. Your task is to analyze each step in the student’s solution to determine whether it is both logically correct and relevant.

Instructions:
- Carefully examine each student step for logical errors or unnecessary/redundant reasoning.
- If all steps are correct and they lead to the same final answer as the ground-truth solution, conclude that there are no errors.
- If any step contains an error that would prevent the student from reaching the correct solution, identify and report those specific steps with an explanation.

PROBLEM:
{problem}

GROUND-TRUTH SOLUTION:
{answer}

STUDENT'S SOLUTION STEPS:
{steps}

Now, please evaluate whether the student’s steps are correct and logical.'''
}


PROMPT2_TEMPLATE = {
    'correctness_redundancy': r'''
You are given:
- A problem
- A student's step-by-step solution (as a Python list of string steps)
- An assessment of student's solution

Your task:
Output a json object with the following fields:
- "correctness": a list of 0/1 values, where 1 (correct) indicates the step contains no mistakes in calculation and logic; otherwise 0 (incorrect).
- "informativeness": a list of 0/1 values, where 1 (informative) means the step provides new information that is not a paraphrase of existing context and previous steps, and it contributes towards getting closer to the answer. Otherwise 0 (non-informative).

Important:
- Output only the json object with the fields "correctness" and "informativeness", nothing else.
- The correctness list must have correctness labels for all steps and the final answer (in this case, list length should be {list_length}).
- The informativeness list must have one fewer entry than the number of steps (i.e., {list_length_1}), because it should only score the reasoning steps and NOT the final answer step.

PROBLEM:
{problem}

STUDENT'S SOLUTION STEPS:
{steps}

ASSESSMENT OF STUDENT SOLUTION STEPS:
{reply}

OUTPUT JSON:
''',
    'correctness': r"""
You are given:
- A problem
- A student's step-by-step solution (as a Python list of string steps)
- An assessment of student's solution

Your task:
Output a single Python list where each element is:
- 1 if the corresponding step is correct
- 0 if the step is incorrect

Important:
- Output only the list, nothing else.
- The list must have the same length as the number of steps (in this case, list length must be {list_length}).

PROBLEM:
{problem}

STUDENT'S SOLUTION STEPS:
{steps}

ASSESSMENT OF STUDENT SOLUTION STEPS:
{reply}

OUTPUT LIST:
""",
}


PHI4_SYSTEM_PROMPT = "You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. Now, try to solve the following question through the above guidelines:"

class Phi4ReasoningChat:
    """
    Allows for the implementation of a singleton class to chat with Phi-4 model for dataset marking.
    """
    
    
    def __init__(
        self,
        model: str = "microsoft/Phi-4-reasoning",
        base_url: str = "http://localhost:8000/v1",
        cache_path: str = os.path.expanduser("~") + "/.cache",
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
            print(f"Sending request to {self.model}")
            messages = [
                {"role": "system", "content": PHI4_SYSTEM_PROMPT},
                {"role": "user", "content": message},
            ]
            chat = self._send_request(messages)
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

    def _send_request(self, messages):
        try:
            response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_tokens=28000,
                            temperature=0.8,
                            top_p=0.95,
                            extra_body={"top_k": 50},
                        )
        except Exception as e:
            log.info(
                f"Request to OpenAI failed with exception: {e}."
            )
            return None

        return response



class Qwen3Chat:
    """
    Allows for the implementation of a singleton class to chat with OpenAI model for dataset marking.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3-8B",
        base_url: str = "http://localhost:8000/v1",
        cache_path: str = os.path.expanduser("~") + "/.cache",
        enable_thinking: bool = True,
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
        self.enable_thinking = enable_thinking
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
            print(f"Sending request to {self.model}")
            messages = [
                {"role": "user", "content": message},
            ]
            chat = self._send_request(messages, enable_thinking=self.enable_thinking)
            if chat is None:
                reply, reasoning_content = '', ''
            else:
                reasoning_content = chat.choices[0].message.reasoning_content
                reply = chat.choices[0].message.content
                if reply is None:
                    reply, reasoning_content = '', ''

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

    def _send_request(self, messages, enable_thinking=True):

        try:
            response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_tokens=32768,
                            temperature=0.6,
                            top_p=0.95,
                            extra_body={"enable_thinking": enable_thinking, "top_k": 20},
                        )
        except Exception as e:
            log.info(
                f"Request to OpenAI failed with exception: {e}."
            )
            return None
        return response



class StepFactCheck(GenerationMetric):
    def __init__(
            self,
            prompt_file: str,
            cache_path: str = "~/.cache",
            model: str = None,
            api_key: str = None,
            progress_bar: bool = True,
            n_threads: int = 1,
            wait_times: tuple = (5, 10, 30, 60, 120),
            version: str = VERSION,
            label_type: str = 'correctness',
            base_url: str = None,
            debug: bool = False,
    ):
        super().__init__(["input_texts", "claims"], "claim")

        with open(prompt_file, 'r') as f:
            self.prompt = f.read()

        if version in ['correctness_redundancy']:
            self.json_output = True
        else:
            self.json_output = False

        if base_url is not None and 'localhost' in base_url:
            print(f"Using Local {model}")
            if 'phi' in model.lower():
                self.chat = Phi4ReasoningChat(model=model, base_url=base_url, cache_path=cache_path)
            elif 'qwen' in model.lower():
                self.chat = Qwen3Chat(model=model, base_url=base_url, cache_path=cache_path, enable_thinking=True)
            else:
                raise ValueError(f"Model {model} not supported")
        else:
            print(f"Using Remote {model}")
            if 'deepseek' in model:
                self.chat = DeepSeekChat(cache_path=cache_path, model=model, api_key=api_key, api_base=base_url, wait_times=wait_times)
            else:
                self.chat = OpenAIChat(model, cache_path=cache_path)

        self.label_type = label_type
        # use this for OpenAI
        # self.chat = DeepSeekChat(api_base=None, model='gpt-4o', cache_path=cache_path, api_key=api_key, wait_times=wait_times)

        self.progress_bar = progress_bar
        self.n_threads = n_threads
        self.version = version
        self.debug = debug

    def __str__(self):
        return "StepFactCheck" + "_" + self.label_type

    def parse_problem(self, input_text: str):
        try:
            return parse(self.prompt, input_text).named['q']
        except Exception as e:
            # For run_extract_verify_claims.py, input texts are raw questions without prompt
            return input_text

    def prompt1(self, input_text: str, claims: list[Claim], answer: str) -> str:
        problem = self.parse_problem(input_text)
        steps = '\n'.join([cl.claim_text.strip() for i, cl in enumerate(claims)])
        return PROMPT1_TEMPLATE[self.version].format(problem=problem, answer=answer,
                                                                                    steps=steps)

    def prompt2(self, input_text: str, claims: list[Claim], answer: str, reply: str) -> str:
        problem = self.parse_problem(input_text)
        steps = [cl.claim_text.strip() for i, cl in enumerate(claims)]
        if self.json_output:
            return PROMPT2_TEMPLATE[self.version].format(problem=problem, steps=steps, reply=reply, list_length=len(steps), list_length_1=len(steps) - 1)
        else:
            return PROMPT2_TEMPLATE[self.version].format(problem=problem, steps=steps, reply=reply, list_length=len(steps))

    def parse_reply(self, reply: str) -> list[int] | None:
        if 'all steps are correct' in reply.lower():
            return []
        orig_reply = reply
        reply = reply.strip().replace(' ', '').replace('Step', '')
        if '```python' in reply:
            reply = reply.split('```python')[-1].split('```')[0].strip()
        if reply.startswith('[') and reply.endswith(']'):
            reply = reply[1:-1]
        try:
            return [int(x) for x in reply.split(',')]
        except Exception as e:
            log.warning('Skipping text, because could not parse DeepSeek reply: {}'.format(orig_reply))
            return None

    def _score_single(self, args: tuple[list, str, str]) -> list:
        claims, input_text, answer = args
        q1 = self.prompt1(input_text, claims, answer)
        response_tuple = self.chat.ask(q1, json_output=False)
        if isinstance(response_tuple, tuple):
            reply, reasoning_content = response_tuple
        else:
            reply, reasoning_content = response_tuple, ""
        if self.debug:
            print(q1)
            print("=================")
            print(f"Reasoning content Step 1: {reasoning_content}")
        q2 = self.prompt2(input_text, claims, answer, reply)
        response_tuple = self.chat.ask(q2, json_output=self.json_output)
        if isinstance(response_tuple, tuple):
            reply, reasoning_content = response_tuple
        else:
            reply, reasoning_content = response_tuple, ""
        if self.debug:
            print(q2)
            print("=================")
            print(f"Reasoning content Step 2: {reasoning_content}")
        if self.json_output:
            try:
                json_reply = json.loads(reply)
                correctness_labels = json_reply['correctness']
                informativeness_labels = json_reply['informativeness']
            except Exception as e:
                log.warning(f"Skipping text, because could not parse DeepSeek reply: {reply}")
                return [np.nan for _ in range(len(claims))]
        else:
            correctness_labels: list[int] | None = self.parse_reply(reply)
            informativeness_labels = None

        if self.label_type == 'correctness':
            claim_labels = correctness_labels
        elif self.label_type == 'informativeness':
            claim_labels = informativeness_labels
        else:
            raise ValueError(f"Label type {self.label_type} not supported")

        if claim_labels is None:
            return [np.nan for _ in range(len(claims))]  # will be skipped at evaluation
        if len(claim_labels) + 1 == len(claims):
            claim_labels.append(np.nan)  # last answer is undefined
        if len(claim_labels) != len(claims):
            log.warning(f"Prompt 2: {q2}")
            log.warning(
                'Skipping text, because of inconsistend number of '
                'labels in DeepSeek reply: expected {}, got {}'.format(len(claims), reply))
            return [np.nan for _ in range(len(claims))]  # will be skipped at evaluation
        
        return [
            (
                np.nan if (len(claims[i].aligned_token_ids) == 0 or
                          (isinstance(claim_labels[i], (int, float)) and np.isnan(claim_labels[i]))) else
                1 if claim_labels[i] == 0 else
                0
            ) for i in range(len(claims))
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
            for item in tqdm(all_inputs, desc=f"Verifying claims ({self.label_type})", total=len(all_inputs), disable=not self.progress_bar):
                claim_labels.append(self._score_single(item))
        else:
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                futures = [executor.submit(self._score_single, item) for item in all_inputs]
                claim_labels = []
                for future in tqdm(futures, desc=f"Verifying claims ({self.label_type})", disable=not self.progress_bar):
                    claim_labels.append(future.result())

        return claim_labels

