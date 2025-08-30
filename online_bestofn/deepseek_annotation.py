import numpy as np
import argparse
import torch
import logging
import re
import os
from tqdm import tqdm
from parse import parse
from concurrent.futures.thread import ThreadPoolExecutor

from synthetic_dataset_generation.utils.deepseek_chat import DeepSeekChat

log = logging.getLogger()

ANNOTATION_PROMPT_NON_UNIQUE_GOLD_ANSWER = r'''\
You are given a problem, a non-unique ground-truth solution for reference, and a step-by-step student solution. Your task is to assess whether the solution is **correct** or **incorrect**.

Instructions:
- Carefully examine each student step for logical errors.
- If student's answer is the semantically same as the ground-truth solution, conclude that the solution is correct.
- If student's answer is semantically different from the ground-truth solution, but still correctlt solves the problem without any logical errors, conclude that the solution is correct.
- Otherwise, conclude that the solution is incorrect.

Respond using the **exact format** below, do not include any text outside this template.
Output format:
<start of response>
Solution comments:
... your comments on the solution, explaining reasoning, pointing out any errors or confirming correctness ...
<Grade>: (Correct|Incorrect)
<end of response>

PROBLEM:
{problem}

GROUND-TRUTH SOLUTION:
{gold_answer}

STUDENT'S SOLUTION:
{solution}
'''

ANNOTATION_PROMPT_UNIQUE_GOLD_ANSWER = r'''\
You are given a problem, a unique ground-truth solution, and a step-by-step student solution. Your task is to assess whether the solution is **correct** or **incorrect**.

Instructions:
- Carefully examine each student step for logical errors.
- If student's answer is the semantically same as the unique ground-truth solution, conclude that the solution is correct.
- Otherwise, conclude that the solution is incorrect.

Respond using the **exact format** below, do not include any text outside this template.
Output format:
<start of response>
Solution comments:
... your comments on the solution, explaining reasoning, pointing out any errors or confirming correctness ...
<Grade>: (Correct|Incorrect)
<end of response>

PROBLEM:
{problem}

GROUND-TRUTH SOLUTION:
{gold_answer}

STUDENT'S SOLUTION:
{solution}
'''


OLD_ANNOTATION_PROMPT = r'''
You will be given a <Problem> and its proposed <Solution>. Your task is to assess whether the solution is **correct** or **incorrect**.

Respond using the **exact format** below, do not include any text outside this template.
Output format:
<start of response>
Solution comments:
... your comments on the solution, explaining reasoning, pointing out any errors or confirming correctness ...
<Grade>: (Correct|Incorrect)
<end of response>

<Problem>: {problem}

<Solution>: {solution}
'''

class Annotator:
    def __init__(
            self,
            prompt: str,
            cache_path: str = "~/.cache",
            model: str = 'deepseek-reasoner',
            api_key: str | None = None,
            n_threads: int = 1,
            wait_times: tuple = (5, 10, 30, 60, 120),
            annotation_prompt_type: str = "non_unique",  # "unique" or "non_unique"
    ):
        # If api_key is None, try to read from environment variable
        if api_key is None:
            api_key = os.getenv('DEEPSEEK_API_KEY')

        # import pdb; pdb.set_trace()
        
        self.chat = DeepSeekChat(cache_path, model=model, api_key=api_key, wait_times=wait_times)
        self.prompt = prompt
        self.n_threads = n_threads
        self.api_key = api_key
        
        # Select the appropriate annotation prompt
        if annotation_prompt_type == "unique":
            self.annotation_prompt = ANNOTATION_PROMPT_UNIQUE_GOLD_ANSWER
        elif annotation_prompt_type == "non_unique":
            self.annotation_prompt = ANNOTATION_PROMPT_NON_UNIQUE_GOLD_ANSWER
        else:
            raise ValueError(f"Invalid annotation_prompt_type: {annotation_prompt_type}. Must be 'unique' or 'non_unique'")

    def _score_single(self, inp: tuple[str, str, str]) -> float:
        problem, solution, gold_answer = inp
        # import pdb; pdb.set_trace()
        parsed_result = parse(self.prompt, problem)
        # import pdb; pdb.set_trace()
        if parsed_result is not None:
            problem = parsed_result.named['q']
        else:
            # Fallback: extract question using regex between "<Question>: " and "<|im_end|>"
            match = re.search(r'<Question>:\s*(.*?)<\|im_end\|>', problem, re.DOTALL)
            if match:
                problem = match.group(1).strip()
            # If regex also fails, keep the original problem text
        prompt = self.annotation_prompt.format(problem=problem, solution=solution, gold_answer=gold_answer)
        print(f'Using prompt:\n{prompt}')
        # import pdb; pdb.set_trace()
        reply = self.chat.ask(prompt)
        if '<Grade>: Correct' in reply:
            return 0
        elif '<Grade>: Incorrect' in reply:
            return 1
        else:
            return np.nan

    def __call__(self, problems: list[str], solutions: list[str], gold_answers: list[str]) -> list[float]:
        all_inputs = zip(problems, solutions, gold_answers)
        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            futures = [executor.submit(self._score_single, item) for item in all_inputs]
            labels = []
            for future in tqdm(futures, desc="Verifying solutions"):
                labels.append(future.result())
            return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-path', type=str, required=True,
                        help="Path to bestofn file to calculate annotations for")
    parser.add_argument('--prompt-file', type=str, required=True,
                        help="Path to prompt file used to generate bestofn")
    parser.add_argument('--n-threads', type=int, default=1, help="Number of threads to use")
    args = parser.parse_args()

    b = torch.load(args.save_path, weights_only=False)
    problems, solutions = [], []
    for r in b:
        if "sample_texts" not in r:
            continue
        problems += [r["input"] for _ in r["sample_texts"]]
        solutions += r["sample_texts"]
    anno = Annotator(prompt=open(args.prompt_file, 'r').read(), n_threads=args.n_threads)
    log.info(f"Annotating {len(solutions)} solutions to {len(b)} problems")
    annotations = anno(problems, solutions)
    for i in range(len(b)):
        if "sample_texts" not in b[i]:
            continue
        l = len(b[i]["sample_texts"])
        b[i]["deepseek_annotations"] = annotations[:l]
        annotations = annotations[l:]
    log.info(f'Saving to {args.save_path}')
    torch.save(b, args.save_path)