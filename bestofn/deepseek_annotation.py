import numpy as np
import argparse
import torch
import logging
from tqdm import tqdm
from parse import parse
from concurrent.futures.thread import ThreadPoolExecutor

from synthetic_dataset_generation.utils.deepseek_chat import DeepSeekChat

log = logging.getLogger()

ANNOTATION_PROMPT = r'''
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
    ):
        self.chat = DeepSeekChat(cache_path, model=model, api_key=api_key, wait_times=wait_times)
        self.prompt = prompt
        self.n_threads = n_threads

    def _score_single(self, inp: tuple[str, str]) -> float:
        problem, solution = inp
        problem = parse(self.prompt, problem).named['q']
        prompt = ANNOTATION_PROMPT.format(problem=problem, solution=solution)
        reply = self.chat.ask(prompt)
        if '<Grade>: Correct' in reply:
            return 0
        elif '<Grade>: Incorrect' in reply:
            return 1
        else:
            return np.nan

    def __call__(self, problems: list[str], solutions: list[str]) -> list[float]:
        all_inputs = zip(problems, solutions)
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

    b = torch.load(args.save_path)
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
    # log.info(f'Saving to {args.save_path}')
    # torch.save(b, args.save_path)
