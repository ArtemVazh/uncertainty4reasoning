import numpy as np
import pandas as pd
import re

from transformers import AutoTokenizer
from lm_polygraph import UEManager
from argparse import Namespace
from parse import parse
from collections import defaultdict
from metrics import ROCAUC, PRAUC, ECE
from lm_polygraph.ue_metrics.ue_metric import UEMetric
from synthetic_dataset_generation.utils.steps_extractor import StepsExtractor


def extract_steps(man, base_model_path: str, hf_cache: str = None) -> list[list[str]]:
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True,
                                                   cache_dir=hf_cache)
    return StepsExtractor()(
        man.stats,
        man.stats['input_texts'],
        model=Namespace(tokenizer=base_tokenizer),
    )["claims"]


def extract_questions(man, prompt_file: str) -> list[str]:
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    input_texts = man.stats['input_texts']
    return [parse(prompt, inp_text).named['q'] for inp_text in input_texts]


def parse_ans(s):
    if '####' in s:
        return float(s.split('####')[-1].replace(',', ''))

    if r'\boxed{' in s:
        x = s.split(r'\boxed{')[-1].split('}')[0].replace(',', '')
        x = x.split('=')[-1]
        if x.endswith('%'):
            x = x[:-1]
        try:
            return float(x)
        except:
            return None

    if r'<Answer>:' in s:
        x = s.split(r'<Answer>:')[-1].replace(',', '')
        matches = re.findall(r'[-+]?\d*\.?\d+', x)
        if not matches:
            return None
        number_str = matches[-1]
        number = float(number_str)
        if number.is_integer():
            number = int(number)
        return number

    matches = re.findall(r'[-+]?\d*\.?\d+', s.replace(',', ''))
    if matches:
        number_str = matches[-1]
        number = float(number_str)
        if number.is_integer():
            number = int(number)
        return number

    return None


def print_test_stats(
        man: UEManager,
        has_final_ans: bool,
        model_path: str,
        hf_cache: str | None = None,
):
    claim_extractor = StepsExtractor()
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=hf_cache)
    claims = claim_extractor(man.stats, man.stats['input_texts'], model=Namespace(tokenizer=tokenizer))["claims"]
    targets = np.array(man.gen_metrics['claim', 'StepFactCheck'])

    acc, last_tgt = [], 0
    for t, h, cl in zip(man.stats['target_texts'], man.stats['greedy_texts'], claims):
        if has_final_ans:
            at, ah = parse_ans(t), parse_ans(h)
            acc.append(np.isclose(at, ah) if ah is not None else 0)
        else:
            acc.append(all(t == 1 for t in targets[last_tgt:last_tgt + len(cl)]))
            last_tgt += len(cl)

    print('Skipping {} nan steps'.format(np.isnan(targets).sum()))
    print()
    targets = targets[~np.isnan(targets)].astype(int)
    print('Total problems:', len(man.stats['input_texts']))
    print('Correct answers: {} ({}%)'.format(sum(acc), round(100 * np.mean(acc), 2)))
    print('Incorrect answers: {} ({}%)'.format(len(acc) - sum(acc), round(100 - 100 * np.mean(acc), 2)))
    print()
    print('Total steps: {}'.format(len(targets)))
    print('Correct steps: {} ({}%)'.format(len(targets) - sum(targets), round(100 - 100 * np.mean(targets), 2)))
    print('Incorrect steps: {} ({}%)'.format(sum(targets), round(100 * np.mean(targets), 2)))


def calculate_metrics(
        man: UEManager,
        metrics: list[UEMetric] = [ROCAUC(), PRAUC(), ECE()],
) -> pd.DataFrame:
    methods = {ue_name: ue_vals for (_, ue_name), ue_vals in man.estimations.items()}
    targets = man.gen_metrics['claim', 'StepFactCheck']

    for key, val in methods.items():
        print(f'{key}: {len(val)} values')
    print(f'Targets: {len(targets)} values')

    res_df = defaultdict(dict)
    for method_nm, method_vals in methods.items():
        if len(method_vals) != len(targets):
            print(f'Skipping {method_nm}: inconsistent number of samples, '
                  f'expected {len(targets)}, got {len(method_vals)}')
            continue
        for m in metrics:
            res_df[str(m)][method_nm] = m(method_vals, targets)

    return pd.DataFrame(res_df)
