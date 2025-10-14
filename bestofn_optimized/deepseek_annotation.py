import argparse
import torch
import logging
from bestofn.deepseek_annotation import Annotator
from bestofn_optimized.run_uhead import load_bon_dataset, parse_tuple, load_prompt

log = logging.getLogger()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=parse_tuple, default=("openai/gsm8k", "main"))
    parser.add_argument('--dataset-split', type=parse_tuple, default=None)
    parser.add_argument('--question-col', type=str, default="question")
    parser.add_argument('--answer-col', type=str, default="answer")
    parser.add_argument('--prompt-file', type=str, default=None)
    parser.add_argument('--n-threads', type=int, default=1, help="Number of threads to use")
    parser.add_argument('--save-path', required=True)
    args = parser.parse_args()

    prompt = load_prompt(args.prompt_file)
    dataset = load_bon_dataset(args.dataset_path, args.dataset_split)

    problems, solutions = [], []
    for i in range(len(dataset)):
        problems.append(dataset[args.question_col][i])
        solutions.append(dataset['reply'][i])

    anno = Annotator(prompt=open(args.prompt_file, 'r').read(), n_threads=args.n_threads)
    log.info(f"Annotating {len(solutions)} solutions")
    annotations = anno(problems, solutions)

    if 'deepseek_anno' in dataset.column_names:
        dataset = dataset.remove_columns('deepseek_anno')
    dataset = dataset.add_column('deepseek_anno', annotations)

    log.info(f'Saving to {args.save_path}')
    dataset.save_to_disk(args.save_path)
    log.info('Done.')
