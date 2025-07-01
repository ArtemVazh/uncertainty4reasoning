import argparse
import torch
import logging
from bestofn.deepseek_annotation import Annotator

log = logging.getLogger()

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
        for model in ['uhead', 'prm', 'reasoneval']:
            if f'min_{model}_final_texts' not in r:
                continue
            problems += r['input_texts']
            solutions += r[f'min_{model}_final_texts']
    anno = Annotator(prompt=open(args.prompt_file, 'r').read(), n_threads=args.n_threads)
    log.info(f"Annotating {len(solutions)} solutions to {len(b)} problems")
    annotations = anno(problems, solutions)
    anno_dict: dict[tuple[str, str], float] = {}
    for problem, solution, a in zip(problems, solutions, annotations):
        anno_dict[problem, solution] = a
    for r in b:
        for model in ['uhead', 'prm', 'reasoneval']:
            if f'min_{model}_final_texts' not in r:
                continue
            problem = r['input_texts'][0]
            solution = r[f'min_{model}_final_texts'][0]
            r[f'deepseek_annotations_{model}'] = anno_dict[problem, solution]
    log.info(f'Saving to {args.save_path}')
    torch.save(b, args.save_path)
