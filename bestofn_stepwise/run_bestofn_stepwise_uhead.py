import argparse
import torch
import os

from tqdm import tqdm
from datasets import load_dataset
from bestofn.run_bestofn_uhead import load_model
from bestofn_stepwise.stat_calculators.stepwise_uhead_minimization import StepwiseUheadMinimizationCalculator
from lm_polygraph import WhiteboxModel
from lm_polygraph.stat_calculators import StatCalculator
from luh import AutoUncertaintyHead


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Dataset to evaluate on (HuggingFace name or local path)")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split (e.g., test)")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the output .torch")
    parser.add_argument("--hf-cache", type=str, default=None, help="Path to HuggingFace cache directory")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-1.7B", help="Model name or path for generation")
    parser.add_argument("--uhead-path", type=str, default="rediska0123/uhead_Qwen3-1.7B_gsm8k",
                        help="Model name or path for generation")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (e.g., 'cuda' or 'cpu')")
    return parser


def update_stats(stats: list[dict], stat_calculators: list[StatCalculator], model: WhiteboxModel, save_path: str):
    for i in tqdm(range(len(stats)), total=len(stats), desc='Generating samples'):
        if all(stat in stats[i].keys() for stat_calc in stat_calculators for stat in stat_calc.stats):
            continue
        if 'input_texts' not in stats[i].keys():
            raise Exception(f'Could not find input texts in stat keys: {stats[i].keys()}')
        for stat_calc in stat_calculators:
            print(f'Calculating {str(stat_calc)}...')
            stats[i].update(stat_calc(
                dependencies=stats[i],
                texts=stats[i]['input_texts'],
                model=model,
            ))
        torch.save(stats, save_path)
        print(f'Saved at {save_path}')


def main(args):
    model = load_model(args.model_path, args.device)
    uhead = AutoUncertaintyHead.from_pretrained(args.uhead_path, model.model)
    if os.path.exists(args.save_path):
        stats = torch.load(args.save_path, weights_only=False)
    else:
        assert args.dataset_path is not None
        dataset = load_dataset(args.dataset_path, split=args.dataset_split, cache_dir=args.hf_cache)
        stats: list[dict] = [{'input_texts': [sample["question"]]} for sample in dataset]
    update_stats(stats, [StepwiseUheadMinimizationCalculator(uhead)], model, args.save_path)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
