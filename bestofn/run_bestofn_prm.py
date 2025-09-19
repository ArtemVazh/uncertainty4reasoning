import argparse
import os
import logging
import random
import numpy as np
import torch

from datasets import load_dataset

from baselines.prm import load_prm_calculator_by_model_path
from bestofn.estimators.prm import PRMEstimator
from lm_polygraph import WhiteboxModel
from bestofn.bestofn_utils import update_bestofn
from configs.load_qwen import load_model as load_qwen_model, load_tokenizer as load_qwen_tokenizer
from synthetic_dataset_generation.utils.steps_extractor import StepsExtractor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bestofn_eval")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Dataset to evaluate on (HuggingFace name or local path)")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split (e.g., test)")
    parser.add_argument('--prompt-file', type=str, required=True, help="Path to prompt template file")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the output .torch")
    parser.add_argument("--hf-cache", type=str, default=None, help="Path to HuggingFace cache directory")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-1.7B", help="Model name or path for generation")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--subset", type=int, default=None, help="Only process first N samples from dataset")
    parser.add_argument('--prm-model-path', type=str, nargs='+', default=[
        "Qwen/Qwen2.5-Math-7B-PRM800K",
        "Qwen/Qwen2.5-Math-PRM-7B",
        "peiyi9979/math-shepherd-mistral-7b-prm",
        "RLHFlow/Llama3.1-8B-PRM-Mistral-Data",
        "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
        "GenPRM/GenPRM-1.5B-simple",
        "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",  # loads slow (can take up to 15 mins)
        # # "GenPRM/GenPRM-1.5B",  # very slow
        "universalprm/Universal-PRM",
        "HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2",
    ], help="Path(s) or name(s) of the PRM model(s)")
    return parser


def load_model(model_path, device):
    tokenizer = load_qwen_tokenizer(model_path)
    base_model = load_qwen_model(model_path, device)
    base_model.eval()
    return WhiteboxModel(base_model, tokenizer)


def main(args):
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    log.info(f"Set random seed to {args.seed}")
    
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    log.info(f"Loading dataset: {args.dataset_path} ({args.dataset_split})")
    dataset = load_dataset(args.dataset_path, split=args.dataset_split, cache_dir=args.hf_cache)
    
    if args.subset is not None:
        log.info(f"Using subset: first {args.subset} samples")
        dataset = dataset.select(range(min(args.subset, len(dataset))))

    log.info(f"Loading model: {args.model_path}")
    model = load_model(args.model_path, args.device)
    for prm_model_path in args.prm_model_path:
        print(f'Running {prm_model_path}')
        prm_name = prm_model_path.split('/')[-1]
        stat_calculators = [
            StepsExtractor(progress_bar=False),
            load_prm_calculator_by_model_path(
                model_path=prm_model_path,
                device=args.device,
                scores_key=prm_name,  # save score under PRM name
            )
        ]
        estimators = [PRMEstimator(scores_key=prm_name)]
        update_bestofn(
            dataset, model, estimators, stat_calculators,
            args.save_path, save_frequency=None,
            verbose=False,
        )
        print(f'Done running {prm_model_path}')


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
