import argparse
import os
import logging

from datasets import load_dataset

from baselines.prm import PRMStatCalculator
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
    return parser


def load_model(model_path, device):
    tokenizer = load_qwen_tokenizer(model_path)
    base_model = load_qwen_model(model_path, device)
    base_model.eval()
    return WhiteboxModel(base_model, tokenizer)


def main(args):
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    log.info(f"Loading dataset: {args.dataset_path} ({args.dataset_split})")
    dataset = load_dataset(args.dataset_path, split=args.dataset_split, cache_dir=args.hf_cache)

    log.info(f"Loading model: {args.model_path}")
    model = load_model(args.model_path, args.device)

    estimators = [
        PRMEstimator(),
    ]
    stat_calculators = [
        StepsExtractor(progress_bar=False),
        PRMStatCalculator(args.prompt_file),
    ]

    update_bestofn(
        dataset, model, estimators, stat_calculators,
        args.save_path, save_frequency=None,
        verbose=False,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
