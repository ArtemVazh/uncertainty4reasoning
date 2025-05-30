import argparse
import os
import logging

from datasets import load_dataset

from bestofn.estimators.uhead import UHeadEstimator
from bestofn.stat_calculators.load_stat_calculators import load_relevant_stat_calculators
from lm_polygraph import WhiteboxModel
from lm_polygraph.estimators import MaximumSequenceProbability, PTrue, Perplexity, MeanTokenEntropy, \
    ClaimConditionedProbability
from bestofn.bestofn_utils import bestofn
from configs.load_qwen import load_model as load_qwen_model, load_tokenizer as load_qwen_tokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bestofn_eval")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Dataset to evaluate on (HuggingFace name or local path)")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split (e.g., test)")
    parser.add_argument('--prompt-file', type=str, required=True, help="Path to prompt template file")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the output .torch")
    parser.add_argument("--save-frequency", type=int, default=1, help="Save every n iterations")
    parser.add_argument("--hf-cache", type=str, default=None, help="Path to HuggingFace cache directory")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-1.7B", help="Model name or path for generation")
    parser.add_argument("--uhead-path", type=str, default="rediska0123/uhead_Qwen3-1.7B_gsm8k", help="UHead HF path")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--n", type=int, default=10, help="Number of completions to generate per input")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation max_new_tokens")
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

    log.info(f"Loading estimators")
    # sequence-level LM-Polygraph estimators
    estimators = [
        # No reason to calculate Eccentricity/LexicalSimilarity/SemanticEntropy etc
        # because they treat each sample the same
        # TODO: Maybe add Dissimilarity?
        MaximumSequenceProbability(),
        MeanTokenEntropy(),
        Perplexity(),
        PTrue(),
        # ClaimConditionedProbability(),  # TODO: calculate (too computationally intensive)
        UHeadEstimator(),
    ]

    log.info(f"Loading stat calculators")
    stat_calculators = load_relevant_stat_calculators(
        estimators, model, args.uhead_path,
        prompt_path=args.prompt_file,
        hf_cache=args.hf_cache,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    log.info(f"Loaded {len(stat_calculators)} stat calculators:\n" + "\n".join(f" - {s}" for s in stat_calculators))

    bestofn(
        dataset, model, estimators, stat_calculators,
        args.save_path, args.save_frequency,
        args.n, args.max_new_tokens,
    )


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
