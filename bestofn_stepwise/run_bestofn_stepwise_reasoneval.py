import argparse
import torch

from bestofn.run_bestofn_uhead import load_model
from bestofn_stepwise.stat_calculators.stepwise_reasoneval_minimization import StepwiseReasonEvalMinimizationCalculator
from bestofn_stepwise.run_bestofn_stepwise_uhead import update_stats


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt-file', type=str, required=True, help="Path to prompt template file")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the output .torch")
    parser.add_argument("--hf-cache", type=str, default=None, help="Path to HuggingFace cache directory")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-1.7B", help="Model name or path for generation")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (e.g., 'cuda' or 'cpu')")
    return parser


def main(args):
    model = load_model(args.model_path, args.device)
    stats: list[dict] = torch.load(args.save_path, weights_only=False)
    update_stats(stats, [StepwiseReasonEvalMinimizationCalculator(args.prompt_file)], model, args.save_path)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
