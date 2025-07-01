import argparse
from tqdm import tqdm

from baselines.prm import PRMStatCalculator
from lm_polygraph import UEManager
from utils import extract_steps, extract_questions


def get_parser():
    parser = argparse.ArgumentParser(description="Reward extraction using PRM model.")

    parser.add_argument('--hf-manager-path', type=str, required=True, help="HuggingFace repo for the UE manager file")
    parser.add_argument('--base-model-path', type=str, required=True, help="Path or name of the base model")
    parser.add_argument('--hf-save-path', type=str, default=None,
                        help="Path to save manager with rewards, default: same as hf-manager-path")
    parser.add_argument('--prm-model-path', type=str, default="Qwen/Qwen2.5-Math-7B-PRM800K",
                        help="Path or name of the PRM model")
    parser.add_argument('--device', type=str, default="auto", help="Device map setting for model loading")
    parser.add_argument('--prompt-file', type=str, default="configs/gsm8k_3shot_prompt.txt",
                        help="Path to prompt template file")
    parser.add_argument('--hf-cache', type=str, default=None, help="Cache directory for HF models")

    return parser


def main(args):
    prm = PRMStatCalculator(model_path=args.prm_model_path, device=args.device)

    man = UEManager.load_from_hub(args.hf_manager_path)
    steps = extract_steps(man, args.base_model_path, args.hf_cache)
    questions = extract_questions(man, args.prompt_file)

    rewards: list[float] = []
    for i in tqdm(range(len(questions)), desc='Evaluating PRM'):
        r = prm.get_rewards(questions[i], steps[i])
        assert len(r) == len(steps[i])
        rewards += r

    if args.hf_save_path is None:
        args.hf_save_path = args.hf_manager_path
    # higher values indicate higher uncertainty
    man.estimations['claim', f'PRM_{args.prm_model_path}'] = [-r for r in rewards]
    man.push_to_hub(args.hf_save_path)
    print('Saved to {}'.format(args.hf_save_path))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
