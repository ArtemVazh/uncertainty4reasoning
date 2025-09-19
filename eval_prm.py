import argparse
from tqdm import tqdm

from baselines.prm import load_prm_calculator_by_model_path
from lm_polygraph import UEManager
from utils import extract_steps, extract_questions


def get_parser():
    parser = argparse.ArgumentParser(description="Reward extraction using PRM model.")

    parser.add_argument('--hf-manager-path', type=str, required=True, help="HuggingFace repo for the UE manager file")
    parser.add_argument('--base-model-path', type=str, required=True, help="Path or name of the base model")
    parser.add_argument('--hf-save-path', type=str, default=None,
                        help="Path to save manager with rewards, default: same as hf-manager-path")
    parser.add_argument('--prm-model-path', type=str, nargs='+', default=[
        "Qwen/Qwen2.5-Math-7B-PRM800K",
        "Qwen/Qwen2.5-Math-PRM-7B",
        "peiyi9979/math-shepherd-mistral-7b-prm",
        "RLHFlow/Llama3.1-8B-PRM-Mistral-Data",
        "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
        "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",  # loads slow (can take up to 15 mins)
        "GenPRM/GenPRM-1.5B-simple",
        # "GenPRM/GenPRM-1.5B",  # very slow
        "RLHFlow/Llama3.1-8B-PRM-Mistral-Data",
        "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
        "universalprm/Universal-PRM",
        "HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2",
    ], help="Path(s) or name(s) of the PRM model(s)")
    parser.add_argument('--device', type=str, default="auto", help="Device map setting for model loading")
    parser.add_argument('--prompt-file', type=str, default="configs/gsm8k_3shot_prompt.txt",
                        help="Path to prompt template file")
    parser.add_argument('--hf-cache', type=str, default=None, help="Cache directory for HF models")

    return parser


def main(args):
    print('Evaluating following PRMs:')
    for prm_model_path in args.prm_model_path:
        print(f' - {prm_model_path}')
    for prm_model_path in args.prm_model_path:
        prm = load_prm_calculator_by_model_path(model_path=prm_model_path, device=args.device)
        prm.init()

        man = UEManager.load_from_hub(args.hf_manager_path)
        steps = extract_steps(man, args.base_model_path, args.hf_cache)
        questions = extract_questions(man, args.prompt_file)
        if len(steps) < len(questions):
            questions = questions[:len(steps)]

        rewards: list[float] = []
        for i in tqdm(range(len(questions)), desc=f'Evaluating {prm_model_path}'):
            r = prm.get_rewards(questions[i], steps[i])
            if len(r) != len(steps[i]):
                print('Question:', questions[i])
                print('Steps:', '\n'.join(s.claim_text.strip() for s in steps[i]))
                print('rewards:', r)
            assert len(r) == len(steps[i])
            rewards += r

        if args.hf_save_path is None:
            args.hf_save_path = args.hf_manager_path
        # higher values indicate higher uncertainty
        man.estimations['claim', f'PRM_{prm_model_path}'] = [-r for r in rewards]
        man.push_to_hub(args.hf_save_path)
        print('Saved to {}'.format(args.hf_save_path))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
