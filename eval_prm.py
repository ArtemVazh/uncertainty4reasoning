import argparse
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from lm_polygraph import UEManager
from utils import extract_steps, extract_questions


def make_step_rewards(logits, token_masks):
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res


def get_rewards(model, tokenizer, question, steps) -> list[float]:
    if len(steps) == 0:
        return []
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"},
    ]
    conversation_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    input_ids = tokenizer.encode(conversation_str, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
    step_sep_id = tokenizer.encode("<extra_0>")[0]
    token_masks = (input_ids == step_sep_id)
    step_reward = make_step_rewards(outputs[0], token_masks)
    return step_reward[0]


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
    prm_tokenizer = AutoTokenizer.from_pretrained(args.prm_model_path, trust_remote_code=True, cache_dir=args.hf_cache)
    prm_model = AutoModel.from_pretrained(
        args.prm_model_path,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=args.hf_cache,
    ).eval()

    man = UEManager.load_from_hub(args.hf_manager_path)
    steps = extract_steps(man, args.base_model_path, args.hf_cache)
    questions = extract_questions(man, args.prompt_file)

    rewards: list[float] = []
    for i in tqdm(range(len(questions)), desc='Evaluating PRM'):
        r = get_rewards(prm_model, prm_tokenizer, questions[i], steps[i])
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
