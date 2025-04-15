import torch
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset
from functools import partial


def parse_ans(s, ignore_unfinished=False):
    if '####' in s:
        return float(s.split('####')[-1].replace(',', ''))
    if r'\boxed{' in s:
        x = s.split(r'\boxed{')[-1].split('}')[0].replace(',', '')
        x = x.split('=')[-1]
        if x.endswith('%'):
            x = x[:-1]
        try:
            return float(x)
        except:
            return None
    if not ignore_unfinished:
        print(f'Couldnt parse answer from:\n{s}')
    return None


def generate_replies(inst, prompt):
    question = prompt.format(q=inst["question"])
    inputs = tokenizer(question, return_tensors='pt')['input_ids']
    inputs = inputs.to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            num_return_sequences=1,
            generation_config=generation_config,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.,
            diversity_penalty=0.,
            length_penalty=1.,
            stop_strings=['\n\n', '}\n'],
            tokenizer=tokenizer,
        )
    reply = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    inst["input_ids"] = outputs[0]
    inst["reply"] = reply
    return inst


def print_stats(dataset):
    accuracies = []
    finished = []
    for a, d in zip(dataset['answer'], dataset['reply']):
        gt_ans = parse_ans(a, True)
        llm_ans = parse_ans(d, True)
        accuracies.append(gt_ans == llm_ans if llm_ans is not None else 0)
        finished.append(llm_ans is not None)
    print('Accuracy:', np.mean(accuracies))
    print('Finished:', np.mean(finished))


def parse_tuple(s):
    try:
        parts = s.strip("()").split(",")
        return tuple(part.strip() for part in parts)
    except Exception:
        raise argparse.ArgumentTypeError("Tuple must be in the form: value1,value2")


def parse_args():
    parser = argparse.ArgumentParser(description="Create generation texts for model.")
    parser.add_argument('--dataset-path', type=parse_tuple, default=("openai/gsm8k", "main"),
                        help='Path to the dataset as a tuple, e.g. "openai/gsm9k,main"')
    parser.add_argument('--dataset-split', type=parse_tuple, default="test", help='Dataset split')
    parser.add_argument('--n-samples', type=int, required=True, help='Number of samples to evaluate from the dataset')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--prompt-file', type=str, required=True, help='Path to the prompt text file')
    parser.add_argument('--save-path', type=str, required=True, help='Path to save the processed dataset')
    parser.add_argument('--hf-cache', type=str, default=None, help='Path to the HuggingFace cache directory')
    parser.add_argument('--device', type=str, default="auto", help='Device to infer model on')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, device_map=args.device, trust_remote_code=True, cache_dir=args.hf_cache)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.hf_cache)
    generation_config = GenerationConfig.from_pretrained(args.model_path)

    prompt = open(args.prompt_file, 'r').read()

    dataset = load_dataset(*args.dataset_path, cache_dir=args.hf_cache)['test']
    dataset = dataset.select(range(args.n_samples))
    dataset = dataset.map(partial(generate_replies, prompt=prompt))
    print_stats(dataset)

    dataset.save_to_disk(args.save_path)
    print("Done.")
