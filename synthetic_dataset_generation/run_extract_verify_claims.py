import numpy as np
import argparse

from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from argparse import Namespace
from synthetic_dataset_generation.utils.steps_extractor import StepsExtractor
from synthetic_dataset_generation.utils.step_fact_check import StepFactCheck


def get_question(dataset, i, prompt):
    return prompt.format(q=dataset[i]["question"])


def extract_tokens_of_reply(dataset, tokenizer, prompt):
    greedy_tokens = []
    inpt_ids = dataset["input_ids"]
    for i in range(len(dataset)):
        question = get_question(dataset, i, prompt)
        question_tokens = tokenizer(question, return_tensors='pt')['input_ids'][0]
        greedy_tokens.append(inpt_ids[i][len(question_tokens):])
    return greedy_tokens


def generate_targets(dataset, reply_tokens_all):
    targets = []
    for idx in range(len(dataset)):
        reply_tokens = reply_tokens_all[idx]
        claims = dataset["claims"][idx]
        verified = dataset["verified"][idx]
        target = [-100.] * len(reply_tokens)
        for claim, label in zip(claims, verified):
            for t in claim["aligned_token_ids"]:
                if not np.isnan(label):
                    target[t] = float(label == 1.0)
        targets.append(target)
    return targets


def main(args):
    dataset = load_from_disk(args.dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.hf_cache)
    prompt = open(args.prompt_file, 'r').read()

    greedy_tokens = extract_tokens_of_reply(dataset, tokenizer, prompt)
    deps = {"greedy_texts": dataset["reply"], "greedy_tokens": greedy_tokens}

    print('Extracting claims...')
    claim_extractor = StepsExtractor()
    claims = claim_extractor(deps, dataset["question"], model=Namespace(tokenizer=tokenizer))["claims"]
    print("Done.")

    print("Verifying claims...")
    stats = {"input_texts": dataset["question"], "claims": claims, "answers": dataset["answer"]}
    api_key = open(args.api_key_file, 'r').read()
    fact_checker = StepFactCheck(
        prompt_file=args.prompt_file,
        api_key=api_key,
        n_threads=args.n_threads,
        cache_path=args.hf_cache,
    )
    verified = fact_checker(stats, None)
    print("Done.")

    print("Generating targets...")
    result = dataset.to_dict()
    result.update({
        "claims": [[claim.__dict__ for claim in e] for e in claims],
        "verified": verified,
    })
    new_dataset = Dataset.from_dict(result)
    result["uncertainty_labels"] = generate_targets(new_dataset, greedy_tokens)
    print("Done.")

    print(f"Saving data to: {args.save_path}")
    anno_dataset = Dataset.from_dict(result)
    anno_dataset.save_to_disk(args.save_path)
    print("Done.")

    if args.hf_save_path is not None:
        anno_dataset.push_to_hub(args.hf_save_path)

    print_stats(anno_dataset)


def print_stats(anno_dataset):
    all_ue = []
    for d in anno_dataset:
        all_ue += d['verified']
    print('Total:', len(all_ue), 'steps')
    t, f = all_ue.count(0.0), all_ue.count(1.0)
    print('True: {} steps ({}%)'.format(t, round(100 * t / (t + f), 2)))
    print('False: {} steps ({}%)'.format(f, round(100 * f / (t + f), 2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate annotated synthetic dataset.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to load dataset from.")
    parser.add_argument("--model-path", type=str, required=True, help="Model path for tokenizer.")
    parser.add_argument("--prompt-file", type=str, required=True, help="Path to the prompt file.")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the annotated dataset.")
    parser.add_argument("--hf-cache", type=str, default=None, help="Cache directory for HuggingFace models.")
    parser.add_argument("--api-key-file", type=str, default="configs/deepseek_api_key.txt",
                        help="Path to file containing OpenAI API key.")
    parser.add_argument("--hf-save-path", type=str, default=None, help="HuggingFace Hub path to push dataset to.")
    parser.add_argument("--n-threads", type=int, default=1, help="Number of threads for fact checking.")

    args = parser.parse_args()
    main(args)
