import numpy as np
import argparse
import random

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


def generate_targets(dataset, reply_tokens_all, key="verified"):
    targets = []
    for idx in range(len(dataset)):
        reply_tokens = reply_tokens_all[idx]
        claims = dataset["claims"][idx]
        verified = dataset[key][idx]
        target = [-100.] * len(reply_tokens)
        for claim, label in zip(claims, verified):
            for t in claim["aligned_token_ids"]:
                if not np.isnan(label):
                    target[t] = float(label == 1.0)
        targets.append(target)
    return targets


def main(args):
    dataset = load_from_disk(args.dataset_path)
    if args.sample > 0:
        unique_questions = []
        for q in dataset["question"]:
            if q not in unique_questions:
                unique_questions.append(q)
        random.seed(42)
        unique_questions = random.sample(unique_questions, args.sample)
        dataset = dataset.filter(lambda x: x["question"] in unique_questions)
        print(unique_questions)

    print("Length of dataset after sampling:", len(dataset))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.hf_cache)
    prompt = open(args.prompt_file, 'r').read()

    greedy_tokens = extract_tokens_of_reply(dataset, tokenizer, prompt)
    deps = {"greedy_texts": dataset["reply"], "greedy_tokens": greedy_tokens}

    print('Extracting claims...')
    claim_extractor = StepsExtractor(skip_starts=['Reasoning Steps:', '<start', '<end'])
    claims = claim_extractor(deps, dataset["question"], model=Namespace(tokenizer=tokenizer))["claims"]
    print("Done.")

    print("Verifying claims...")
    if 'gsm8k' in args.dataset_path:
        stats = {"input_texts": dataset["question"], "claims": claims, "answers": dataset["answer"]}
    elif 'strategy_qa' in args.dataset_path:
        def parse_strategy_qa_answer(dataset):
            parsed_answers = []
            for answer, facts in zip(dataset["answer"], dataset["facts"]):
                answer_str = ""
                if answer:
                    answer_str += "Yes, according to the facts: "
                else:
                    answer_str += "No, according to the facts: "
                for i, fact in enumerate(facts):
                    answer_str += f"{i+1}. {fact} "
                parsed_answers.append(answer_str)
            return parsed_answers
        stats = {"input_texts": dataset["question"], "claims": claims, "answers": parse_strategy_qa_answer(dataset)}
    elif 'science_qa' in args.dataset_path:
        def parse_science_qa_answer(dataset):
            parsed_answers = []
            for answer, solution in zip(dataset["answer"], dataset["solution"]):
                answer_str = f"The answer is {answer}. Reasoning: {solution}"
                parsed_answers.append(answer_str)
            return parsed_answers
        stats = {"input_texts": dataset["question"], "claims": claims, "answers": parse_science_qa_answer(dataset)}
    elif 'plan' in args.dataset_path:
        stats = {"input_texts": dataset['prompt_0shot'], "claims": claims, "answers": dataset['golden_plan']}
    else:
        raise ValueError(f"Dataset path {args.dataset_path} not supported")

    api_key = open(args.api_key_file, 'r').read()
    fact_checker_correctness = StepFactCheck(
        prompt_file=args.prompt_file,
        api_key=api_key,
        n_threads=args.n_threads,
        cache_path=args.api_cache if args.api_cache is not None else args.hf_cache,
        label_type="correctness",
    )
    fact_checker_informativeness = StepFactCheck(
        prompt_file=args.prompt_file,
        api_key=api_key,
        n_threads=args.n_threads,
        cache_path=args.api_cache if args.api_cache is not None else args.hf_cache,
        label_type="informativeness",
    )
    correctness_labels = fact_checker_correctness(stats, None)
    informativeness_labels = fact_checker_informativeness(stats, None)
    print("Done.")

    print("Generating targets...")
    result = dataset.to_dict()
    result.update({
        "claims": [[claim.__dict__ for claim in e] for e in claims],
        "verified": correctness_labels,
        "informativeness": informativeness_labels,
    })
    new_dataset = Dataset.from_dict(result)
    result["uncertainty_labels"] = generate_targets(new_dataset, greedy_tokens, key="verified")
    result["informativeness_labels"] = generate_targets(new_dataset, greedy_tokens, key="informativeness")
    print("Done.")

    print(f"Saving data to: {args.save_path}")
    anno_dataset = Dataset.from_dict(result)
    anno_dataset.save_to_disk(args.save_path)
    print("Done.")

    if args.hf_save_path is not None:
        anno_dataset.push_to_hub(args.hf_save_path)

    print_stats(anno_dataset, key="verified")
    print_stats(anno_dataset, key="informativeness")


def print_stats(anno_dataset, key="verified"):
    all_ue = []
    for d in anno_dataset:
        all_ue += d[key]
    print('Total:', len(all_ue), 'steps')
    t, f = all_ue.count(0.0), all_ue.count(1.0)
    print('{} True: {} steps ({}%)'.format(key, t, round(100 * t / (t + f), 2)))
    print('{} False: {} steps ({}%)'.format(key, f, round(100 * f / (t + f), 2)))


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
    parser.add_argument("--sample", type=int, default=-1, help="Sampling for debugging.")
    parser.add_argument("--api-cache", type=str, default=None, help="Cache directory for API calls.")

    args = parser.parse_args()
    main(args)