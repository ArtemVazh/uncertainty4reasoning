import numpy as np
import argparse
import random
import pickle
from tqdm import tqdm
import os
from datasets import load_from_disk, Dataset, load_dataset, DatasetDict
from transformers import AutoTokenizer
from argparse import Namespace
from synthetic_dataset_generation.utils.steps_extractor_factuality import StepsExtractorFactuality
from synthetic_dataset_generation.utils.step_fact_check_factuality import StepFactCheckFactuality


def get_question(dataset, i, prompt):
    return prompt.format(q=dataset[i]["question"])


def extract_tokens_of_reply(dataset, tokenizer, prompt):
    greedy_tokens = []
    inpt_ids = dataset["input_ids"]
    for i in tqdm(range(len(dataset)), desc="Extracting tokens"):
        question = get_question(dataset, i, prompt)
        question_tokens = tokenizer(question, return_tensors='pt')['input_ids'][0]
        greedy_tokens.append(inpt_ids[i][len(question_tokens):])
    return greedy_tokens


def generate_targets(dataset, reply_tokens_all, key="verified"):
    targets = []
    for idx in tqdm(range(len(dataset)), desc=f"Generating {key} targets"):
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
    if os.path.isdir(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path)['train']

    if args.factuality_csv is None:
        args.factuality_csv = f"{args.save_path}_factuality.csv"
    elif not args.factuality_csv.endswith(".csv"):
        args.factuality_csv += ".csv"

    # Partition the dataset
    partition_size = len(dataset) // args.partition_num
    dataset = dataset.select(range(args.partition_idx * partition_size, (args.partition_idx + 1) * partition_size))
    
    # Apply start index and subset if specified
    start_idx = args.start_idx
    end_idx = len(dataset)

    # Determine the end index based on subset size if provided
    if args.subset is not None:
        end_idx = min(start_idx + args.subset, len(dataset))
        print(f"Using subset of {end_idx - start_idx} samples from index {start_idx} to {end_idx - 1} (exclusive of {end_idx})")
    elif start_idx != 0:
        print(f"Skipping the first {start_idx} samples (processing indices {start_idx} to {end_idx - 1})")

    # Only perform selection when needed to avoid unnecessary dataset copy
    if start_idx != 0 or args.subset is not None:
        dataset = dataset.select(range(start_idx, end_idx))
    
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

    if os.path.exists(args.save_path + '_claims.pkl'):
        print("Loading claims from cache...")
        with open(args.save_path + '_claims.pkl', 'rb') as f:
            claims = pickle.load(f)
    else:
        print("Extracting claims...")
        claim_extractor = StepsExtractorFactuality()
        claims = claim_extractor(deps, dataset["question"], model=Namespace(tokenizer=tokenizer))["claims"]
        with open(args.save_path +'_claims.pkl', 'wb') as f:
            pickle.dump(claims, f)
            print(f"Claims saved to {args.save_path + '_claims.pkl'}")
        print(claims[0][0].claim_text)
    print("Done.")

    print("Verifying claims...")
    stats = {"input_texts": dataset["question"], "claims": claims, "answers": dataset["answer"]}
        # raise ValueError(f"Dataset path {args.dataset_path} not supported")

    fact_checker_correctness = StepFactCheckFactuality(
        fact_check_prompt=args.fact_check_prompt,
        prompt_file=args.prompt_file,
        cache_path=args.api_cache if args.api_cache is not None else args.hf_cache,
        model=args.anno_model,
        api_key=args.fact_check_api_key,
        progress_bar=not args.debug,
        n_threads=args.n_threads,
        base_url=args.fact_check_base_url,
        debug=args.debug,
    )
    correctness_labels = fact_checker_correctness(stats, None, args.factuality_csv)
    print("Done.")

    print("Generating targets...")
    result = dataset.to_dict()
    result.update({
        "claims": [[claim.__dict__ for claim in e] for e in claims[:len(correctness_labels)]],
        "verified": correctness_labels,
    })
    new_dataset = DatasetDict({"train": Dataset.from_dict(result)})
    print('finish making new dataset')
    new_dataset.save_to_disk(args.save_path)
    if args.hf_save_path is not None:
        new_dataset.push_to_hub(args.hf_save_path)
    print("Done.")


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
    parser.add_argument("--fact-check-prompt", type=str, required=True, help="Path to the fact check prompt file.")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the annotated dataset.")
    parser.add_argument("--hf-cache", type=str, default=None, help="Cache directory for HuggingFace models.")
    parser.add_argument("--anno-model", type=str, default="deepseek-reasoner")
    parser.add_argument("--hf-save-path", type=str, default=None, help="HuggingFace Hub path to push dataset to.")
    parser.add_argument("--n-threads", type=int, default=1, help="Number of threads for fact checking.")
    parser.add_argument("--subset", type=int, default=None, help="Number of samples to use from the dataset. If not specified, uses the full dataset.")
    parser.add_argument("--start-idx", type=int, default=0, help="The starting index (offset) in the dataset to begin processing. Use this to skip already processed samples.")
    parser.add_argument("--partition-num", type=int, default=1, help="Number of partitions.")
    parser.add_argument("--partition-idx", type=int, default=0, help="Partition index.")
    parser.add_argument("--sample", type=int, default=-1, help="Sampling for debugging.")
    parser.add_argument("--api-cache", type=str, default=None, help="Cache directory for API calls.")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument("--factuality-csv", type=str, default=None, help="Path to store intermediate factuality CSV (defaults to '<save-path>_factuality.csv').")
    parser.add_argument("--fact-check-base-url", type=str, default="http://localhost:8000/v1", help="Set it to None to use OpenAI API instead of local vLLM.")
    parser.add_argument("--fact-check-api-key", type=str, default="EMPTY", help="API key for remote factuality checker (if required).")
    args = parser.parse_args()
    main(args)