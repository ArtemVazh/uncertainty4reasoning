from datasets import Dataset, DatasetDict, load_dataset, concatenate_datasets
import argparse
import random
from transformers import AutoTokenizer
import numpy as np
random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, nargs='+', required=True, help="Path to load dataset from.")
    parser.add_argument("--response-length", type=int, required=True, help="Length of the response to truncate to.")
    parser.add_argument("--prompt-file", type=str, default="configs/qwen3_prompt_thinking.txt", help="Path to the prompt file.")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen3-1.7B", help="Path to the model to use for truncation.")
    parser.add_argument("--hf-save-path", type=str, default=None, help="Path to save the dataset to HuggingFace.")
    parser.add_argument("--stats", action='store_true', help="Whether to print the statistics of the dataset.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    datasets = [load_dataset(path)['train'] for path in args.dataset_path]
    dataset = concatenate_datasets(datasets)
    
    claim_col = dataset["claims"]
    label_col = dataset["verified"]
    input_ids_col = dataset["input_ids"]
    questions_col = dataset["question"]
    answers_col = dataset["answer"]
    reply_col = dataset["reply"]
    original_index_col = dataset["original_index"]

    if args.stats:
        label_nums = []
        num_positives = []
        has_positive = 0
        for labels in label_col:
            label_nums.append(len(labels))
            num_positives.append(labels.count(1))
            if 1 in labels:
                has_positive += 1
        print(f"Max number of claims: {max(label_nums)}")
        print(f"Min number of claims: {min(label_nums)}")
        print(f"Avg number of claims: {np.mean(label_nums)} (std: {np.std(label_nums)})")
        print(f"Avg number of positives: {np.mean(num_positives)} (std: {np.std(num_positives)})")
        print(f"Percentage of positive: {np.sum(num_positives) / np.sum(label_nums)}")
        print(f"Has positive: {has_positive / len(label_col)}")

    with open(args.prompt_file, 'r') as f:
        prompt = f.read()

    new_input_ids_col = []
    new_claim_col = []
    new_label_col = []

    for i, (question, input_ids) in enumerate(zip(questions_col, input_ids_col)):
        question_tokens = tokenizer(question, return_tensors='pt')['input_ids'][0].tolist()
        output_ids = input_ids[len(question_tokens):]
        if len(output_ids) > args.response_length:
            output_ids = output_ids[:args.response_length]
        new_input_ids_col.append(question_tokens + output_ids)
        

    for claims, labels in zip(claim_col, label_col):
        stop_idx = None
        for i, claim in enumerate(claims):
            if claim["aligned_token_ids"][-1] >= args.response_length:
                stop_idx = i
                break
        if stop_idx is None:
            stop_idx = len(claims)
        if stop_idx == 0:
            print(f"Stop index is 0 for claim {claims}")
        new_claim_col.append(claims[:stop_idx])
        new_label_col.append(labels[:stop_idx])


    new_dataset = Dataset.from_dict({
        "question": questions_col,
        "answer": answers_col,
        "input_ids": new_input_ids_col,
        "reply": reply_col,
        "original_index": original_index_col,
        "claims": new_claim_col,
        "verified": new_label_col,
    })
    new_dataset = DatasetDict({"train": new_dataset})
    if args.stats:
        label_nums = []
        num_positives = []
        has_positive = 0
        for labels in new_label_col:
            label_nums.append(len(labels))
            num_positives.append(labels.count(1))
            if 1 in labels:
                has_positive += 1
        print(f"Max number of claims: {max(label_nums)}")
        print(f"Min number of claims: {min(label_nums)}")
        print(f"Avg number of claims: {np.mean(label_nums)} (std: {np.std(label_nums)})")
        print(f"Avg number of positives: {np.mean(num_positives)} (std: {np.std(num_positives)})")
        print(f"Percentage of positive: {np.sum(num_positives) / np.sum(label_nums)}")
        print(f"Has positive: {has_positive / len(new_label_col)}")
    else:
        new_dataset.push_to_hub(args.hf_save_path)  

