import argparse
import os
import random
from argparse import Namespace

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer

from synthetic_dataset_generation.run_extract_verify_claims_thinking import (
    extract_tokens_of_reply,
    get_question,
)
from synthetic_dataset_generation.utils.steps_extractor_thinking import (
    StepsExtractorThinking,
)


def select_partition_indices(total_len: int, partition_num: int, partition_idx: int) -> list[int]:
    if partition_num <= 1:
        return list(range(total_len))
    if partition_idx < 0 or partition_idx >= partition_num:
        raise ValueError(f"partition_idx ({partition_idx}) must be in [0, {partition_num})")
    return np.array_split(np.arange(total_len), partition_num)[partition_idx].tolist()


def main(args):
    if os.path.isdir(args.dataset_path):
        dataset = load_from_disk(args.dataset_path)
    else:
        dataset = load_dataset(args.dataset_path)["train"]

    if isinstance(dataset, DatasetDict):
        dataset = dataset["train"]
    
    start_idx = args.start_idx
    end_idx = len(dataset)
    if args.subset is not None:
        end_idx = min(start_idx + args.subset, len(dataset))
        print(f"Using subset of {end_idx - start_idx} samples from index {start_idx} to {end_idx - 1} (exclusive of {end_idx})")
    elif start_idx != 0:
        print(f"Skipping the first {start_idx} samples (processing indices {start_idx} to {end_idx - 1})")

    if start_idx != 0 or args.subset is not None:
        dataset = dataset.select(range(start_idx, end_idx))

    shard_indices = select_partition_indices(len(dataset), args.partition_num, args.partition_idx)
    dataset = dataset.select(shard_indices)
    dataset = dataset.add_column("original_index", shard_indices)

    print("Length of dataset after sampling:", len(dataset))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.hf_cache)
    prompt = open(args.prompt_file, "r").read()

    greedy_tokens = extract_tokens_of_reply(dataset, tokenizer, prompt)
    deps = {
        "greedy_texts": [tokenizer.decode(greedy_token_list) for greedy_token_list in greedy_tokens],
        "greedy_tokens": greedy_tokens,
    }

    print("Extracting claims...")
    claim_extractor = StepsExtractorThinking()
    claims = claim_extractor(deps, dataset["question"], model=Namespace(tokenizer=tokenizer))["claims"]
    claims_for_save = [[claim.__dict__ for claim in claim_group] for claim_group in claims]
    print("Done extracting claims.")

    dataset = dataset.add_column("claims", claims_for_save)

    output_dataset = DatasetDict({"train": dataset})
    output_dataset.save_to_disk(args.save_path)
    output_dataset.push_to_hub(args.hf_save_path, private=False)
    print(f"Pushed extracted claims to {args.hf_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract thinking-style claims into sharded datasets.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to load dataset from.")
    parser.add_argument("--model-path", type=str, required=True, help="Model path for tokenizer.")
    parser.add_argument("--prompt-file", type=str, required=True, help="Path to the prompt file.")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the dataset with extracted claims.")
    parser.add_argument("--hf-save-path", type=str, required=True, help="HuggingFace Hub path to push dataset to.")
    parser.add_argument("--hf-cache", type=str, default=None, help="Cache directory for HuggingFace models.")
    parser.add_argument("--subset", type=int, default=None, help="Number of samples to use from the shard. If not specified, uses the full shard.")
    parser.add_argument("--start-idx", type=int, default=0, help="The starting index (offset) in the shard to begin processing.")
    parser.add_argument("--partition-num", type=int, default=1, help="Number of partitions to split the dataset into (use with slurm array).")
    parser.add_argument("--partition-idx", type=int, default=0, help="Partition index for this run (use with slurm array task id).")

    args = parser.parse_args()
    main(args)
