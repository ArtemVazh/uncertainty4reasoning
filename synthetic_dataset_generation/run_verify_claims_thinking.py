import argparse
import os
from typing import List

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from lm_polygraph.stat_calculators.extract_claims import Claim
from synthetic_dataset_generation.utils.step_fact_check_thinking_concurrent import StepFactCheckThinking


def load_extracted_claim_shards(claim_path_prefix: str, shards_to_load: List[int]):
    shards = []
    for shard_idx in shards_to_load:
        path = f"{claim_path_prefix}_{shard_idx}"
        try:
            shard = load_from_disk(path)
        except Exception as e:
            print(f"Error loading shard {path} from local directory, trying to load from HuggingFace Hub...")
            shard = load_dataset(path)
        if isinstance(shard, DatasetDict):
            shard = shard["train"]
        shards.append(shard)
    merged = shards[0] if len(shards) == 1 else concatenate_datasets(shards)
    if "original_index" in merged.column_names:
        merged = merged.sort("original_index")
    return merged


def deserialize_claims(claim_records: list[list]) -> list[list[Claim]]:
    claims = []
    for claim_set in claim_records:
        claims.append([c if isinstance(c, Claim) else Claim(**c) for c in claim_set])
    return claims


def main(args):
    if not args.extracted_claims_path_prefix or not args.shards_to_load:
        raise ValueError("--extracted-claims-path-prefix is required for verification-only mode.")

    dataset = load_extracted_claim_shards(args.extracted_claims_path_prefix, args.shards_to_load)

    start_idx = args.start_idx
    end_idx = len(dataset)
    if args.subset is not None:
        end_idx = min(start_idx + args.subset, len(dataset))
        print(f"Using subset of {end_idx - start_idx} samples from index {start_idx} to {end_idx - 1} (exclusive of {end_idx})")
    elif start_idx != 0:
        print(f"Skipping the first {start_idx} samples (processing indices {start_idx} to {end_idx - 1})")
    if start_idx != 0 or args.subset is not None:
        dataset = dataset.select(range(start_idx, end_idx))

    claims_for_save = dataset["claims"]
    claims_for_verification = deserialize_claims(claims_for_save)

    print(f"Verifying {len(dataset)} samples...")
    stats = {"input_texts": dataset["question"], "claims": claims_for_verification, "answers": dataset["answer"]}

    fact_checker_correctness = StepFactCheckThinking(
        model=args.anno_model,
        prompt_file=args.prompt_file,
        api_key=args.fact_check_api_key,
        base_url=args.fact_check_base_url,
        n_threads=args.n_threads,
        cache_path=args.api_cache if args.api_cache is not None else args.hf_cache,
        debug=args.debug,
    )
    correctness_labels = fact_checker_correctness(stats, None, output_path=args.save_path + '_details.csv')
    print("Verification done.")

    result = dataset.to_dict()
    result.update({
        "claims": claims_for_save[:len(correctness_labels)],
        "verified": correctness_labels,
    })
    new_dataset = DatasetDict({"train": Dataset.from_dict(result)})
    new_dataset.save_to_disk(args.save_path)
    if args.hf_save_path is not None:
        new_dataset.push_to_hub(args.hf_save_path)
    print(f"Saved annotated dataset to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify pre-extracted thinking claims.")
    parser.add_argument("--shards-to-load", type=int, nargs="+", required=True, help="Index of shards to load.")
    parser.add_argument("--prompt-file", type=str, required=True, help="Path to the prompt file.")
    parser.add_argument("--save-path", type=str, required=True, help="Path to save the annotated dataset.")
    parser.add_argument("--extracted-claims-path-prefix", type=str, required=True, help="Path prefix to list of shards with pre-extracted claims.")
    parser.add_argument("--hf-cache", type=str, default=None, help="Cache directory for HuggingFace models.")
    parser.add_argument("--anno-model", type=str, default="deepseek-reasoner")
    parser.add_argument("--hf-save-path", type=str, default=None, help="HuggingFace Hub path to push dataset to.")
    parser.add_argument("--n-threads", type=int, default=1, help="Number of threads for fact checking.")
    parser.add_argument("--subset", type=int, default=None, help="Number of samples to use from the loaded shard(s).")
    parser.add_argument("--start-idx", type=int, default=0, help="The starting index to begin processing.")
    parser.add_argument("--api-cache", type=str, default=None, help="Cache directory for API calls.")
    parser.add_argument("--debug", action="store_true", help="Debug mode.")
    parser.add_argument("--fact-check-base-url", type=str, default="", help="Set it to None to use OpenAI API instead of local vLLM.")
    parser.add_argument("--fact-check-api-key", type=str, default="EMPTY", help="API key for remote factuality checker (if required).")
    args = parser.parse_args()
    main(args)
