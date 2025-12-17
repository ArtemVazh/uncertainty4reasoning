#!/usr/bin/env python3
"""
Compute basic statistics for datasets with extracted claims.

Given a Hugging Face dataset path (local directory created via `save_to_disk`
or a hub repo id), this script reports:
 - Average number of claims per row.
 - Average token-length of claims (using the `aligned_token_ids` field).

Example:
    python -m synthetic_dataset_generation.claim_dataset_stats \\
        --dataset-path path/to/dataset \\
        --split train
"""

import argparse
import os
from typing import Any, Dict, Optional, Union

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk


def _load_dataset(path: str) -> Union[Dataset, DatasetDict]:
    if os.path.isdir(path):
        return load_from_disk(path)
    return load_dataset(path)


def _pick_split(dataset: Union[Dataset, DatasetDict], split: Optional[str]) -> Dataset:
    if isinstance(dataset, DatasetDict):
        if split:
            if split not in dataset:
                raise ValueError(f"Split '{split}' not found in dataset. Available splits: {list(dataset.keys())}")
            return dataset[split]
        # No split specified: merge all splits so stats cover the full dataset.
        return concatenate_datasets(list(dataset.values()))
    if split:
        raise ValueError("Split provided but dataset is not a DatasetDict.")
    return dataset


def _claim_token_length(claim: Any) -> Optional[int]:
    if isinstance(claim, dict):
        tokens = claim.get("aligned_token_ids")
    else:
        tokens = getattr(claim, "aligned_token_ids", None)
    if tokens is None:
        return None
    return len(tokens)


def compute_statistics(dataset: Dataset) -> Dict[str, Union[float, int]]:
    if "claims" not in dataset.column_names:
        raise ValueError("Dataset does not contain a 'claims' column.")

    total_rows = len(dataset)
    total_claims = 0
    total_token_length = 0
    claims_with_tokens = 0
    claims_missing_tokens = 0

    for claims in dataset["claims"]:
        if claims is None:
            continue
        total_claims += len(claims)
        for claim in claims:
            token_length = _claim_token_length(claim)
            if token_length is None:
                claims_missing_tokens += 1
                continue
            total_token_length += token_length
            claims_with_tokens += 1

    avg_claims_per_row = total_claims / total_rows if total_rows else 0.0
    avg_token_length = total_token_length / claims_with_tokens if claims_with_tokens else 0.0

    return {
        "rows": total_rows,
        "total_claims": total_claims,
        "avg_claims_per_row": avg_claims_per_row,
        "avg_token_length": avg_token_length,
        "claims_with_tokens": claims_with_tokens,
        "claims_missing_tokens": claims_missing_tokens,
    }


def main(args: argparse.Namespace) -> None:
    raw_dataset = _load_dataset(args.dataset_path)
    dataset = _pick_split(raw_dataset, args.split)
    stats = compute_statistics(dataset)

    print(f"Dataset rows: {stats['rows']}")
    print(f"Total claims: {stats['total_claims']}")
    print(f"Average claims per row: {stats['avg_claims_per_row']:.3f}")
    print(f"Average token-length per claim: {stats['avg_token_length']:.3f}")
    if stats["claims_missing_tokens"]:
        print(f"Claims missing token alignment: {stats['claims_missing_tokens']} (skipped in token-length average)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute statistics for extracted-claim datasets.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path or repo id of the dataset.")
    parser.add_argument("--split", type=str, default=None, help="Optional split name (merge all splits when omitted).")
    args = parser.parse_args()
    main(args)
