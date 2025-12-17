from datasets import Dataset, DatasetDict, load_dataset
import argparse
import random

random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to load dataset from.")
    parser.add_argument("--claim-num", type=int, required=True, help="Number of claims to sample.")
    parser.add_argument("--random", action="store_true", default=False, help="Randomly sample claims.")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path)['train']
    claim_col = dataset["claims"]
    label_col = dataset["verified"]

    data_dict = {k: v for k, v in dataset.to_dict().items() if k not in ["claims", "verified"]}

    new_claim_col = []
    new_label_col = []
    for claims, labels in zip(claim_col, label_col):
        assert len(claims) == len(labels)
        if args.random:
            random_idx = random.sample(range(len(claims)), min(args.claim_num, len(claims)))
            random_idx.sort()
            new_claim_col.append([claims[i] for i in random_idx])
            new_label_col.append([labels[i] for i in random_idx])
        else:
            new_claim_col.append(claims[:args.claim_num])
            new_label_col.append(labels[:args.claim_num])

    data_dict["claims"] = new_claim_col
    data_dict["verified"] = new_label_col
    new_dataset = Dataset.from_dict(data_dict)
    new_dataset = DatasetDict({"train": new_dataset})
    if args.random:
        save_name = f"{args.dataset_path}_{args.claim_num}_random"
    else:
        save_name = f"{args.dataset_path}_{args.claim_num}"
    new_dataset.push_to_hub(save_name)  

