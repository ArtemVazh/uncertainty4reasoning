from datasets import load_from_disk, concatenate_datasets, DatasetDict
from datasets import Sequence, Value


dataset_name = "train_prm800k_Phi-4-reasoning-plus_finished_self_annotate"
huggingface_dataset_name = "JingweiNi/train_prm800k_finished_self_annotate_phi-4_prm"

dataset_paths = [
    f"gen_data/{dataset_name}_0",
    f"gen_data/{dataset_name}_1",
    f"gen_data/{dataset_name}_2",
    f"gen_data/{dataset_name}_3",
    f"gen_data/{dataset_name}_4",
    f"gen_data/{dataset_name}_5",
    f"gen_data/{dataset_name}_6",
    f"gen_data/{dataset_name}_7",
]

datasets = [load_from_disk(dataset_path) for dataset_path in dataset_paths]

datasets = [dataset.cast_column('verified', Sequence(feature=Value(dtype='float64'))) for dataset in datasets]

merged_dataset = concatenate_datasets(datasets)

merged_dataset = DatasetDict({
    "train": merged_dataset
})

merged_dataset.push_to_hub(huggingface_dataset_name)


