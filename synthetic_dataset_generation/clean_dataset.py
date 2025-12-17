from datasets import load_dataset, DatasetDict

dataset_name = "JingweiNi/train_akimbio_apertus_8b_factuality_texts"
columns_to_drop = ['claims', 'verified', 'uncertainty_labels']

ds = load_dataset(dataset_name, split="train")
ds = ds.remove_columns(columns_to_drop)

# Push cleaned dataset back to the hubt
cleaned_ds = DatasetDict({"train": ds})
cleaned_ds.push_to_hub("JingweiNi/train_akimbio_apertus_8b_factuality_texts_cleaned", private=False)
