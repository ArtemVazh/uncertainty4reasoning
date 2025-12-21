import argparse
from datasets import load_dataset, Dataset, DatasetDict
import os
import pandas as pd
from parse import parse


def main(args):
    with open(args.new_prompt_file, 'r') as f:
        new_prompt_template = f.read()
    with open(args.old_prompt_file, 'r') as f:
        old_prompt_template = f.read()

    dataset = load_dataset(args.dataset_path)['train']
    new_question = []
    for question in dataset['question']:
        original_question = parse(old_prompt_template, question)['q']
        new_question.append(new_prompt_template.format(q=original_question))
    
    new_dataset = Dataset.from_dict({'question': new_question, 'answer': dataset['answer']})
    new_dataset = DatasetDict({'train': new_dataset})
    new_dataset.push_to_hub(args.hf_save_path, private=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare test dataset with prompt formatting.")
    parser.add_argument('--dataset-path', type=str,
                        help='HuggingFace dataset path (e.g. "JingweiNi/test_strategy_qa_Qwen3-8B_np")')
    parser.add_argument('--new-prompt-file', type=str, required=True, help='Path to the new prompt template file')
    parser.add_argument('--old-prompt-file', type=str, required=True, help='Path to the old prompt template file')
    parser.add_argument('--hf-save-path', type=str, default=None, help='HuggingFace repository name to save dataset')

    args = parser.parse_args()
    main(args)

    # JingweiNi/test_strategy_qa_Qwen3-8B_np
    # JingweiNi/test_science_qa_Qwen3-8B_np_fixed
    # awsuineg/test_meeting_Qwen3-8B_texts
    # awsuineg/test_trip_Qwen3-8B_texts
    # awsuineg/test_calendar_Qwen3-8B_texts
    # rediska0123/test_gsm8k_Qwen3-8B
    # rediska0123/test_math_Qwen3-8B
    # rediska0123/test_proofnet_Qwen3-8B
