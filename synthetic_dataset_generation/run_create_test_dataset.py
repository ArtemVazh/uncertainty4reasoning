import argparse
from datasets import load_dataset, Dataset, load_from_disk
import os
import pandas as pd


def main(args):
    with open(args.prompt_file, 'r') as f:
        prompt_template = f.read()
    
    # Handle both file paths and HuggingFace dataset paths
    if isinstance(args.dataset_path, str):
        # Load from local file
        if args.dataset_path.endswith('.csv'):
            df = pd.read_csv(args.dataset_path)
            df_new = df[[args.question_col, args.answer_col]]   
            dataset = Dataset.from_pandas(df_new)
        elif args.dataset_path.endswith('.json') or args.dataset_path.endswith('.jsonl'):
            df = pd.read_json(args.dataset_path, lines=True)
            df_new = df[[args.question_col, args.answer_col]]   
            dataset = Dataset.from_pandas(df_new)
        else:
            dataset = load_dataset(args.dataset_path)[args.dataset_split]
    elif isinstance(args.dataset_path, tuple):
        if args.dataset_path[0] == 'local':
            dataset = load_from_disk(args.dataset_path[1])[args.dataset_split]
        else:
            # Load from HuggingFace dataset
            dataset = load_dataset(*args.dataset_path, cache_dir=args.hf_cache)[args.dataset_split]
    else:
        raise ValueError("dataset_path must be either a file path (str) or a HuggingFace dataset tuple")
    
    # Slice dataset if needed
    if args.start_index is not None:
        dataset = dataset.select(range(args.start_index, len(dataset)))
    if args.dataset_size is not None:
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(args.dataset_size))
    print(f'Creating test dataset of size {len(dataset)}')

    # Format questions and extract answers
    questions, answers = [], []
    for inst in dataset:
        if 'strategy-qa' in str(args.dataset_path):
            questions.append(prompt_template.format(q=inst[args.question_col]))
            def parse_strategy_qa_answer(example):
                answer_str = ""
                if example[args.answer_col]:
                    answer_str += "Yes, according to the facts: "
                else:
                    answer_str += "No, according to the facts: "
                for i, fact in enumerate(example["facts"]):
                    answer_str += f"{i+1}. {fact} "
                return answer_str
            answers.append(parse_strategy_qa_answer(inst))
        elif 'science_qa' in str(args.dataset_path):
            def parse_science_qa_question(example):
                LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                question = example["question"]
                choices = example["choices"]
                ret = f"Answer this multiple choice question with one correct answer: {question}\nChoices:\n"
                for i, choice in enumerate(choices):
                    ret += f"  {LETTERS[i]}. {choice}\n"
                return ret
            
            def parse_science_qa_answer(example):
                LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                answer_str = f"The answer is {LETTERS[example[args.answer_col]]}. Reasoning: {example['solution']}"
                return answer_str
            
            questions.append(prompt_template.format(q=parse_science_qa_question(inst)))
            answers.append(parse_science_qa_answer(inst))
        else:   
            questions.append(prompt_template.format(q=inst[args.question_col]))
            answers.append(inst[args.answer_col])

    # Create and save new dataset
    ds = Dataset.from_dict({'question': questions, 'answer': answers})
    ds.save_to_disk(args.save_path)

    if args.hf_save_path is not None:
        ds.push_to_hub(args.hf_save_path)


def parse_tuple(s):
    if ',' not in s:
        # If no comma, treat as a file path
        return s
    try:
        parts = s.strip("()").split(",")
        return tuple(part.strip() for part in parts)
    except Exception:
        raise argparse.ArgumentTypeError("Input must be either a file path or a tuple in the form: value1,value2")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare test dataset with prompt formatting.")
    parser.add_argument('--dataset-path', type=parse_tuple,
                        help='Either a local file path or a HuggingFace dataset tuple (e.g. "openai/gsm8k,main")')
    parser.add_argument('--dataset-split', type=str, default='test', help='Dataset split to load')
    parser.add_argument('--question-col', type=str, default="question", help='Column in the dataset with questions')
    parser.add_argument('--answer-col', type=str, default="answer", help='Column in the dataset with answers')
    parser.add_argument('--start-index', type=int, default=None, help='Start index for slicing dataset')
    parser.add_argument('--dataset-size', type=int, default=None, help='Number of test instances, default: all')
    parser.add_argument('--save-path', type=str, required=True, help='Directory to save the processed dataset')
    parser.add_argument('--prompt-file', type=str, default='configs/gsm8k_3shot_prompt.txt',
                        help='Path to the prompt template file')
    parser.add_argument('--hf-cache', type=str, default=None, help='HuggingFace cache directory')
    parser.add_argument('--hf-save-path', type=str, default=None, help='HuggingFace repository name to save dataset')

    args = parser.parse_args()
    main(args)
