import argparse
from datasets import load_dataset, Dataset


def main(args):
    with open(args.prompt_file, 'r') as f:
        prompt_template = f.read()
    dataset = load_dataset(*args.dataset_path, cache_dir=args.hf_cache)[args.dataset_split]
    # Slice dataset if needed
    if args.start_index is not None:
        dataset = dataset.select(range(args.start_index, len(dataset)))

    # Format questions and extract answers
    questions, answers = [], []
    for inst in dataset:
        questions.append(prompt_template.format(q=inst[args.question_col]))
        answers.append(inst[args.answer_col])

    # Create and save new dataset
    ds = Dataset.from_dict({'question': questions, 'answer': answers})
    ds.save_to_disk(args.save_path)

    if args.hf_save_path is not None:
        ds.push_to_hub(args.hf_save_path)


def parse_tuple(s):
    try:
        parts = s.strip("()").split(",")
        return tuple(part.strip() for part in parts)
    except Exception:
        raise argparse.ArgumentTypeError("Tuple must be in the form: value1,value2")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare test dataset with prompt formatting.")
    parser.add_argument('--dataset-path', type=parse_tuple, default=("openai/gsm8k", "main"),
                        help='Path to the dataset as a tuple, e.g. "openai/gsm9k,main')
    parser.add_argument('--dataset-split', type=str, default='test', help='Dataset split to load')
    parser.add_argument('--question-col', type=str, default="question", help='Column in the dataset with questions')
    parser.add_argument('--answer-col', type=str, default="answer", help='Column in the dataset with answers')
    parser.add_argument('--start-index', type=int, required=True, help='Start index for slicing dataset')
    parser.add_argument('--save-path', type=str, required=True, help='Directory to save the processed dataset')
    parser.add_argument('--prompt-file', type=str, default='configs/gsm8k_3shot_prompt.txt',
                        help='Path to the prompt template file')
    parser.add_argument('--hf-cache', type=str, default=None, help='HuggingFace cache directory')
    parser.add_argument('--hf-save-path', type=str, default=None, help='HuggingFace repository name to save dataset')

    args = parser.parse_args()
    main(args)
