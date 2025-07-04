import argparse
from datasets import load_dataset, Dataset, load_from_disk


def main(args):
    with open(args.prompt_file, 'r') as f:
        prompt_template = f.read()
    if 'local' in str(args.dataset_path):
        dataset = load_from_disk(args.dataset_path[1])[args.dataset_split]
    else:
        dataset = load_dataset(*args.dataset_path, cache_dir=args.hf_cache)[args.dataset_split]
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
        if 'strategy_qa' in args.dataset_path:
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
        elif 'science_qa' in args.dataset_path:
            def parse_science_qa_question(example):
                LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                question = example["question"]
                choices = example["choices"]
                ret = f"Answer this multiple choice question with one correct answer: {question}\nChoices:\n"
                for i, choice in enumerate(choices):
                    ret += f"  {LETTERS[i]}. {choice}\n"
                return ret
            
            def parse_science_qa_answer(example):
                answer_str = f"The answer is {example[args.answer_col]}. Reasoning: {example['solution']}"
                return answer_str
            
            questions.append(parse_science_qa_question(inst))
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
    parser.add_argument('--start-index', type=int, default=None, help='Start index for slicing dataset')
    parser.add_argument('--dataset-size', type=int, default=None, help='Number of test instances, default: all')
    parser.add_argument('--save-path', type=str, required=True, help='Directory to save the processed dataset')
    parser.add_argument('--prompt-file', type=str, default='configs/gsm8k_3shot_prompt.txt',
                        help='Path to the prompt template file')
    parser.add_argument('--hf-cache', type=str, default=None, help='HuggingFace cache directory')
    parser.add_argument('--hf-save-path', type=str, default=None, help='HuggingFace repository name to save dataset')

    args = parser.parse_args()
    main(args)