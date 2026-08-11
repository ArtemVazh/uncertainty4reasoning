import multiprocessing
# Set multiprocessing start method to 'spawn' for CUDA compatibility
# Must be done before importing torch to avoid fork-related CUDA errors
multiprocessing.set_start_method('spawn', force=True)

import torch
import traceback
import argparse
import numpy as np
import copy
from spacy.tokens.doc import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset, load_from_disk, Dataset
from functools import partial
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from utils import parse_ans
from transformers import modeling_utils
import os
import pandas as pd
from datasets import Dataset

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']
from tqdm import tqdm

# GPU_NUM = torch.cuda.device_count()


def get_stop_strings(tokenizer):
    stop_strings = [
        "<|im_end|>",
        "<|endoftext|>",
        "<|return|>",
        "<|end|>",
    ]
    for token in (getattr(tokenizer, "eos_token", None), getattr(tokenizer, "pad_token", None)):
        if token:
            stop_strings.append(token)
    return list(dict.fromkeys(stop_strings))

def clean_assistant_channel_artifacts(text):
    if "assistantfinal" in text:
        text = text.rsplit("assistantfinal", 1)[-1].lstrip()
    for marker in ("assistantanalysis", "assistantcommentary"):
        if marker in text:
            text = text.split(marker, 1)[0].rstrip()
    end_marker = "<end of response>"
    if end_marker in text:
        text = text[:text.index(end_marker) + len(end_marker)]
    for prefix in ("<start of response>\nReasoning Steps:", "Reasoning Steps:"):
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip()
    return text.strip()


def build_input_ids(tokenizer, prompt_text, reply_text):
    return tokenizer(prompt_text + reply_text, return_tensors='pt')['input_ids'][0].tolist()

def generate_replies(inst, prompt, args, model, tokenizer, generation_config):
    inst["question"] = inst[args.question_col]
    inst["answer"] = inst[args.answer_col]
    question = prompt.format(q=inst["question"])
    inputs = tokenizer(question, return_tensors='pt')['input_ids'].to(model.device)

    total = args.n_samples_per_input
    batch_size = args.batch_size
    replies = []

    with torch.no_grad():
        for start in range(0, total, batch_size):
            current_bs = min(batch_size, total - start)

            outputs = model.generate(
                inputs.repeat(current_bs, 1),
                num_return_sequences=current_bs,
                generation_config=generation_config,
                pad_token_id=tokenizer.eos_token_id,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                max_new_tokens=256,
                do_sample=args.temperature > 0,
                repetition_penalty=1.0,
                diversity_penalty=0.0,
                length_penalty=1.0,
            )

            for i in range(current_bs):
                reply_text = tokenizer.decode(outputs[i][inputs.shape[1]:], skip_special_tokens=True)
                reply_text = clean_assistant_channel_artifacts(reply_text)
                reply = copy.deepcopy(inst)
                reply.update({
                    "input_ids": build_input_ids(tokenizer, question, reply_text),
                    "reply": reply_text,
                })
                replies.append(reply)

    return replies


def jaccard_similarity(a, b):
    a_set = set(a.lower().split())
    b_set = set(b.lower().split())
    intersection = a_set.intersection(b_set)
    union = a_set.union(b_set)
    return len(intersection) / len(union) if union else 0.0


def print_stats(dataset, args):
    stats = defaultdict(list)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    for a, d in zip(dataset['answer'], dataset['reply']):
        if args.final_answers:
            gt_ans = parse_ans(a)
            llm_ans = parse_ans(d)
            stats['Accuracy'].append(gt_ans == llm_ans if llm_ans is not None else 0)
            stats['Finished'].append(llm_ans is not None)
        else:
            stats['BLEU'].append(sentence_bleu([a.split()], d.split()))
            for key, val in scorer.score(a, d).items():
                stats[f'{key[0].upper() + key[1:]}'].append(val.fmeasure)
            stats['Jaccard'].append(jaccard_similarity(a, d))
    for key, vals in stats.items():
        print(f'{key}: {np.mean(vals)}')


def parse_tuple(s):
    try:
        parts = s.strip("()").split(",")
        return tuple(part.strip() for part in parts)
    except Exception:
        raise argparse.ArgumentTypeError("Tuple must be in the form: value1,value2")


def parse_args():
    parser = argparse.ArgumentParser(description="Create generation texts for model.")

    # Dataset
    parser.add_argument('--model-path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--dataset-path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--dataset-split', type=str, default=None, help='Dataset config/split name (e.g., "main")')
    parser.add_argument('--dataset-subsplit', type=str, default='train', help='Specific split within the config (e.g., "train", "test")')
    parser.add_argument('--question-col', type=str, default="question", help='Column in the dataset with questions')
    parser.add_argument('--answer-col', type=str, default="answer", help='Column in the dataset with answers')
    parser.add_argument('--final-answers', action=argparse.BooleanOptionalAction, default=False, help='Whether dataset contains final answers for each problem')
    parser.add_argument('--n-samples', type=int, default=None, help='Number of samples to evaluate from the dataset')
    parser.add_argument('--prompt-file', type=str, default=None, help='Path to the prompt text file')
    # LLMdd_argument('--model-path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--device', type=str, default="auto", help='Device to infer model on')
    parser.add_argument('--save-path', type=str, required=True, help='Path to save the processed dataset')
    parser.add_argument('--vllm', action='store_true', default=False,
                        help='Whether to use vLLM as the inference backend')

    # Generation config
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for the model')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p for the model')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k for the model')
    parser.add_argument('--max-new-tokens', type=int, default=1024, help='Max new tokens in generations')

    # Batching
    parser.add_argument('--n-samples-per-input', type=int, default=3, help='Number of samples to generate')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batching for n-samples-per-input. '
                             'E.g., with "--n-samples-per-input=32 --batch-size 8", '
                             'there will be 4 model.generate() calls per each input text.'
                        )
    return parser.parse_args()


def load_bon_dataset(dataset_path, dataset_split):
    if dataset_path[0] == 'local':
        dataset = load_from_disk(dataset_path[1])
        if dataset_split is not None:
            try:
                dataset = dataset[dataset_split[0]]
            except Exception as e:
                print(f'Unknown split: {dataset_split[0]} for dataset: {dataset}')
                raise e
    elif os.path.isfile(dataset_path[0]):
        # Load from local file
        file_path = dataset_path[0]
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_json(file_path, lines=True)

        # df_new = df[[question_col, answer_col]]
        dataset = Dataset.from_pandas(df)
    else:
        dataset = load_dataset(dataset_path)
        if dataset_split is not None:
            dataset = dataset[dataset_split]
    return dataset

def load_prompt(prompt_file):
    if prompt_file is None:
        return '{q}'
    return open(prompt_file, 'r').read()


def main(args):
    if args.vllm:
        from vllm import LLM, SamplingParams
        GPU_NUM = torch.cuda.device_count() 

    prompt = load_prompt(args.prompt_file)
    dataset_dict = load_dataset(args.dataset_path, args.dataset_split)
    # DatasetDict may contain multiple splits (train, test, etc.)
    dataset = dataset_dict[args.dataset_subsplit]

    if 'scienceqa' in str(args.dataset_path):
        def format_scienceqa_question(example):
            LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            question = example["question"]
            choices = example["choices"]
            ret = f"Answer this multiple choice question with one correct answer: {question}\nChoices:\n"
            for i, choice in enumerate(choices):
                ret += f"  {LETTERS[i]}. {choice}\n"
            return {"question_with_choices": ret, "answer_choice": LETTERS[example["answer"]]}

        dataset = dataset.map(format_scienceqa_question)
        dataset = dataset.rename_column("question", "question_without_choices")
        dataset = dataset.rename_column("answer", "answer_raw")
        dataset = dataset.rename_column("question_with_choices", "question")
        dataset = dataset.rename_column("answer_choice", "answer")

    if args.n_samples is not None:
        dataset = dataset.select(range(args.n_samples))
    else:
        dataset = dataset.select(range(len(dataset)))
    generation_config = GenerationConfig.from_pretrained(args.model_path)

    if not args.vllm:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map=args.device, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        results = []
        for inst in tqdm(dataset, total=len(dataset), desc='Generating texts'):
            results.extend(generate_replies(inst, prompt, args, model, tokenizer, generation_config))
        dataset = Dataset.from_list(results)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        prompts = [prompt.format(q=q) for q in dataset[args.question_col]]
        print(prompts[0])
        # Determine effective temperature for vLLM (same logic as transformers backend)
        # need_sampling = args.n_samples_per_input > 1 or args.temperature > 0
        # print(f"Need sampling: {need_sampling}")
        # print(f"Temperature: {args.temperature}")
        # effective_temperature = args.temperature if args.temperature > 0 else (0.6 if need_sampling else 0.0)

        sampling_params = SamplingParams(
            n=args.n_samples_per_input,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=42,
            max_tokens=args.max_new_tokens,
            repetition_penalty=1.,
            stop=get_stop_strings(tokenizer),
            include_stop_str_in_output=False,
        )
        sampling_params.update_from_generation_config(generation_config.to_dict())
        sampling_params.stop = get_stop_strings(tokenizer)
        sampling_params.include_stop_str_in_output = False

        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=GPU_NUM,
            tokenizer=args.model_path,
            dtype='auto',
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )

        outputs = llm.generate(prompts, sampling_params)

        new_dataset = []
        for data, prompt_text, output in zip(dataset, prompts, outputs):
            for gen in output.outputs:
                reply_text = clean_assistant_channel_artifacts(gen.text)
                new_data_point = data.copy()
                new_data_point["question"] = data[args.question_col]
                new_data_point["answer"] = data[args.answer_col]
                new_data_point["input_ids"] = build_input_ids(tokenizer, prompt_text, reply_text)
                new_data_point["reply"] = reply_text
                new_dataset.append(new_data_point)
        # Convert list of dicts to HuggingFace Dataset
        dataset = Dataset.from_dict({k: [d[k] for d in new_dataset] for k in new_dataset[0]})

    dataset.save_to_disk(args.save_path)
    print(f"Saved to {args.save_path}")

    try:
        print_stats(dataset, args)
    except Exception as e:
        print(f'Error trying to print stats: {e}')
        traceback.print_exc()

    print("Done.")

if __name__ == "__main__":
    args = parse_args()
    main(args)
