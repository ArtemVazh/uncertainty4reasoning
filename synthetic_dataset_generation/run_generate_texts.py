import torch
import argparse
import numpy as np
import multiprocessing
from spacy.tokens.doc import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset, load_from_disk, Dataset
from functools import partial
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from utils import parse_ans
from vllm import LLM, SamplingParams
from transformers import modeling_utils
import os
import pandas as pd
from datasets import Dataset

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']
from itertools import chain
from tqdm import tqdm

GPU_NUM = torch.cuda.device_count()


def generate_replies(inst, prompt, args, model, tokenizer, generation_config):
    inst["question"] = inst[args.question_col]
    inst["answer"] = inst[args.answer_col]
    question = prompt.format(q=inst["question"])
    inputs = tokenizer(question, return_tensors='pt')['input_ids']
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            num_return_sequences=1,
            generation_config=generation_config,
            pad_token_id=tokenizer.eos_token_id,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=1024,
            do_sample=False,
            repetition_penalty=1.,
            diversity_penalty=0.,
            length_penalty=1.,
            stop_strings=[tokenizer.eos_token],
            tokenizer=tokenizer,
        )
    reply = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    inst["input_ids"] = outputs[0]
    inst["reply"] = reply
    return inst


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

    parser.add_argument('--dataset-path', type=parse_tuple, default=("openai/gsm8k", "main"),
                        help='Path to the dataset as a tuple, e.g. "openai/gsm9k,main". Start with "local" to load from local path, e.g. "local,./scienceqa_missing_images"')
    parser.add_argument('--dataset-split', type=parse_tuple, default=None, help='Dataset split')
    parser.add_argument('--question-col', type=str, default="question", help='Column in the dataset with questions')
    parser.add_argument('--answer-col', type=str, default="answer", help='Column in the dataset with answers')
    parser.add_argument('--final-answers', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether dataset contains final answers for each problem')
    parser.add_argument('--n-samples', type=int, required=True, help='Number of samples to evaluate from the dataset')
    parser.add_argument('--prompt-file', type=str, required=True, help='Path to the prompt text file')

    parser.add_argument('--model-path', type=str, required=True, help='Path to the pretrained model')
    parser.add_argument('--device', type=str, default="auto", help='Device to infer model on')

    parser.add_argument('--save-path', type=str, required=True, help='Path to save the processed dataset')
    parser.add_argument('--hf-cache', type=str, default=None, help='Path to the HuggingFace cache directory')
    parser.add_argument('--vllm', action='store_true', default=False,
                        help='Whether to use vLLM as the inference backend')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for the model')
    parser.add_argument('--top-p', type=float, default=0.95, help='Top-p for the model')
    parser.add_argument('--top-k', type=int, default=50, help='Top-k for the model')
    parser.add_argument('--n', type=int, default=3, help='Number of samples to generate')
    return parser.parse_args()

def main(args):
    if args.vllm:
        from vllm import LLM, SamplingParams

    prompt = open(args.prompt_file, 'r').read()

    if args.dataset_path[0] == 'local':
        dataset = load_from_disk(args.dataset_path[1])
        if args.dataset_split is not None:
            dataset = dataset[args.dataset_split[0]]
    elif os.path.isfile(args.dataset_path):
        # Load from local file
        if args.dataset_path.endswith('.csv'):
            df = pd.read_csv(args.dataset_path)
        else:
            df = pd.read_json(args.dataset_path, lines=True)

        # df_new = df[[args.question_col, args.answer_col]]   
        dataset = Dataset.from_pandas(df)
    else:
        dataset = load_dataset(*args.dataset_path, cache_dir=args.hf_cache)
        if args.dataset_split is not None:
            dataset = dataset[args.dataset_split[0]]
    
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
    
    dataset = dataset.select(range(args.n_samples))
    generation_config = GenerationConfig.from_pretrained(args.model_path)

    if not args.vllm:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map=args.device, trust_remote_code=True, cache_dir=args.hf_cache)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.hf_cache)

        dataset = dataset.map(partial(
            generate_replies, prompt=prompt, args=args,
            model=model, tokenizer=tokenizer,
            generation_config=generation_config,
        ))
    else:
        prompts = [prompt.format(q=q) for q in dataset[args.question_col]]
        print(prompts[0])
        import pdb; pdb.set_trace()
        # Determine effective temperature for vLLM (same logic as transformers backend)
        need_sampling = args.n_samples_per_input > 1 or args.temperature > 0
        print(f"Need sampling: {need_sampling}")
        print(f"Temperature: {args.temperature}")
        effective_temperature = args.temperature if args.temperature > 0 else (0.6 if need_sampling else 0.0)
        
        sampling_params = SamplingParams(
            n=args.n,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=42,
            max_tokens=1024,
            repetition_penalty=1.,
            stop=["<|im_end|>", "<|endoftext|>"],
            include_stop_str_in_output=True,
            temperature=effective_temperature,  # Use effective temperature
        )
        sampling_params.update_from_generation_config(generation_config.to_dict())

        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=GPU_NUM,
            download_dir=args.hf_cache,
            tokenizer=args.model_path,
            dtype='auto',
            trust_remote_code=True,
            gpu_memory_utilization=0.75,
        )

        outputs = llm.generate(prompts, sampling_params)

        new_dataset = []
        for data, output in zip(dataset, outputs):
            for gen in output.outputs:
                new_data_point = data.copy()
                new_data_point["question"] = data[args.question_col]
                new_data_point["answer"] = data[args.answer_col]
                new_data_point["input_ids"] = list(output.prompt_token_ids) + list(gen.token_ids)
                new_data_point["reply"] = gen.text
                new_dataset.append(new_data_point)
        # Convert list of dicts to HuggingFace Dataset
        dataset = Dataset.from_dict({k: [d[k] for d in new_dataset] for k in new_dataset[0]})

    if 'gsm8k' in args.dataset_path[0]:
        print_stats(dataset, args)

    dataset.save_to_disk(args.save_path)
    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    