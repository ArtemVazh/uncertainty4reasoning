import torch
import argparse
import copy
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset, Dataset
from functools import partial
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from utils import parse_ans
from vllm import LLM, SamplingParams
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
            inputs.repeat(args.n_samples_per_input, 1),
            num_return_sequences=args.n_samples_per_input,
            generation_config=generation_config,
            pad_token_id=tokenizer.eos_token_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=256,
            do_sample=args.temperature > 0,
            repetition_penalty=1.,
            diversity_penalty=0.,
            length_penalty=1.,
            stop_strings=[tokenizer.eos_token],
            tokenizer=tokenizer,
        )
    replies = []
    for i in range(args.n_samples_per_input):
        reply_text = tokenizer.decode(outputs[i][inputs.shape[1]:], skip_special_tokens=True)
        reply = copy.deepcopy(inst)
        reply.update({
            "input_ids": outputs[i].tolist(),
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

    parser.add_argument('--dataset-path', type=parse_tuple, default=("openai/gsm8k", "main"),
                        help='Path to the dataset as a tuple, e.g. "openai/gsm9k,main"')
    parser.add_argument('--dataset-split', type=parse_tuple, default="test", help='Dataset split')
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

    parser.add_argument('--temperature', type=float, default=0, help='Temperature to generate training data with')
    parser.add_argument('--top-k', type=int, default=None, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=1.0, help='Top-p (nucleus) sampling')
    parser.add_argument('--n-samples-per-input', type=int, default=1,
                        help='How many completions to generate per input')

    return parser.parse_args()


def main(args):
    prompt = open(args.prompt_file, 'r').read()

    dataset = load_dataset(*args.dataset_path, cache_dir=args.hf_cache)[args.dataset_split[0]]
    dataset = dataset.select(range(args.n_samples))
    generation_config = GenerationConfig.from_pretrained(args.model_path)

    if not args.vllm:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, device_map=args.device, trust_remote_code=True, cache_dir=args.hf_cache)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.hf_cache)

        results = []
        for inst in tqdm(dataset, total=len(dataset), desc='Generating texts'):
            results.extend(generate_replies(inst, prompt, args, model, tokenizer, generation_config))
        dataset = Dataset.from_list(results)
    else:
        prompts = [prompt.format(q=q) for q in dataset[args.question_col]]
        sampling_params = SamplingParams(
            n=args.n_samples_per_input,
            seed=42,
            max_tokens=256,
            repetition_penalty=1.,
            stop=["<|im_end|>", "<|endoftext|>"],
            include_stop_str_in_output=True,
        )
        sampling_params.update_from_generation_config(generation_config.to_dict())
        sampling_params.temperature = args.temperature
        if args.top_k is not None:
            sampling_params.top_k = args.top_k
        sampling_params.top_p = args.top_p

        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=GPU_NUM,
            download_dir=args.hf_cache,
            tokenizer=args.model_path,
            dtype='auto',
            trust_remote_code=True,
        )

        outputs = llm.generate(prompts, sampling_params)

        def parse_vllm_output(example, idx, vllm_outputs):
            return [{
                "question": example[args.question_col],
                "answer": example[args.answer_col],
                "input_ids": list(vllm_outputs[idx].prompt_token_ids) + list(out.token_ids),
                "reply": out.text,
            } for out in vllm_outputs[idx].outputs]

        # Apply using map with indices
        mapped = dataset.map(
            partial(parse_vllm_output, vllm_outputs=outputs),
            with_indices=True
        )
        dataset = Dataset.from_list(list(chain.from_iterable(mapped["__array__"])))

    dataset.save_to_disk(args.save_path)
    if any(x in args.dataset_path[0].lower() for x in ['gsm8k', 'proofnet', 'math']):
        print_stats(dataset, args)

    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
