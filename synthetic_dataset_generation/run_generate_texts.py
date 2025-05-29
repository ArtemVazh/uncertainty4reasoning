import torch
import argparse
import numpy as np
from spacy.tokens.doc import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset
from functools import partial
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from utils import parse_ans
from vllm import LLM, SamplingParams

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
            temperature=0.,
            max_new_tokens=256,
            do_sample=False,
            repetition_penalty=1.,
            diversity_penalty=0.,
            length_penalty=1.,
            stop_strings=['\n\n', '}\n'],
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
    return parser.parse_args()


def main(args):
    prompt = open(args.prompt_file, 'r').read()

    dataset = load_dataset(*args.dataset_path, cache_dir=args.hf_cache)['test']
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
        sampling_params = SamplingParams(
            n=1,
            temperature=0,
            seed=42,
            max_tokens=256,
            repetition_penalty=1.,
            stop=['\n\n', '}\n'],
            include_stop_str_in_output=True,
        )
        sampling_params.update_from_generation_config(generation_config)

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
            return {
                "question": example[args.question_col],
                "answer": example[args.answer_col],
                "input_ids": list(vllm_outputs[idx].prompt_token_ids) + list(vllm_outputs[idx].outputs[0].token_ids),
                "reply": vllm_outputs[idx].outputs[0].text,
            }

        # Apply using map with indices
        dataset = dataset.map(
            partial(parse_vllm_output, vllm_outputs=outputs),
            with_indices=True
        )

    print_stats(dataset, args)

    dataset.save_to_disk(args.save_path)
    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
