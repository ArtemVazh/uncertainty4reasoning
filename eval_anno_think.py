import argparse
import pickle
import os

from transformers import AutoTokenizer

from lm_polygraph import UEManager
from synthetic_dataset_generation.utils.steps_extractor_thinking import StepsExtractorThinking
from synthetic_dataset_generation.utils.step_fact_check_thinking_concurrent import StepFactCheckThinking


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--man-path', type=str, required=True, help='HF or local path to UEManager')
    parser.add_argument('--model-path', type=str, default='Qwen/Qwen3-1.7B', help='HF path to LLM')
    parser.add_argument('--prompt-path', type=str, default='configs/qwen3_prompt_thinking.txt',
                        help='Path to prompt file used with LLM')
    parser.add_argument('--anno-model', type=str, default='openai/gpt-oss-120b', help='Annotator model')
    parser.add_argument('--n-threads', type=int, default=64, help='Number of threads')
    parser.add_argument('--base-url', type=str, default='http://localhost:8000/v1', help='Base URL for local vLLM')
    parser.add_argument('--extracted-claims-path', type=str, required=True, help='Path to save the extracted claims')
    return parser


def load_man(man_path: str) -> UEManager:
    try:
        man = UEManager.load_from_hub(man_path)
        source = 'hf'
        print(f'Loaded UEManager from HF: {man_path}')
    except Exception as hf_err:
        try:
            man = UEManager.load(man_path)
            source = 'local'
            print(f'Loaded UEManager from local path: {man_path}')
        except Exception as local_err:
            raise Exception(
                f'Error loading UEManager from {man_path}:\n'
                f'Trying to load from HF: {hf_err}\n'
                f'Trying to load from local path: {local_err}'
            )
    return man, source


class MockModel:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)


def save_man(man: UEManager, source: str, man_path: str):
    if source == 'local':
        man.save(man_path)
        print(f'Saved to local path: {man_path}')
    elif source == 'hf':
        man.push_to_hub(man_path)
        print(f'Pushed to HF: {man_path}')
    else:
        raise Exception(f'Internal: unknown source {source}')


def main(args):
    man, source = load_man(args.man_path)
    annotator = StepFactCheckThinking(
        prompt_file=args.prompt_path,
        model=args.anno_model,
        n_threads=args.n_threads,
        base_url=args.base_url,
    )
    
    steps_extractor = StepsExtractorThinking(
    )


    if os.path.exists(args.extracted_claims_path):
        with open(args.extracted_claims_path, 'rb') as f:
            extracted_claims = pickle.load(f)
    else:
        extracted_claims = steps_extractor(
            man.stats,
            man.stats['input_texts'],
            MockModel(args.model_path),
        )
        with open(args.extracted_claims_path, 'wb') as f:
            pickle.dump(extracted_claims, f)
        print(f'Saved extracted claims to {args.extracted_claims_path}')

    man.stats.update(extracted_claims)
    annotations = annotator(man.stats, man.stats['target_texts'], None)
    man.gen_metrics[annotator.level, str(annotator)] = [a for sample_anno in annotations for a in sample_anno]
    save_man(man, source, args.man_path)
    print('Done.')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
