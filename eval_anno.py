import argparse

from transformers import AutoTokenizer

from lm_polygraph import UEManager
from lm_polygraph.stat_calculators.step.steps_extractor import StepsExtractor
from synthetic_dataset_generation.utils.step_fact_check import StepFactCheck


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--man-path', type=str, required=True, help='HF or local path to UEManager')
    parser.add_argument('--model-path', type=str, default='Qwen/Qwen3-8B', help='HF path to LLM')
    parser.add_argument('--prompt-path', type=str, default='configs/qwen3_prompt_general.txt',
                        help='Path to prompt file used with LLM')
    parser.add_argument('--anno-model', type=str, default='deepseek-reasoner', help='Annotator model')
    parser.add_argument('--n-threads', type=int, default=16, help='Number of threads')
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
    annotator = StepFactCheck(
        prompt_file=args.prompt_path,
        model=args.anno_model,
        n_threads=args.n_threads,
    )
    steps_extractor = StepsExtractor()
    man.stats.update(steps_extractor(
        man.stats,
        man.stats['input_texts'],
        MockModel(args.model_path),
    ))
    annotations = annotator(man.stats, man.stats['target_texts'])
    man.gen_metrics[annotator.level, str(annotator)] = [a for sample_anno in annotations for a in sample_anno]
    save_man(man, source, args.man_path)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
