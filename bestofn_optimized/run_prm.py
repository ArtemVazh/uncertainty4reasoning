import argparse
import logging
from transformers import AutoTokenizer

from baselines.prm import load_prm_calculator_by_model_path
from bestofn.estimators.prm import PRMEstimator
from bestofn.estimators.uhead import UHeadEstimator
from bestofn_optimized.run_uhead import update_with_estimators
from lm_polygraph.stat_calculators.step.steps_extractor import StepsExtractor
from synthetic_dataset_generation.run_generate_texts import parse_tuple, load_bon_dataset, load_prompt
from configs.load_qwen import load_model as load_qwen_model, load_tokenizer as load_qwen_tokenizer
from lm_polygraph import WhiteboxModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bestofn_eval")


def load_model(model_path, device):
    tokenizer = load_qwen_tokenizer(model_path)
    base_model = load_qwen_model(model_path, device)
    base_model.eval()
    return WhiteboxModel(base_model, tokenizer)


def parse_args():
    parser = argparse.ArgumentParser(description="Score generated texts for offline BoN.")

    # Dataset
    parser.add_argument('--dataset-path', type=parse_tuple, default=("openai/gsm8k", "main"))
    parser.add_argument('--dataset-split', type=parse_tuple, default=None)
    parser.add_argument('--question-col', type=str, default="question")
    parser.add_argument('--answer-col', type=str, default="answer")
    parser.add_argument('--prompt-file', type=str, default=None)

    # Models
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--device', type=str, default="auto")

    # Other
    parser.add_argument('--prm-path', type=str, nargs='+', default=[
        "Qwen/Qwen2.5-Math-7B-PRM800K",
        "Qwen/Qwen2.5-Math-PRM-7B",
        "peiyi9979/math-shepherd-mistral-7b-prm",
        "RLHFlow/Llama3.1-8B-PRM-Mistral-Data",
        "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
        # "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B",  # loads slow (can take up to 15 mins)
        "universalprm/Universal-PRM",
        "HuggingFaceH4/Qwen2.5-Math-1.5B-Instruct-PRM-0.2",
    ], help="Path(s) or name(s) of the PRM model(s)")
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=1)

    # Periodic saving
    parser.add_argument('--save-every', type=int, default=50,
                        help='Save after this many newly-processed rows')

    return parser.parse_args()


def estimator_column_names(estimators, uhead_path: str):
    cols = []
    for est in estimators:
        if isinstance(est, UHeadEstimator):
            cols.append(uhead_path)
        else:
            cols.append(str(est))
    # de-duplicate in case multiple UHeadEstimator instances exist
    return list(dict.fromkeys(cols))


def ensure_columns(dataset, column_names):
    n = len(dataset)
    for col in column_names:
        if col not in dataset.column_names:
            dataset = dataset.add_column(col, [None] * n)
    return dataset


def get_indices_to_process(dataset, column_names):
    masks = []
    for col in column_names:
        col_vals = dataset[col]
        masks.append([v is None for v in col_vals])
    idxs = []
    for i in range(len(dataset)):
        if any(m[i] for m in masks):
            idxs.append(i)
    return idxs


def periodic_save(dataset, path):
    log.info("Saving dataset to disk...")
    dataset.save_to_disk(path)
    log.info("Saved.")


class MockModel:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)


def main(args):
    dataset = load_bon_dataset(args.dataset_path, args.dataset_split)
    prompt = load_prompt(args.prompt_file)
    # model = load_model(args.model_path, args.device)
    model = MockModel(args.model_path)

    for prm_path in args.prm_path:
        print(f'Running {prm_path}...')

        estimators = [
            PRMEstimator(scores_key=prm_path, reduction='max'),
        ]
        stat_calculators = [
            StepsExtractor(progress_bar=False),
            load_prm_calculator_by_model_path(
                model_path=prm_path,
                device=args.device,
                scores_key=prm_path,  # save score under PRM name
            )
        ]

        dataset = update_with_estimators(
            dataset, prompt, model,
            stat_calculators, estimators,
            args.save_path,
            None,
            args.batch_size,
            args.question_col,
            args.answer_col,
            args.save_every,
        )
        print(f'Done with {prm_path}.')

    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
