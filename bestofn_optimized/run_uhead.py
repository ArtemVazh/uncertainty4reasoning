import argparse
import time
import logging
from tqdm import tqdm
from datasets import Dataset

from bestofn.estimators.uhead import UHeadEstimator
from bestofn_optimized.proxy_calculator import ProxyCalculator
from lm_polygraph.defaults.register_default_stat_calculators import create_container, register_default_stat_calculators
from lm_polygraph.stat_calculators.step.steps_extractor import StepsExtractor
from lm_polygraph.utils.builder_enviroment_stat_calculator import BuilderEnvironmentStatCalculator
from lm_polygraph.utils.factory_stat_calculator import FactoryStatCalculator
from lm_polygraph.utils.manager import initialize_stat_calculators
from luh.calculator_apply_uq_head import CalculatorApplyUQHead
from lm_polygraph.estimators import *
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
    parser.add_argument(
        '--uhead-path',
        type=str,
        default="JingweiNi/uhead_claim_Qwen3-8B_fixed_prm_layer1_dim512_head16_e5_lr5e-4_pos3",
        help="Path or name of the single UHead model"
    )
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
    dataset = Dataset.from_dict(dataset.to_dict())  # Otherwise fails to overwrite existing dataset
    dataset.save_to_disk(path)
    log.info("Saved.")


def update_with_estimators(
        dataset, prompt, model,
        stat_calculators, estimators,
        save_path: str,
        uhead_path: str = None,
        batch_size: int = 1,
        question_col: str = "question",
        answer_col: str = "answer",
        save_every: int = 50,
):
    required_cols = estimator_column_names(estimators, uhead_path) + ['saved_stats']
    dataset = ensure_columns(dataset, required_cols)

    pending_indices = get_indices_to_process(dataset, required_cols)
    if not pending_indices:
        log.info("All rows already have scores. Nothing to do.")
        periodic_save(dataset, save_path)
        print("Done.")
        return

    log.info(f"{len(pending_indices)} rows need processing out of {len(dataset)} total.")

    processed_since_save = 0
    col_buffers = {col: list(dataset[col]) for col in required_cols}

    for start in tqdm(range(0, len(pending_indices), batch_size), desc="Scoring"):
        batch_indices = pending_indices[start:start + batch_size]
        samples = [dataset[i] for i in batch_indices]

        texts = [prompt.format(q=s[question_col]) for s in samples]
        tokens = [model.tokenizer(t)["input_ids"] for t in texts]
        stats = {
            'input_texts': texts,
            'input_tokens': tokens,
            'proxy_tokens': [s['input_ids'] for s in samples],  # texts generated from run_generate_texts.py
            'target_texts': [s[answer_col] for s in samples],
            'greedy_texts': [s['reply'] for s in samples],
            'greedy_tokens': [s['input_ids'][len(t):] for s, t in zip(samples, tokens)],
        }

        # Stat calculators (once per batch)
        for stat_calc in stat_calculators:
            t0 = time.time()
            log.info(f"Calculating {stat_calc}...")
            stats.update(stat_calc(stats, texts, model, max_new_tokens=1024))
            log.info(f"Done calculating in {round(time.time() - t0, 2)}s")

        # Estimators → results list with one value per batch item
        for est in estimators:
            t0 = time.time()
            log.info(f"Calculating {est}...")
            results = list(est(stats))  # length == len(batch_indices)
            if isinstance(est, UHeadEstimator):
                col_name = uhead_path
            else:
                col_name = str(est)

            # Assign results[j] → dataset row batch_indices[j]
            for j, val in enumerate(results):
                col_buffers[col_name][batch_indices[j]] = val

            log.info(f"Done calculating in {round(time.time() - t0, 2)}s")

        for j in range(len(batch_indices)):
            col_buffers['saved_stats'][batch_indices[j]] = {}
            for k, v in stats.items():
                if any(x in k for x in [
                    'log_probs',
                    'attention',
                    'embeddings',
                    'tokenizer',
                    'entailment_id',
                    'uhead_features',
                    'greedy_tokens_alternatives',
                ]):
                    continue
                try:
                    val = v[j]
                except Exception as e:
                    continue
                if k == 'claims':
                    val = [
                        {key: getattr(c, key) for key in ['claim_text', 'sentence', 'aligned_token_ids']}
                        for c in val
                    ]
                col_buffers['saved_stats'][batch_indices[j]][k] = val

        processed_since_save += len(batch_indices)

        if processed_since_save >= save_every:
            log.info("Writing updated columns and saving (periodic)...")
            for col, buf in col_buffers.items():
                if col in dataset.column_names:
                    dataset = dataset.remove_columns(col)
                dataset = dataset.add_column(col, buf)
            periodic_save(dataset, save_path)
            processed_since_save = 0

    # Final write-back and save
    log.info("Final write-back and save...")
    for col, buf in col_buffers.items():
        if col in dataset.column_names:
            dataset = dataset.remove_columns(col)
        dataset = dataset.add_column(col, buf)

    periodic_save(dataset, save_path)
    return dataset


def main(args):
    dataset = load_bon_dataset(args.dataset_path, args.dataset_split)
    prompt = load_prompt(args.prompt_file)
    model = load_model(args.model_path, args.device)

    factory = FactoryStatCalculator(BuilderEnvironmentStatCalculator(model=model))
    stat_containers = register_default_stat_calculators(model_type="Whitebox")
    stat_containers.extend([
        create_container(
            ProxyCalculator,
            builder="bestofn_optimized.proxy_calculator",
            default_config={
                "predict_token_uncertainties": False,
                "uq_head_path": args.uhead_path,
            },
        ),
        create_container(
            StepsExtractor,
            builder="synthetic_dataset_generation.utils.steps_extractor",
            default_config={},
        ),
        create_container(
            CalculatorApplyUQHead,
            builder="luh.builder_CalculatorApplyUQHead",
            default_config={},
        )
    ])
    estimators = [
        MaximumSequenceProbability(),
        MeanTokenEntropy(),
        Perplexity(),
        PTrue(),
        # ClaimConditionedProbability(),  # optional, heavy
        UHeadEstimator(reduction='max'),
    ]
    stat_calculators = initialize_stat_calculators(factory, stat_containers, model, estimators)[0]

    print("Will calculate:")
    for stat_calc in stat_calculators:
        print(f" - {stat_calc}")

    update_with_estimators(
        dataset, prompt, model,
        stat_calculators, estimators,
        args.save_path,
        args.uhead_path,
        args.batch_size,
        args.question_col,
        args.answer_col,
        args.save_every,
    )
    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
