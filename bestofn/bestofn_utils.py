import time
import torch
from tqdm import tqdm
import logging
import numpy as np

from datasets import Dataset

from lm_polygraph import WhiteboxModel
from lm_polygraph.estimators import Estimator
from lm_polygraph.stat_calculators import StatCalculator
from utils import parse_ans

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("bestofn_eval")


def is_correct_answer(generated_output: str, gold_answer: str) -> bool:
    pred = parse_ans(generated_output)
    gold = parse_ans(gold_answer)
    if pred is None or gold is None:
        return False
    return np.isclose(pred, gold).item()


def process_stats(stats):
    return {
        k: v for k, v in stats.items()
        if k in [
            "greedy_texts",
            "greedy_tokens",
            "greedy_logprobs",
            "uncertainty_claim_logits",
            "reasoneval_scores",
            "prm_scores",
        ]
    }


def _update_sample(
        r: dict,
        model: WhiteboxModel,
        estimators: list[Estimator],
        stat_calculators: list[StatCalculator],
        n: int,
        max_new_tokens: int = 100,
        verbose: bool = True,
):
    stats = r["stats"]
    input_text = r["input"]
    texts = [input_text for _ in range(n)]

    # stat calculators
    for stat_calc in stat_calculators:
        start_time = time.time()
        if verbose:
            log.info(f"Calculating {stat_calc}...")
        stats.update(stat_calc(stats, texts, model, max_new_tokens=max_new_tokens))
        if verbose:
            log.info(f"Done calculating in {round(time.time() - start_time, 2)} seconds")
    r["stats"] = process_stats(stats)

    # estimations
    estimations = {str(est): est(stats) for est in estimators}
    r["scores"].update(estimations)

    r.update({
        "sample_texts": r["stats"]["greedy_texts"],
        "correctness": [is_correct_answer(t, r["gold_answer"]) for t in r["stats"]["greedy_texts"]]
    })

    return r


def _bestofn(
        dataset: Dataset,
        model: WhiteboxModel,
        estimators: list[Estimator],
        stat_calculators: list[StatCalculator],
        save_path: str,
        save_frequency: int | None,
        n: int,
        max_new_tokens,
        results: list[dict],
        verbose: bool = True,
):
    assert len(dataset) == len(results)
    log.info(f"Processing {len(dataset)} samples with {n} completions each...")

    for i, (sample, r) in tqdm(enumerate(zip(dataset, results)), total=len(dataset)):
        results[i] = _update_sample(r, model, estimators, stat_calculators, n, max_new_tokens, verbose)

        if (save_frequency is not None and (i + 1) % save_frequency == 0) or i + 1 == len(dataset):
            if verbose:
                log.info(f"Saving results to {save_path}")
            torch.save(results, save_path)

    log.info("Done.")


def bestofn(
        dataset: Dataset,
        model: WhiteboxModel,
        estimators: list[Estimator],
        stat_calculators: list[StatCalculator],
        save_path: str,
        save_frequency: int | None,
        n: int,
        max_new_tokens: int = 100,
        verbose: bool = True,
):
    results = [{
        "input": sample["question"],
        "gold_answer": sample["answer"],
        "scores": {},
        "stats": {},
    } for sample in dataset]
    _bestofn(
        dataset, model, estimators, stat_calculators,
        save_path, save_frequency,
        n, max_new_tokens, results,
        verbose=verbose,
    )


def update_bestofn(
        dataset: Dataset,
        model: WhiteboxModel,
        estimators: list[Estimator],
        stat_calculators: list[StatCalculator],
        save_path: str,
        save_frequency: int | None,
        verbose: bool = True,
):
    results = torch.load(save_path)
    n = len(results[0]["sample_texts"])
    _bestofn(
        dataset, model, estimators, stat_calculators,
        save_path, save_frequency,
        n, max_new_tokens=100, results=results,
        verbose=verbose,
    )
