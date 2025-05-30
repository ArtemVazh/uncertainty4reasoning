import copy

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.utils.model import Model
from transformers import GenerationConfig

from luh import AutoUncertaintyHead

from typing import Dict, List, Tuple
import torch
import time
import numpy as np
import os
import logging

log = logging.getLogger()


class SampleGenerationCalculator(StatCalculator):
    def __init__(
            self,
            uncertainty_head,
            n_alternatives=10,
            tokenize=True,
            args_generate=dict(),
            predict_token_uncertainties=True,
            device="cuda",
            top_k: int = 50,
            top_p: float = 0.95,
            temperature: float = 1.0,
    ):
        super().__init__()

        self.n_alternatives = n_alternatives
        self._tokenize = tokenize
        self.args_generate = args_generate

        self.uncertainty_head = uncertainty_head.to(device)
        self.uncertainty_head.eval()
        self.output_attentions = self.uncertainty_head.output_attentions
        self.predict_token_uncertainties = predict_token_uncertainties

        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return [
            "hidden_states",
            "greedy_log_probs",
            "greedy_logits",
            "greedy_tokens",
            "greedy_tokens_alternatives",
            "greedy_texts",
            "greedy_log_likelihoods",
            "uncertainty_logits",
            "uhead_features",
            "input_texts",
            "input_tokens",
        ], []

    def postprocess_predictions(self, batch, out, tokenizer):
        logits = torch.stack(out.scores, dim=1)
        sequences = out.sequences

        cut_logits, cut_sequences, cut_texts, cut_alternatives, ll = [], [], [], [], []
        for i in range(batch['input_ids'].shape[0]):
            idx = batch["input_ids"].shape[1]
            seq = sequences[i, idx:].cpu()
            length = next((j + 1 for j, token in enumerate(seq) if token == tokenizer.eos_token_id), len(seq))
            cut_seq = seq[:length]
            cut_sequences.append(cut_seq.tolist())
            cut_texts.append(tokenizer.decode(cut_seq))
            cut_logits.append(logits[i, :length, :].cpu().numpy())

            alt = []
            for j in range(length):
                lt = logits[i, j, :].cpu().numpy()
                best_tokens = np.argpartition(lt, -self.n_alternatives)[-self.n_alternatives:]
                best_tokens = best_tokens[np.argsort(-lt[best_tokens])]
                alt_j = [(t.item(), lt[t].item()) for t in best_tokens]
                alt_j.sort(key=lambda x: x[0] == cut_seq[j].item(), reverse=True)
                alt.append(alt_j)
            cut_alternatives.append(alt)
            ll.append([cut_logits[-1][j, cut_seq[j]] for j in range(len(cut_seq))])

        return {
            "input_tokens": batch["input_ids"].to("cpu").tolist(),
            "greedy_log_probs": cut_logits,
            "greedy_tokens": cut_sequences,
            "greedy_tokens_alternatives": cut_alternatives,
            "greedy_texts": cut_texts,
            "greedy_log_likelihoods": ll,
            "logits": logits[:, :-1, :],
        }

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: Model, max_new_tokens: int = 100,
                 **kwargs) -> Dict[str, np.ndarray]:
        cache = None

        batch = model.tokenize(texts) if self._tokenize else texts
        device_batch = batch.to(model.device())
        log.info(f"Generating {max_new_tokens} new tokens on device={model.device()}...")

        # Overwrite new parameters
        old_params: GenerationParameters = model.generation_parameters
        params = copy.deepcopy(old_params)
        params.top_p, params.top_k, params.temperature = self.top_p, self.top_k, self.temperature
        model.generation_parameters = params

        start_time = time.time()
        with torch.no_grad():
            out = model.generate(
                **device_batch,
                output_scores=True,
                return_dict_in_generate=True,
                output_attentions=self.output_attentions,
                output_hidden_states=True,
                do_sample=True,
                suppress_tokens=(
                    []
                    if model.generation_parameters.allow_newlines
                    else [
                        t
                        for t in range(len(model.tokenizer))
                        if "\n" in model.tokenizer.decode([t])
                    ]
                ),
                pad_token_id=model.tokenizer.eos_token_id,
                tokenizer=model.tokenizer,
                **self.args_generate,
            )
        model.generation_parameters = old_params
        log.info(f"Done generating in {round(time.time() - start_time, 2)} seconds")

        result_dict = self.postprocess_predictions(batch, out, model.tokenizer)
        result_dict["input_texts"] = texts

        if cache:
            for i in range(len(texts)):
                cache.get(texts[i], lambda: result_dict["greedy_tokens"][i])

        output_bounds = []
        full_attn_mask = torch.zeros_like(out.sequences).bool()
        for i in range(batch['input_ids'].shape[0]):
            idx = batch["input_ids"].shape[1]
            full_attn_mask[i, :idx] = batch["attention_mask"][i]
            length = len(result_dict["greedy_tokens"][i])
            full_attn_mask[i][idx: idx + length] = 1
            output_bounds.append((idx - 1, idx + length - 1))

        out["full_attention_mask"] = full_attn_mask
        out["context_lengths"] = torch.tensor([len(it) for it in batch["input_ids"]])
        batch["context_lenghts"] = out["context_lengths"]

        if self.predict_token_uncertainties:
            with torch.no_grad():
                uncertainty_logits = self.uncertainty_head(batch, out)
                result_dict["uncertainty_logits"] = [
                    ue[output_bounds[i][0]: output_bounds[i][1]]
                    for i, ue in enumerate(uncertainty_logits.cpu().detach().squeeze(-1))
                ]
        else:
            result_dict["uhead_features"] = self.uncertainty_head.feature_extractor(batch, out)
            result_dict["llm_inputs"] = batch
            result_dict["full_attention_mask"] = full_attn_mask

        return result_dict


def load_stat_calculator(config, builder):
    uncertainty_head = AutoUncertaintyHead.from_pretrained(
        config.uq_head_path,
        builder.model.model)
    builder.uncertainty_head = uncertainty_head
    return SampleGenerationCalculator(
        uncertainty_head=uncertainty_head,
        tokenize=True,
        args_generate=config.args_generate,
        predict_token_uncertainties=config.predict_token_uncertainties
    )
