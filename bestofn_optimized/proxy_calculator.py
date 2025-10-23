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


class ProxyCalculator(StatCalculator):
    def __init__(
            self,
            uncertainty_head,
            n_alternatives=10,
            tokenize=True,
            predict_token_uncertainties=True,
            device="cuda",
            top_k: int = 50,
            top_p: float = 0.95,
            temperature: float = 1.0,
    ):
        super().__init__()

        self.n_alternatives = n_alternatives
        self._tokenize = tokenize

        self.uncertainty_head = uncertainty_head
        if self.uncertainty_head is not None:
            self.uncertainty_head = self.uncertainty_head.to(device)
            self.uncertainty_head.eval()
            self.output_attentions = self.uncertainty_head.output_attentions
            self.predict_token_uncertainties = predict_token_uncertainties
        else:
            self.output_attentions = False

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

    def _proxy_call(
            self,
            dependencies: Dict[str, np.array],
            texts: List[str],
            model: Model,
            n_alternatives: int,
    ) -> Dict[str, np.ndarray]:
        if 'proxy_tokens' not in dependencies.keys():
            raise Exception(
                "No 'hyp_texts' found in depencendies. "
                "Only proxy-model generations are supported."
            )

        input_tokens = [model.tokenizer(t)["input_ids"] for t in texts]

        hyp_tokens = dependencies['proxy_tokens']
        assert len(input_tokens) == len(hyp_tokens)
        for i in range(len(input_tokens)):
            assert hyp_tokens[i][:len(input_tokens[i])] == input_tokens[i]
            hyp_tokens[i] = hyp_tokens[i][len(input_tokens[i]):]
        hyp_texts = [model.tokenizer.decode(t) for t in hyp_tokens]

        combined_tokens = [it + ht for it, ht in zip(input_tokens, hyp_tokens)]
        combined_batch = model.tokenizer.pad(
            {"input_ids": combined_tokens},
            padding=True,
            return_tensors="pt",
        )
        combined_batch = {k: v.to(model.device()) for k, v in combined_batch.items()}

        log.info(f"Infering LLM with new tokens of shape {combined_batch['input_ids'].shape} on device={model.device()}...")
        start_time = time.time()
        with torch.no_grad():
            out = model(**combined_batch, output_attentions=self.output_attentions, output_hidden_states=True)
            logits = out.logits.log_softmax(-1)
        log.info(f"Done infering in {round(time.time() - start_time, 2)} seconds")

        cut_logits = []
        cut_sequences = []
        cut_texts = []
        cut_alternatives = []
        for i in range(len(texts)):
            begin_pos = len(input_tokens[i])
            end_pos = begin_pos + len(hyp_tokens[i])
            cut_sequences.append(hyp_tokens[i])
            cut_texts.append(hyp_texts[i])
            cut_logits.append(logits[i][begin_pos - 1:end_pos - 1].cpu().numpy())
            cut_alternatives.append([[] for _ in range(begin_pos, end_pos)])

            for j in range(begin_pos, end_pos):
                lt = logits[i, j - 1, :].cpu().numpy()
                best_tokens = np.argpartition(lt, -n_alternatives)[-n_alternatives:]
                best_tokens = best_tokens[np.argsort(-lt[best_tokens])].tolist()

                # as hyp_texts are not necessarily greedy, so
                # need to make sure that first token is from hyp_texts
                cur_token = hyp_tokens[i][j - begin_pos]
                if cur_token not in best_tokens:
                    best_tokens = [cur_token] + best_tokens[:-1]
                else:
                    best_tokens = [cur_token] + [t for t in best_tokens if t != cur_token]

                for t in best_tokens:
                    cut_alternatives[-1][j - begin_pos].append((t, lt[t].item()))

        ll = []
        for i in range(len(texts)):
            log_probs = cut_logits[i]
            tokens = cut_sequences[i]
            assert len(tokens) == len(log_probs)
            ll.append([log_probs[j, tokens[j]] for j in range(len(log_probs))])

        result_dict = {
            "input_tokens": input_tokens,
            "greedy_log_probs": cut_logits,
            "greedy_tokens": cut_sequences,
            "greedy_tokens_alternatives": cut_alternatives,
            "greedy_texts": cut_texts,
            "greedy_log_likelihoods": ll,
        }

        if self.uncertainty_head is not None:
            out["full_attention_mask"] = combined_batch["attention_mask"]
            out["context_lengths"] = torch.tensor([len(it) for it in input_tokens])
            combined_batch["context_lenghts"] = out["context_lengths"]

            if self.predict_token_uncertainties:
                raise NotImplemented()
            else:
                result_dict["uhead_features"] = self.uncertainty_head.feature_extractor(combined_batch, out)
                result_dict["llm_inputs"] = combined_batch
                result_dict["full_attention_mask"] = out["full_attention_mask"]

        return result_dict

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: Model, max_new_tokens: int = 100,
                 **kwargs) -> Dict[str, np.ndarray]:
        return self._proxy_call(dependencies, texts, model, max_new_tokens)


def load_stat_calculator(config, builder):
    if config.uq_head_path is None:
        uncertainty_head = None
    else:
        uncertainty_head = AutoUncertaintyHead.from_pretrained(
            config.uq_head_path,
            builder.model.model)
        builder.uncertainty_head = uncertainty_head
    return ProxyCalculator(
        uncertainty_head=uncertainty_head,
        tokenize=True,
        predict_token_uncertainties=config.predict_token_uncertainties
    )
