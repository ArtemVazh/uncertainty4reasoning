import torch
import torch.nn as nn

from typing import Dict, List, Tuple
from parse import parse
import numpy as np
import logging

from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model
from transformers import (
    MistralModel, MistralPreTrainedModel,
    LlamaModel, LlamaPreTrainedModel,
    AutoTokenizer
)
from transformers.configuration_utils import PretrainedConfig

log = logging.getLogger()


class ReasonEval_7B(MistralPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['lm_head.weight']

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.model = MistralModel(config)
        self.score_head = nn.Linear(config.hidden_size, config.score_dimension, bias=config.use_bias)
        self.post_init()

    def forward(self, input_ids, attention_mask, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        scores = self.score_head(hidden_states)
        return scores


class ReasonEval_34B(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['lm_head.weight']

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)
        self.score_head = nn.Linear(config.hidden_size, config.score_dim, bias=config.bias)
        self.post_init()

    def forward(self, input_ids, attention_mask, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        scores = self.score_head(hidden_states)
        return scores


class ReasonEvalStatCalculator(StatCalculator):
    def __init__(
            self,
            prompt_path: str | None = None,
            reasoneval_model_path: str = "GAIR/ReasonEval-7B",
            device: str = "auto",
            offload_to_cpu_between_calls: bool = False,
    ):
        super().__init__()
        self.reasoneval_model_path = reasoneval_model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self.prompt = open(prompt_path, 'r').read() if prompt_path else "{q}"
        self.offload_to_cpu_between_calls = offload_to_cpu_between_calls
        if offload_to_cpu_between_calls:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return ["reasoneval_scores"], ["claims"]

    def init(self):
        if self.model is not None:
            return
        device = "cpu" if self.offload_to_cpu_between_calls else self.device
        log.info(f"Initializing {self.reasoneval_model_path} model on device={self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.reasoneval_model_path, device_map=device)
        if self.reasoneval_model_path.endswith('7B'):
            self.model = ReasonEval_7B.from_pretrained(self.reasoneval_model_path, device_map=device)
            self.model_size = '7B'
        elif self.reasoneval_model_path.endswith('34B'):
            self.model = ReasonEval_34B.from_pretrained(self.reasoneval_model_path, device_map=device)
            self.model_size = '34B'
        else:
            raise ValueError(f"Could not determine model size from path: {self.reasoneval_model_path}")

    def get_step_level_scores(self, question, reasoning_steps) -> list[dict[str, float]]:
        self.init()
        PROMPT_FORMAT = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
        step_separator = f"{self.tokenizer.pad_token}"
        combined_steps = "".join(step.claim_text + step_separator for step in reasoning_steps)
        prompt = PROMPT_FORMAT.format(input=question)
        tokenized_result = self.tokenizer(prompt + step_separator + combined_steps)['input_ids']
        separator_token_id = self.tokenizer(step_separator)['input_ids'][-1]

        labeled_token_indices = []
        adjusted_token_ids = []
        separator_count = 0
        for idx, token_id in enumerate(tokenized_result):
            if token_id == separator_token_id:
                labeled_token_indices.append(idx - 1 - separator_count)
                separator_count += 1
            else:
                adjusted_token_ids.append(token_id)

        if self.model_size == '7B':
            adjusted_token_ids = [1] + adjusted_token_ids
            adjusted_token_ids = torch.tensor([adjusted_token_ids])
            labeled_token_indices = labeled_token_indices[2:]
        elif self.model_size == '34B':
            adjusted_token_ids = torch.tensor([adjusted_token_ids])
            labeled_token_indices = labeled_token_indices[1:]
        else:
            raise ValueError(f"Invalid model size: {self.model_size}")

        attention_mask = adjusted_token_ids.new_ones(adjusted_token_ids.size(), dtype=torch.bool)
        adjusted_token_ids = adjusted_token_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        with torch.no_grad():
            reasoning_scores = self.model(adjusted_token_ids, attention_mask)[0, labeled_token_indices, :]
            scores = torch.softmax(reasoning_scores, dim=-1).tolist()

        step_level_validity_scores = [(score[1] + score[2]) for score in scores]
        step_level_redundancy_scores = [score[1] for score in scores]
        return [{'validity': v, 'redundancy': r} for v, r in
                zip(step_level_validity_scores, step_level_redundancy_scores)]

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: Model, max_new_tokens: int = 100,
                 **kwargs) -> Dict[str, np.ndarray]:
        self.init()
        if self.offload_to_cpu_between_calls:
            log.info(f"Uploading ReasonEval model to {self.device}...")
            self.model = self.model.to(self.device)
            log.info(f"Done.")
        scores: list[list[dict]] = []
        for input_text, claims in zip(texts, dependencies["claims"]):
            question = parse(self.prompt, input_text).named['q']
            r = self.get_step_level_scores(question, claims)
            assert len(r) == len(claims)
            scores.append(r)
        if self.offload_to_cpu_between_calls:
            log.info(f"Offloading ReasonEval model to cpu...")
            self.model = self.model.cpu()
            log.info(f"Done.")
        return {"reasoneval_scores": scores}


def load_stat_calculator(config, builder):
    return ReasonEvalStatCalculator(
        prompt_path=config.prompt_path,
        reasoneval_model_path=config.get("model_path", "GAIR/ReasonEval-7B"),
        device=config.get("device", "auto"),
        offload_to_cpu_between_calls=config.get("offload_to_cpu_between_calls", False),
    )
