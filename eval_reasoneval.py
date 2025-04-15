import argparse
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    MistralModel, MistralPreTrainedModel,
    LlamaModel, LlamaPreTrainedModel,
    AutoTokenizer
)
from transformers.configuration_utils import PretrainedConfig
from utils import load_manager, extract_steps, extract_questions


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


def get_step_level_scores(question, reasoning_steps, tokenizer, model, model_size):
    PROMPT_FORMAT = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
    step_separator = f"{tokenizer.pad_token}"
    combined_steps = "".join(step + step_separator for step in reasoning_steps)
    prompt = PROMPT_FORMAT.format(input=question)
    tokenized_result = tokenizer(prompt + step_separator + combined_steps)['input_ids']
    separator_token_id = tokenizer(step_separator)['input_ids'][-1]

    labeled_token_indices = []
    adjusted_token_ids = []
    separator_count = 0
    for idx, token_id in enumerate(tokenized_result):
        if token_id == separator_token_id:
            labeled_token_indices.append(idx - 1 - separator_count)
            separator_count += 1
        else:
            adjusted_token_ids.append(token_id)

    if model_size == '7B':
        adjusted_token_ids = [1] + adjusted_token_ids
        adjusted_token_ids = torch.tensor([adjusted_token_ids])
        labeled_token_indices = labeled_token_indices[2:]
    elif model_size == '34B':
        adjusted_token_ids = torch.tensor([adjusted_token_ids])
        labeled_token_indices = labeled_token_indices[1:]
    else:
        raise ValueError(f"Invalid model size: {model_size}")

    attention_mask = adjusted_token_ids.new_ones(adjusted_token_ids.size(), dtype=torch.bool)
    adjusted_token_ids = adjusted_token_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        reasoning_scores = model(adjusted_token_ids, attention_mask)[0, labeled_token_indices, :]
        scores = torch.softmax(reasoning_scores, dim=-1).tolist()

    step_level_validity_scores = [(score[1] + score[2]) for score in scores]
    step_level_redundancy_scores = [score[1] for score in scores]
    return [{'validity': v, 'redundancy': r} for v, r in zip(step_level_validity_scores, step_level_redundancy_scores)]


def main():
    parser = argparse.ArgumentParser(description="Run ReasonEval step-level scoring for reasoning chains.")

    parser.add_argument('--hf-manager-path', type=str, required=True, help="HuggingFace repo for the UE manager file")
    parser.add_argument('--base-model-path', type=str, required=True, help="Path or name of the base model")
    parser.add_argument('--save-path', type=str, required=True, help="Path to save the output rewards JSON")
    parser.add_argument('--reasoneval-model-path', type=str, default='GAIR/ReasonEval-7B',
                        help='Path to the ReasonEval model.')
    parser.add_argument('--device', type=str, default="auto", help="Device map setting for model loading")
    parser.add_argument('--prompt-file', type=str, default="configs/gsm8k_3shot_prompt.txt",
                        help="Path to prompt template file")
    parser.add_argument('--hf-cache', type=str, default=None, help="Cache directory for HF models")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.reasoneval_model_path, cache_dir=args.hf_cache,
                                              device_map=args.device)
    if args.reasoneval_model_path.endswith('7B'):
        model = ReasonEval_7B.from_pretrained(args.reasoneval_model_path, cache_dir=args.hf_cache,
                                              device_map=args.device)
        model_size = '7B'
    elif args.reasoneval_model_path.endswith('34B'):
        model = ReasonEval_34B.from_pretrained(args.reasoneval_model_path, cache_dir=args.hf_cache,
                                               device_map=args.device)
        model_size = '34B'
    else:
        raise ValueError(f"Could not determine model size from path: {args.reasoneval_model_path}")

    man = load_manager(args.hf_manager_path)
    steps = extract_steps(man, args.base_model_path, args.hf_cache)
    questions = extract_questions(man, args.prompt_file)

    scores = []
    for i in tqdm(range(len(questions)), desc='Evaluating ReasonEval'):
        s = get_step_level_scores(questions[i], steps[i], tokenizer, model, model_size)
        assert len(s) == len(steps[i])
        scores.append(s)

    with open(args.save_path, 'w') as f:
        json.dump(scores, f)
    print('Saved to {}'.format(args.save_path))


if __name__ == "__main__":
    main()
