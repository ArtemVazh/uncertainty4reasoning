import torch
import torch.nn.functional as F

from typing import Dict, List, Tuple
from parse import parse
import numpy as np
import logging

from lm_polygraph.stat_calculators.extract_claims import Claim
from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model
from transformers import AutoTokenizer, AutoModel

log = logging.getLogger()


class PRMStatCalculator(StatCalculator):
    def __init__(
            self,
            prompt_path: str | None = None,
            model_path: str = "Qwen/Qwen2.5-Math-7B-PRM800K",
            device: str = "auto",
            offload_to_cpu_between_calls: bool = False,
    ):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.prm_tokenizer = None
        self.prm_model = None
        self.prompt = open(prompt_path, 'r').read() if prompt_path else "{q}"
        self.offload_to_cpu_between_calls = offload_to_cpu_between_calls
        if offload_to_cpu_between_calls:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return ["prm_scores"], ["claims"]

    def init(self):
        if self.prm_model is not None:
            return
        device = "cpu" if self.offload_to_cpu_between_calls else self.device
        log.info(f"Initializing {self.model_path} model on device={self.device}")
        self.prm_tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.prm_model = AutoModel.from_pretrained(
            self.model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()

    def make_step_rewards(self, logits, token_masks):
        self.init()
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities * token_masks.unsqueeze(-1)
        all_scores_res = []
        for i in range(probabilities.size(0)):
            sample = probabilities[i]
            positive_probs = sample[sample != 0].view(-1, 2)[:, 1]
            non_zero_elements_list = positive_probs.cpu().tolist()
            all_scores_res.append(non_zero_elements_list)
        return all_scores_res

    def get_rewards(self, question: str, steps: list[Claim]) -> list[float]:
        self.init()
        if self.offload_to_cpu_between_calls:
            log.info(f"Uploading PRM model to {self.device}...")
            self.prm_model = self.prm_model.to(self.device)
            log.info('Done.')
        if len(steps) == 0:
            return []
        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "<extra_0>".join([c.claim_text for c in steps]) + "<extra_0>"},
        ]
        conversation_str = self.prm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = self.prm_tokenizer.encode(conversation_str, return_tensors="pt").to(self.prm_model.device)
        with torch.no_grad():
            outputs = self.prm_model(input_ids=input_ids)
        step_sep_id = self.prm_tokenizer.encode("<extra_0>")[0]
        token_masks = (input_ids == step_sep_id)
        step_reward = self.make_step_rewards(outputs[0], token_masks)
        if self.offload_to_cpu_between_calls:
            log.info(f"Offloading PRM model to cpu...")
            self.prm_model = self.prm_model.cpu()
            log.info('Done.')
        return step_reward[0]

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: Model, max_new_tokens: int = 100,
                 **kwargs) -> Dict[str, np.ndarray]:
        self.init()
        rewards: list[list[float]] = []
        for input_text, claims in zip(texts, dependencies["claims"]):
            question = parse(self.prompt, input_text).named['q']
            r = self.get_rewards(question, claims)
            assert len(r) == len(claims)
            rewards.append(r)
        return {"prm_scores": rewards}


def load_stat_calculator(config, builder):
    return PRMStatCalculator(
        prompt_path=config.prompt_path,
        model_path=config.get("model_path", "Qwen/Qwen2.5-Math-7B-PRM800K"),
        device=config.get("device", "auto"),
        offload_to_cpu_between_calls=config.get("offload_to_cpu_between_calls", False),
    )
