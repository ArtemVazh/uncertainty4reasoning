import torch
import torch.nn.functional as F

from typing import Dict, List, Tuple
from parse import parse
import numpy as np
import logging
import time
import threading

from baselines.skywork_prm.skywork_prm import SkyworkO1_7B, SkyworkO1_1_5B
from lm_polygraph.stat_calculators.extract_claims import Claim
from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from baselines.skywork_prm.io_utils import prepare_batch_input_for_model, derive_step_rewards, prepare_input

log = logging.getLogger()


class PRMStatCalculator(StatCalculator):
    def __init__(
            self,
            prompt_path: str | None = None,
            model_path: str = "Qwen/Qwen2.5-Math-7B-PRM800K",
            device: str = "auto",
    ):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.prm_tokenizer = None
        self.prm_model = None
        self.prompt = open(prompt_path, 'r').read() if prompt_path else "{q}"

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return ["prm_scores"], ["claims"]

    def init(self):
        if self.prm_model is not None:
            return
        device = self.device
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


class MathShepherdPRMCalculator(StatCalculator):
    def __init__(
            self,
            prompt_path: str | None = None,
            model_path: str = "peiyi9979/math-shepherd-mistral-7b-prm",
            device: str = "auto",
    ):
        super().__init__()
        self.model_path = model_path
        self.device = device
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.prompt = open(prompt_path, 'r').read() if prompt_path else "{q}"
        self.step_tag = "ки"
        self.good_token = "+"
        self.bad_token = "-"
        self.step_tag_id = None
        self.candidate_token_ids = None

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return ["prm_scores"], ["claims"]

    def init(self):
        if self.model is not None:
            return
        device = self.device
        log.info(f"Initializing {self.model_path} model on device={device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(device).eval()
        self.step_tag_id = self.tokenizer.encode(self.step_tag)[-1]
        self.candidate_token_ids = self.tokenizer.encode(f"{self.good_token} {self.bad_token}")[1:]  # skip BOS

    def get_rewards(self, question: str, steps: list[Claim]) -> list[float]:
        self.init()

        if len(steps) == 0:
            return []

        # Reconstruct the output with step separator
        output_text = ""
        for i, step in enumerate(steps):
            output_text += f"Step {i + 1}: {step.claim_text.strip()} {self.step_tag}\n"
        input_text = f"{question.strip()} {output_text.strip()}"

        input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            logits = self.model(input_ids).logits[:, :, self.candidate_token_ids]
            probs = F.softmax(logits, dim=-1)[:, :, 0]  # probability of '+'

        step_mask = input_ids == self.step_tag_id
        step_scores = probs[step_mask]

        return step_scores.cpu().tolist()

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


class SkyworkPRMStatCalculator(StatCalculator):
    def __init__(
            self,
            prompt_path: str | None = None,
            model_path: str = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B",
            device: str = "auto",
    ):
        super().__init__()
        self.prompt = open(prompt_path, "r").read() if prompt_path else "{q}"
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self.step_token = "\n"

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return ["prm_scores"], ["claims"]

    def init(self):
        if self.model is not None:
            return
        print(f"Initializing Skywork PRM model from {self.model_path}...")

        start_time = time.time()
        def keep_alive(start_time):
            while not done_loading[0]:
                elapsed = time.time() - start_time
                print(f"\rStill loading... (elapsed: {elapsed:.1f}s)", flush=True)
                time.sleep(1)
        done_loading = [False]
        thread = threading.Thread(target=keep_alive, args=(start_time,))
        thread.start()

        if '1.5B' in self.model_path:
            cls = SkyworkO1_1_5B
        else:
            cls = SkyworkO1_7B
        self.model, self.tokenizer = cls.load_model_and_tokenizer()

        done_loading[0] = True
        thread.join()
        print('Done!')

    def get_rewards(self, question: str, steps: List[Claim]) -> List[float]:
        self.init()
        if len(steps) == 0:
            return []

        # Reconstruct full response from Claim list
        response = ""
        for i, step in enumerate(steps):
            response += step.claim_text.strip() + self.step_token

        print('Preparing input...')

        processed = prepare_input(question, response, tokenizer=self.tokenizer, step_token=self.step_token)
        input_ids, step_locs, reward_flags = processed

        print('Preparing batch input...')
        # Prepare batch-compatible inputs
        input_ids_batch, attention_mask, reward_flags = prepare_batch_input_for_model(
            [input_ids], [reward_flags], self.tokenizer.pad_token_id
        )

        print('Running model...')

        device = self.model.pretrained_model.device
        with torch.no_grad():
            _, _, rewards = self.model(
                input_ids=input_ids_batch.to(device),
                attention_mask=attention_mask.to(device),
                return_probs=True,
            )

        print('Deriving step rewards...')

        step_rewards = derive_step_rewards(rewards.detach().to("cpu", dtype=torch.float32), reward_flags)
        return step_rewards[0].tolist()  # Single input

    def __call__(self, dependencies: Dict[str, np.array], texts: List[str], model: Model, max_new_tokens: int = 100,
                 **kwargs) -> Dict[str, np.ndarray]:
        self.init()
        rewards: list[list[float]] = []
        for input_text, claims in zip(texts, dependencies["claims"]):
            question = parse(self.prompt, input_text).named["q"]
            r = self.get_rewards(question, claims)
            assert len(r) == len(claims)
            rewards.append(r)
        return {"prm_scores": rewards}


def load_prm_calculator_by_model_path(
        prompt_path: str | None = None,
        model_path: str = "Qwen/Qwen2.5-Math-7B-PRM800K",
        device: str = "auto",
):
    if model_path.startswith("Qwen/"):
        return PRMStatCalculator(
            prompt_path=prompt_path,
            model_path=model_path,
            device=device,
        )
    elif model_path.startswith("peiyi9979/"):
        return MathShepherdPRMCalculator(
            prompt_path=prompt_path,
            model_path=model_path,
            device=device,
        )
    elif "Skywork" in model_path:
        return SkyworkPRMStatCalculator(
            prompt_path=prompt_path,
            model_path=model_path,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported model path prefix for PRM model: {model_path}")


def load_stat_calculator(config, builder):
    return PRMStatCalculator(
        prompt_path=config.prompt_path,
        model_path=config.get("model_path", "Qwen/Qwen2.5-Math-7B-PRM800K"),
        device=config.get("device", "auto"),
    )
