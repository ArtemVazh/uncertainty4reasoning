import torch
import torch.nn.functional as F

from parse import parse
import numpy as np
import logging

from lm_polygraph.stat_calculators.extract_claims import Claim
from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model

import re
import io
import math
import signal
from contextlib import redirect_stdout
from typing import Dict, List, Tuple

from transformers import AutoTokenizer, AutoModelForCausalLM

log = logging.getLogger()


class _Timeout:
    """Simple timeout guard for code execution."""

    def __init__(self, seconds: int = 5):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handle)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc, tb):
        signal.alarm(0)

    def _handle(self, signum, frame):
        raise TimeoutError("Code execution timed out")


class _CodeExecutor:
    """Execute the last ```python ...``` block and capture stdout."""

    def __init__(self):
        self.namespace = {}
        self._pat = re.compile(r"```python\s*(.*?)\s*```", re.DOTALL)

    def execute(self, text: str) -> str:
        try:
            code_block = self._pat.findall(text)[-1].strip()
        except Exception:
            return "Code format error: No code found."
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                with _Timeout(seconds=5):
                    exec(code_block, self.namespace)
            return buf.getvalue().strip()
        except TimeoutError as te:
            return f"Code execute time out: {te}"
        except Exception as e:
            return f"Code execute Error: {type(e).__name__}: {e}"


def _trim_eos(prompt: str, tokenizer) -> str:
    eos = tokenizer.eos_token or ""
    if eos:
        if prompt.endswith(f"{eos}\n"):
            return prompt[: -len(eos) - 1]
        if prompt.endswith(eos):
            return prompt[: -len(eos)]
    return prompt


class GenPRMStatCalculator(StatCalculator):
    """
    GenPRM step scorer using HuggingFace transformers (no vLLM).
    Mechanics:
      - Build chat prompt (system + user) with paragraphs up to current step
      - Stage 1: analyze (generate until newline, keep text)
      - Stage 2: verify (optionally generate code, execute it, append [Code Output])
      - Stage 3: output -> '**Judgement**: \\boxed{Yes|No}'
      - Reward = softmax over the *generated* Yes/No token logits at the judgment position
      - Majority voting over multiple runs (majority_num)
    """

    def __init__(
            self,
            prompt_path: str | None = None,
            model_path: str = "GenPRM/GenPRM-1.5B",
            device: str = "auto",
            majority_num: int = 1,
            analyze: bool = True,
            verify: bool = True,
            execute: bool = True,
            time_limit: int = 3,
            max_tokens: int = 2048,
            temperature: float = 0.6,
            top_p: float = 0.95,
            top_k: int = 20,
            repetition_penalty: float = 1.0,
            logging_prompts: bool = False,
    ):
        super().__init__()
        self.model_path = model_path
        self.device = device
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # inference knobs
        self.majority_num = majority_num
        self.analyze = analyze
        self.verify = verify
        self.execute = execute
        self.time_limit = time_limit
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.logging_prompts = logging_prompts

        # prompt pieces
        self.analyze_template = "\nLet's analyze the Paragraph {cur_step} step by step: "
        self.verify_template = "\nLet's use python code to find any potential error:\n```python\n"
        self.output_template = "\n**Judgement**: \\boxed"

        # HF bits
        self.tokenizer = None
        self.model = None

        self.prompt = open(prompt_path, "r").read() if prompt_path else "{q}"
        self._yes_id = None
        self._no_id = None

        self._executor = _CodeExecutor()

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return ["prm_scores"], ["claims"]

    def init(self):
        if self.model is not None:
            return
        log.info(f"Initializing GenPRM (transformers) from {self.model_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        torch_dtype = torch.bfloat16 if ("cuda" in str(self.device) and torch.cuda.is_available()) else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(self.device).eval()
        # Match vLLM approach: use the LAST token id of the string
        self._yes_id = self.tokenizer.encode("Yes", add_special_tokens=False)[-1]
        self._no_id = self.tokenizer.encode("No", add_special_tokens=False)[-1]

    # ---------- Prompt building ----------

    @staticmethod
    def _format_solution_steps(steps: List[Claim]) -> str:
        return "\n".join([f"Paragraph {i + 1}: {c.claim_text.strip()}" for i, c in enumerate(steps)])

    def _messages_for_step(self, question: str, steps: List[Claim], step_index: int) -> List[Dict[str, str]]:
        sys_msg = {
            "role": "system",
            "content": "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step.",
        }
        lines = [f"Question: {question.strip()}", "", "Solution:"]
        for i in range(step_index + 1):
            lines.append(f"Paragraph {i + 1}: {steps[i].claim_text.strip()}")
        user_msg = {"role": "user", "content": "\n".join(lines)}
        return [sys_msg, user_msg]

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return _trim_eos(prompt, self.tokenizer)

    # ---------- HF generation helpers ----------

    def _generate_once(self, prompt: str, max_new_tokens: int, stop_at_first_newline: bool) -> Tuple[
        str, List[int], List[torch.Tensor]]:
        """
        Generate continuation; optionally truncate text at the first newline (for stage stopping).
        Returns (generated_text, gen_token_ids, scores_list) where scores_list[k] are logits
        for token k (0-based in the newly generated sequence).
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = input_ids.shape[1]

        out = self.model.generate(
            input_ids=input_ids,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        seq = out.sequences[0].tolist()
        gen_ids = seq[prompt_len:]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Optionally trim at the first newline for stage boundaries (we don't need to trim "scores" except at judgment stage)
        if stop_at_first_newline:
            cut = gen_text.find("\n")
            if cut != -1:
                gen_text = gen_text[:cut + 1]  # include newline
                # we won't use scores for analyze/verify scoring, so no need to trim scores/gen_ids

        # out.scores: List[Tensor] length == len(gen_ids), each [1, vocab]
        return gen_text, gen_ids, [s[0] for s in out.scores]

    # ---------- Scoring mechanics ----------

    def _get_reward_from_generated(self, gen_ids: List[int], scores: List[torch.Tensor]) -> float:
        """
        Find the last 'Yes' or 'No' token in the generated piece, take its logits from the
        corresponding scores entry, and return softmax(Yes, No)[Yes].
        """

        def last_index(seq, val):
            for i in range(len(seq) - 1, -1, -1):
                if seq[i] == val:
                    return i
            return -1

        yes_idx = last_index(gen_ids, self._yes_id)
        no_idx = last_index(gen_ids, self._no_id)

        # Prefer the token that actually appeared last (mirrors "use the generated token")
        idx = -1
        picked = None
        if yes_idx >= 0 and (no_idx < 0 or yes_idx > no_idx):
            idx = yes_idx
            picked = "Yes"
        elif no_idx >= 0:
            idx = no_idx
            picked = "No"

        if idx < 0 or idx >= len(scores):
            return 0.5

        logits = scores[idx]  # [vocab]
        logprobs = logits.log_softmax(dim=-1)
        yes_lp = float(logprobs[self._yes_id].item())
        no_lp = float(logprobs[self._no_id].item())
        yes_p = math.exp(yes_lp)
        no_p = math.exp(no_lp)
        denom = yes_p + no_p
        return (yes_p / denom) if denom > 0 else 0.5

    # ---------- One step end-to-end ----------

    def _single_inference_transformers(self, base_prompt: str, cur_step: int) -> Tuple[str, float]:
        """
        Stage 1 (analyze) -> Stage 2 (verify with optional exec loop) -> Stage 3 (output judgement)
        Returns (full_text_path, reward_prob_yes).
        """
        ctx = {"cur_step": cur_step}
        analyze_start = self.analyze_template.format(**ctx)
        verify_start = self.verify_template.format(**ctx)
        output_start = self.output_template.format(**ctx)

        # ---- Stage 1: analyze (optional) ----
        cur_prefix = ""
        if self.analyze:
            if self.logging_prompts:
                log.info(f"[GenPRM][step {cur_step}] ANALYZE prompt:\n{base_prompt + analyze_start}")
            analyze_text, _, _ = self._generate_once(
                prompt=base_prompt + analyze_start,
                max_new_tokens=min(128, self.max_tokens),
                stop_at_first_newline=True,
            )
            cur_prefix = analyze_start + analyze_text
        # ---- Stage 2: verify (optional, possibly iterative with code exec) ----
        if self.verify:
            verify_prefix = cur_prefix + verify_start
            iters = 0
            cur_text = ""
            while iters < self.time_limit:
                if self.logging_prompts:
                    log.info(
                        f"[GenPRM][step {cur_step}] VERIFY iter {iters + 1} prompt:\n{base_prompt + verify_prefix + cur_text}")
                gen_piece, _, _ = self._generate_once(
                    prompt=base_prompt + verify_prefix + cur_text,
                    max_new_tokens=min(256, self.max_tokens),
                    stop_at_first_newline=True,
                )
                cur_text += gen_piece
                if gen_piece.endswith("\n"):
                    break  # node finished
                # not finished — try to execute code and append output
                if self.execute:
                    code_out = self._executor.execute(verify_prefix + cur_text)
                    cur_text += f"[Code Output]\n\n```\n{code_out}\n```\n"
                else:
                    cur_text += "[Code Output]\n\n```\n"
                iters += 1
            cur_prefix = verify_prefix + cur_text

        # ---- Stage 3: output (judgement) ----
        # Force model right before the decision token so we can read logits on 'Yes'/'No'
        out_prompt = base_prompt + cur_prefix + output_start + "{"
        if self.logging_prompts:
            log.info(f"[GenPRM][step {cur_step}] OUTPUT prompt:\n{out_prompt}")

        gen_text, gen_ids, scores = self._generate_once(
            prompt=out_prompt,
            max_new_tokens=8,  # enough for "Yes}" or "No}" + newline
            stop_at_first_newline=True,
        )

        # Compose visible path text and compute reward
        full_text = cur_prefix + output_start + "{" + gen_text
        reward = self._get_reward_from_generated(gen_ids, scores)
        return full_text, reward

    # ---------- Public API ----------

    def get_rewards(self, question: str, steps: List[Claim]) -> List[float]:
        self.init()
        if not steps:
            return []

        scores: List[float] = []
        for i in range(len(steps)):
            messages = self._messages_for_step(question, steps, i)
            base_prompt = self._build_prompt(messages)

            # Majority voting (repeat runs and average)
            run_scores = []
            for _ in range(max(1, self.majority_num)):
                _, s = self._single_inference_transformers(base_prompt, cur_step=i + 1)
                run_scores.append(s)
            scores.append(sum(run_scores) / len(run_scores))
        return scores

    def __call__(
            self,
            dependencies: Dict[str, np.array],
            texts: List[str],
            model: Model,
            max_new_tokens: int = 100,
            **kwargs,
    ) -> Dict[str, np.ndarray]:
        self.init()
        rewards: list[list[float]] = []
        for input_text, claims in zip(texts, dependencies["claims"]):
            question = parse(self.prompt, input_text).named["q"]
            r = self.get_rewards(question, claims)
            assert len(r) == len(claims)
            rewards.append(r)
        return {"prm_scores": rewards}


class GenPRMStatCalculatorSimple(StatCalculator):
    """
    GenPRM (Generative PRM) step scorer that follows the official 'Judgement: Yes/No'
    decision format. For each step i, we construct a chat prompt that includes the
    question and the solution steps up to (and including) step i, instruct the model
    to judge ONLY the current step, and then read the next-token probabilities at
    the position immediately after 'Judgement: ' for 'Yes' vs 'No'.

    Default model: "GenPRM/GenPRM-1.5B"
    """

    def __init__(
            self,
            prompt_path: str | None = None,
            model_path: str = "GenPRM/GenPRM-1.5B",
            device: str = "auto",
    ):
        super().__init__()
        self.model_path = model_path
        self.device = device
        if device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.tokenizer = None
        self.model = None
        self.prompt = open(prompt_path, "r").read() if prompt_path else "{q}"

        # Target judgement tokens
        self.yes_token = "Yes"
        self.no_token = "No"
        self.yes_id = None
        self.no_id = None

        # You can flip this to True if you want to actually force the tags to appear
        # (e.g., generate "<analyze> ... </analyze>\n<verify> ... </verify>\n<output>\nJudgement: ").
        # For speed/consistency we keep it False and jump straight to the 'Judgement:' slot.
        self.generate_rationale = False

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return ["prm_scores"], ["claims"]

    def init(self):
        if self.model is not None:
            return
        log.info(f"Initializing GenPRM model {self.model_path} on device={self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

        # Prefer bf16 on CUDA, fall back to float32 on CPU for safety
        torch_dtype = torch.bfloat16 if ("cuda" in str(self.device) and torch.cuda.is_available()) else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(self.device).eval()

        # Resolve token ids for 'Yes'/'No' robustly
        def pick_single_id(text: str) -> int:
            # try exact
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
            # try with leading space (common BPE split)
            ids_space = self.tokenizer.encode(" " + text, add_special_tokens=False)
            if len(ids_space) == 1:
                return ids_space[0]
            # otherwise, fall back to the first sub-token
            return ids[0] if len(ids) > 0 else ids_space[0]

        self.yes_id = pick_single_id(self.yes_token)
        self.no_id = pick_single_id(self.no_token)

    # ---------- Prompt builders ----------

    @staticmethod
    def _format_solution_steps(steps: List[Claim]) -> str:
        # Present steps as numbered paragraphs to match GenPRM’s “review and critique paragraphs” phrasing.
        lines = []
        for i, c in enumerate(steps, start=1):
            s = c.claim_text.strip()
            # ensure each step is one clean line
            lines.append(f"Paragraph {i}: {s}")
        return "\n".join(lines)

    def _messages_for_step(self, question: str, steps: List[Claim], step_index: int) -> List[Dict[str, str]]:
        """Build a GenPRM-style chat for judging ONLY the current step."""
        assert 0 <= step_index < len(steps)
        shown_steps = steps[: step_index + 1]

        system_msg = (
            "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step. "
            "Judge ONLY the CURRENT paragraph. After your analysis and (optional) code verification, in the <output> "
            "section you must write exactly:\n\nJudgement: Yes\n\nor\n\nJudgement: No\n\n"
            "No extra text after that judgement."
        )

        # User content with problem and solution so far
        user_content = (
            f"Question: {question.strip()}\n\n"
            f"Solution:\n{self._format_solution_steps(shown_steps)}\n\n"
            f"Current paragraph index: {step_index + 1}\n"
            f"Only judge the CURRENT paragraph."
        )

        # Assistant prefill: either do the full tag skeleton, or jump straight to <output>/Judgement
        if self.generate_rationale:
            assistant_prefill = "<analyze>\n</analyze>\n<verify>\n</verify>\n<output>\nJudgement: "
        else:
            # Jump straight to the final judgement header so we can read next-token probs
            assistant_prefill = "<output>\nJudgement: "

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_prefill},
        ]

    # ---------- Scoring ----------

    def _score_current_step_yesprob(self, question: str, steps: List[Claim], step_index: int) -> float:
        """Return P(Yes) at the next-token position after 'Judgement: '."""
        self.init()
        messages = self._messages_for_step(question, steps, step_index)
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            logits = self.model(input_ids).logits  # [1, T, V]
        next_logits = logits[:, -1, :]  # next-token distribution at the 'Judgement: ' slot

        pair = torch.stack([next_logits[:, self.yes_id], next_logits[:, self.no_id]], dim=-1)  # [1, 2]
        probs = F.softmax(pair, dim=-1)[0].to("cpu")
        return float(probs[0].item())  # P(Yes)

    # ---------- Public API ----------

    def get_rewards(self, question: str, steps: List[Claim]) -> List[float]:
        self.init()
        if len(steps) == 0:
            return []
        scores: List[float] = []
        for i in range(len(steps)):
            p_yes = self._score_current_step_yesprob(question, steps, i)
            scores.append(p_yes)
        return scores

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
