from lm_polygraph.generation_metrics.openai_fact_check import *
from lm_polygraph.stat_calculators.extract_claims import *
from synthetic_dataset_generation.utils.deepseek_chat import DeepSeekChat


class StepFactCheck(GenerationMetric):
    def __init__(
            self,
            cache_path: str = "~/.cache",
            model: str = 'deepseek-reasoner',
            api_key: str | None = None,
            progress_bar: bool = True,
            n_threads: int = 1,
            wait_times: tuple = (5, 10, 30, 60, 120),
    ):
        super().__init__(["input_texts", "claims"], "claim")
        self.chat = DeepSeekChat(cache_path, model=model, api_key=api_key, wait_times=wait_times)

        # use this for OpenAI
        # self.chat = DeepSeekChat(api_base=None, model='gpt-4o', cache_path=cache_path, api_key=api_key, wait_times=wait_times)

        self.progress_bar = progress_bar
        self.n_threads = n_threads

    def __str__(self):
        return "StepFactCheck"

    def _clean_step(self, st: str):
        return st.strip()

    def prompt1(self, input_text: str, claims: list[Claim], answer: str) -> str:
        problem = input_text.split('<|im_start|>user')[-1].split('<|im_end|>')[0]
        steps = '\n'.join([f'Step {str(i + 1)}. {self._clean_step(cl.claim_text)}' for i, cl in enumerate(claims)])
        return r'''You are given a problem, a ground-truth solution, and a step-by-step student solution. Your task is to analyze each step in the student’s solution to determine whether it is both logically correct and relevant.

Instructions:
- Carefully examine each student step for logical errors or unnecessary/redundant reasoning.
- If all steps are correct and they lead to the same final answer as the ground-truth solution, conclude that there are no errors.
- If any step contains an error that would prevent the student from reaching the correct solution, identify and report those specific steps with an explanation.

Problem:
{problem}

Ground-truth solution:
{answer}

Student's step-by-step solution:
{steps}

Now, please evaluate whether the student’s steps are correct and logical.'''.format(problem=problem, answer=answer,
                                                                                    steps=steps)

    def prompt2(self, input_text: str, claims: list[Claim], answer: str, reply: str) -> str:
        problem = input_text.split('<|im_start|>user')[-1].split('<|im_end|>')[0]
        steps = '\n'.join([f'Step {str(i + 1)}. {cl.claim_text.strip()}' for i, cl in enumerate(claims)])
        return r'''You are given a problem, a step-by-step student solution, and an assessment text indicating which steps are correct or incorrect. 
Your task is to output a single line listing the indices (step numbers) of all the steps that are assessed as incorrect.

Problem:
{problem}

Student's step-by-step solution:
{steps}

Step-by-step assessment:
{reply}

Now, please output only the indices of all incorrect steps found, separated by commas. If all steps are correct, output "All steps are correct."'''.format(
            problem=problem, steps=steps, reply=reply)

    def parse_reply(self, reply: str) -> list[int] | None:
        if 'all steps are correct' in reply.lower():
            return []
        orig_reply = reply
        reply = reply.strip().replace(' ', '').replace('Step', '')
        if reply.startswith('[') and reply.endswith(']'):
            reply = reply[1:-1]
        try:
            return [int(x) - 1 for x in reply.split(',')]
        except Exception as e:
            log.warning('Skipping text, because could not parse DeepSeek reply: {}'.format(orig_reply))
            return None

    def _score_single(self, args: tuple[list, str, str]) -> list:
        claims, input_text, answer = args
        q1 = self.prompt1(input_text, claims, answer)
        reply = self.chat.ask(q1)
        q2 = self.prompt2(input_text, claims, answer, reply)
        reply = self.chat.ask(q2)
        wrong_claim_ids: list[int] | None = self.parse_reply(reply)
        if wrong_claim_ids is None:
            return [np.nan for _ in range(len(claims))]  # will be skipped at evaluation
        return [(1 if i in wrong_claim_ids else 0) for i in range(len(claims))]

    def __call__(
            self,
            stats: Dict[str, np.ndarray],
            target_texts: List[str],
    ) -> list:
        input_texts = stats["input_texts"]

        if "answers" in stats.keys():
            target_texts = stats["answers"]

        all_inputs = [
            (claims, input_text, answer)
            for input_text, claims, answer in zip(input_texts, stats["claims"], target_texts)
        ]

        with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
            claim_labels: list[list] = list(
                tqdm(
                    executor.map(self._score_single, all_inputs),
                    total=len(all_inputs),
                    desc="Verifying claims",
                    disable=not self.progress_bar,
                )
            )
        return claim_labels
