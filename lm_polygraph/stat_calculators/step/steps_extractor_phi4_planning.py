from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any

from tqdm import tqdm
from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.stat_calculators.extract_claims import Claim, WhiteboxModel


class StepsExtractor(StatCalculator):
    STEP_RE = re.compile(r'(^|\n)(-\s*Step\s+\d+\s*:\s*)')
    ANSWER_RE = re.compile(r'(^|\n)(<\s*Answer\s*>\s*:\s*)', re.IGNORECASE)

    def __init__(
        self,
        skip_starts: List[str] | None = None,
        progress_bar: bool = True,
    ):
        super().__init__()
        self.skip_starts = skip_starts or ['Reasoning Steps:']
        self.progress_bar = progress_bar

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return (
            [
                "claims",
                "claim_texts_concatenated",
                "claim_input_texts_concatenated",
            ],
            [
                "greedy_texts",
                "greedy_tokens",
            ],
        )

    def __call__(
        self,
        dependencies: Dict[str, object],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        *args,
        **kwargs,
    ) -> Dict[str, List]:
        claims: List[List[Claim]] = []
        claim_texts_concatenated: List[str] = []
        claim_input_texts_concatenated: List[str] = []

        data = zip(
            texts,
            dependencies["greedy_texts"],
            dependencies["greedy_tokens"],
        )
        if self.progress_bar:
            data = tqdm(data, total=len(texts), desc='Extracting steps')

        for input_text, greedy_text, greedy_tokens in data:
            steps: List[Claim] = self.split_to_steps(
                greedy_text, greedy_tokens, model.tokenizer
            )
            claims.append(steps)
            claim_texts_concatenated += [c.claim_text for c in steps]
            claim_input_texts_concatenated += [input_text for _ in steps]

        return {
            "claims": claims,
            "claim_texts_concatenated": claim_texts_concatenated,
            "claim_input_texts_concatenated": claim_input_texts_concatenated,
        }

    def filter_claim_texts(self, claim_text: str) -> bool:
        claim_text = claim_text.strip()
        return len(claim_text) > 0 and not any(
            claim_text.lower().startswith(b.lower()) for b in self.skip_starts
        )

    def _find_spans(self, text: str) -> List[Tuple[int, int]]:
        markers: List[int] = []

        for m in self.STEP_RE.finditer(text):
            markers.append(m.start(2))
        for m in self.ANSWER_RE.finditer(text):
            markers.append(m.start(2))

        if not markers:
            return []

        markers.sort()
        spans: List[Tuple[int, int]] = []
        for i, start in enumerate(markers):
            end = markers[i + 1] if i + 1 < len(markers) else len(text)
            spans.append((start, end))
        return spans

    def _char_to_token_index_boundaries(
        self, text: str, tokens: List[int], tokenizer, boundaries: List[int]
    ) -> List[int]:
        results: List[int] = []
        token_i = 0
        for boundary in sorted(boundaries):
            while token_i < len(tokens):
                next_decoded = tokenizer.decode(tokens[: token_i + 1])
                if next_decoded == text[: len(next_decoded)] and len(next_decoded) <= boundary:
                    token_i += 1
                else:
                    break
            results.append(token_i)
        return results

    def split_to_steps(
        self,
        text: str,
        tokens: List[int],
        tokenizer,
    ) -> List[Claim]:
        if not tokenizer.decode(tokens).startswith(text):
            return []

        spans = self._find_spans(text)
        if not spans:
            return []

        char_boundaries: List[int] = []
        for start, end in spans:
            char_boundaries.extend([start, end])

        token_boundaries = self._char_to_token_index_boundaries(
            text, tokens, tokenizer, char_boundaries
        )

        claims: List[Claim] = []
        for i, (start, end) in enumerate(spans):
            seg = text[start:end]
            if not self.filter_claim_texts(seg):
                continue
            tok_start = token_boundaries[2 * i]
            tok_end = token_boundaries[2 * i + 1]
            aligned_ids = list(range(tok_start, min(tok_end, max(len(tokens) - 1, 0))))

            claims.append(
                Claim(
                    claim_text=seg.strip(),
                    sentence=seg,
                    aligned_token_ids=aligned_ids,
                )
            )

        return claims


def load_stat_calculator(config, builder):
    return StepsExtractor(
        progress_bar=getattr(config, "progress_bar", False),
    )
