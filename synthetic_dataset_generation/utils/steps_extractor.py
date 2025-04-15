from lm_polygraph.stat_calculators.extract_claims import *


class StepsExtractor:
    def __init__(
            self,
            sent_separators: str = "\n",
            progress_bar: bool = True,
    ):
        self.sent_separators = sent_separators
        self.progress_bar = progress_bar

    def __call__(
            self,
            dependencies: Dict[str, object],
            texts: List[str],
            model: WhiteboxModel,
            *args,
            **kwargs,
    ) -> Dict[str, List]:
        claims: list[list[Claim]] = []
        claim_texts_concatenated: list[str] = []
        claim_input_texts_concatenated: list[str] = []

        data = zip(
            texts,
            dependencies["greedy_texts"],
            dependencies["greedy_tokens"],
        )
        if self.progress_bar:
            data = tqdm(data, total=len(texts), desc='Extracting steps')
        for input_text, greedy_text, greedy_tokens in data:
            steps: list[Claim] = self.split_to_steps(greedy_text, greedy_tokens, model.tokenizer)
            claims.append(steps)
            claim_texts_concatenated += [c.claim_text for c in steps]
            claim_input_texts_concatenated += [input_text for c in steps]

        return {
            "claims": claims,
            "claim_texts_concatenated": claim_texts_concatenated,
            "claim_input_texts_concatenated": claim_input_texts_concatenated,
        }

    def split_to_steps(
            self,
            text: str,
            tokens: list[int],
            tokenizer,
    ) -> list[Claim]:
        if tokenizer.decode(tokens) != text:
            return []
        prev_token_i, token_i = 0, 0
        prev_text_i = 0
        claims: list[Claim] = []
        for text_i in range(len(text)):
            if text[text_i] in self.sent_separators:
                claims.append(Claim(
                    claim_text=text[prev_text_i:text_i + 1],
                    sentence=text[prev_text_i:text_i + 1],
                    aligned_token_ids=list(range(prev_token_i, token_i + 1))
                ))
            while token_i < len(tokens) and tokenizer.decode(tokens[:token_i + 1]) in text[:text_i + 1]:
                token_i += 1
            if text[text_i] in self.sent_separators:
                prev_text_i = text_i + 1
                prev_token_i = token_i
        return claims
