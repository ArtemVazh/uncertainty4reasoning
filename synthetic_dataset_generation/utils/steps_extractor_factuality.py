from lm_polygraph.stat_calculators.extract_claims import *


class StepsExtractorFactuality(StatCalculator):
    """
    Split a model's response into factual claims by treating each sentence
    (terminated by ., !, or ?) as an individual claim.
    After extraction, if a claim has < 20 characters, append it to the previous claim.
    """

    def __init__(
        self,
        skip_starts: list[str] = (),
        progress_bar: bool = True,
    ):
        super().__init__()
        self.skip_starts = list(skip_starts)
        self.progress_bar = progress_bar

        # tiny list of abbreviations to avoid false sentence splits on '.'
        self._abbr = {"e.g", "i.e", "mr", "mrs", "ms", "dr", "prof", "vs", "no", "etc"}

    @staticmethod
    def meta_info() -> tuple[list[str], list[str]]:
        return (
            ["claims", "claim_texts_concatenated", "claim_input_texts_concatenated"],
            ["greedy_texts", "greedy_tokens"],
        )

    def __call__(
        self,
        dependencies: dict[str, object],
        texts: list[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
        *args,
        **kwargs,
    ) -> dict[str, list]:
        claims: list[list[Claim]] = []
        claim_texts_concatenated: list[str] = []
        claim_input_texts_concatenated: list[str] = []

        data = zip(texts, dependencies["greedy_texts"], dependencies["greedy_tokens"])
        if self.progress_bar:
            data = tqdm(data, total=len(texts), desc="Extracting factual claims")
        for input_text, greedy_text, greedy_tokens in data:
            sentence_claims = self.split_to_claims(
                greedy_text, greedy_tokens, model.tokenizer
            )
            claims.append(sentence_claims)
            claim_texts_concatenated += [c.claim_text for c in sentence_claims]
            claim_input_texts_concatenated += [input_text for _ in sentence_claims]

        return {
            "claims": claims,
            "claim_texts_concatenated": claim_texts_concatenated,
            "claim_input_texts_concatenated": claim_input_texts_concatenated,
        }

    def filter_claim_texts(self, claim_text: str) -> bool:
        claim_text = claim_text.strip()
        return len(claim_text) > 0 and not any(claim_text.lower().startswith(b.lower()) for b in self.skip_starts)

    # ---------- helpers for sentence detection ----------

    def _is_sentence_boundary(self, text: str, i: int) -> bool:
        """Return True if char at i ends a sentence (., !, ?), with simple guards."""
        ch = text[i]
        if ch not in ".!?":
            return False
        n = len(text)

        # decimal numbers like 3.14
        if ch == "." and i > 0 and i + 1 < n and text[i-1].isdigit() and text[i+1].isdigit():
            return False

        # common abbreviations like "e.g." / "Dr." (case-insensitive)
        # look back to previous word before the dot
        j = i - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        end = j
        while j >= 0 and (text[j].isalpha() or text[j] in "’'"):
            j -= 1
        word = text[j+1:end+1].strip("’'").lower()
        if word in self._abbr:
            return False

        # typical sentence end: allow quotes/brackets/space after punctuation
        k = i + 1
        while k < n and text[k] in ' \t\n"“”\'’)]}':
            k += 1
        # it's fine to be generous: if something follows, we still consider this a boundary
        return True

    def _skip_ws(self, text: str, i: int) -> int:
        """Skip spaces/tabs/newlines to next non-whitespace char index (or len)."""
        n = len(text)
        while i < n and text[i].isspace():
            i += 1
        return i

    # ---------- main splitter ----------

    def split_to_claims(
        self,
        text: str,
        tokens: list[int],
        tokenizer,
    ) -> list[Claim]:
        # ensure alignment: greedy_text must be prefix of decoded tokens
        if not tokenizer.decode(tokens).startswith(text):
            return []

        claims: list[Claim] = []
        prev_text_i = 0
        prev_token_i = 0
        token_i = 0
        n = len(text)

        i = 0
        while i < n:
            ch = text[i]

            # advance token_i to cover text[:i+1]
            while (
                token_i < len(tokens)
                and tokenizer.decode(tokens[: token_i + 1]) in text[: i + 1]
            ):
                token_i += 1

            # sentence punctuation boundary candidates
            if ch in ".!?":
                if self._is_sentence_boundary(text, i):
                    end_i = i + 1
                    segment = text[prev_text_i:end_i]
                    if self.filter_claim_texts(segment):
                        claims.append(
                            Claim(
                                claim_text=segment.strip(),
                                sentence=segment,
                                aligned_token_ids=list(
                                    range(prev_token_i, min(token_i, len(tokens)))
                                ),
                            )
                        )
                    prev_text_i = end_i
                    prev_token_i = token_i

            i += 1

        # tail
        tail = text[prev_text_i:]
        if self.filter_claim_texts(tail):
            claims.append(
                Claim(
                    claim_text=tail.strip(),
                    sentence=tail,
                    aligned_token_ids=list(
                        range(prev_token_i, min(token_i, len(tokens)))
                    ),
                )
            )

        # ---------- POST-PROCESS: merge short claims (< 20 characters) into the previous ----------
        merged: list[Claim] = []
        for c in claims:
            if len(c.claim_text.strip()) < 20 and merged:
                prev = merged[-1]
                # choose a clean separator between texts
                sep_text = "" if (prev.claim_text.endswith(("\n", " ")) or c.claim_text.startswith((" ", "\n"))) else " "
                # sep_sent = "" if (prev.sentence.endswith(("\n", " ")) or c.sentence.startswith((" ", "\n"))) else " "
                merged[-1] = Claim(
                    claim_text=(prev.claim_text.rstrip() + sep_text + c.claim_text.lstrip()).strip(),
                    sentence=prev.sentence + c.sentence,
                    aligned_token_ids=prev.aligned_token_ids + c.aligned_token_ids,
                )
            else:
                merged.append(c)

        return merged


def load_stat_calculator(config, builder):
    return StepsExtractorFactuality(
        skip_starts=getattr(config, "skip_starts", ()),
        progress_bar=getattr(config, "progress_bar", False),
    )
