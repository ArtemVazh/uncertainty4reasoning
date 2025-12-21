from lm_polygraph.stat_calculators.extract_claims import *

REPLACEMENT = "\uFFFD"

class StepsExtractorThinking(StatCalculator):
    """
    Extract steps from free-form math reasoning:
    - Split textual reasoning by sentences (., !, ?).
    - After extraction, if a step has < min_chars_per_step characters, append it to the previous step.
    """
    def __init__(
        self,
        thinking_prefix: str = "<think>",
        thinking_suffix: str = "</think>",
        skip_starts: list[str] = (),
        progress_bar: bool = True,
        min_chars_per_step: int = 20,
    ):
        super().__init__()
        self.thinking_prefix = thinking_prefix
        self.thinking_suffix = thinking_suffix
        self.skip_starts = list(skip_starts)
        self.progress_bar = progress_bar
        self.min_chars_per_step = min_chars_per_step

        # tiny list of abbreviations to avoid false sentence splits on '.'
        self._abbr = { "e.g", "i.e", "mr", "mrs", "ms", "dr", "prof", "vs", "no", "etc" }

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
            claim_input_texts_concatenated += [input_text for _ in steps]

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

    # ---------- main splitter ----------

    def split_to_steps(
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

        text_i = 0
        while text_i < len(text):
            # if replace char detected, token_i must move forward.
            while tokenizer.decode(tokens[:token_i + 1])[-1] == REPLACEMENT:
                token_i += 1
            # advance token_i to cover text[:i+1]
            while token_i < len(tokens) and tokenizer.decode(tokens[:token_i + 1]) in text[:text_i + 1] and text[:text_i + 1] != tokenizer.decode(tokens[:token_i + 1]):
                token_i += 1

            boundary = False
            end_i = None

            # 1) sentence punctuation boundary candidates
            if self._is_sentence_boundary(text, text_i):
                boundary = True
                end_i = text_i + 1

            # finalize a segment if boundary was confirmed
            if boundary and end_i is not None:
                segment = text[prev_text_i:end_i]
                if self.filter_claim_texts(segment):
                    claims.append(Claim(
                        claim_text=segment.strip(),
                        sentence=segment,
                        aligned_token_ids=list(range(prev_token_i, token_i + 1))
                    ))
                prev_text_i = end_i
                prev_token_i = token_i + 1
            text_i += 1

        # remove tail incomplete sentences
        # tail = text[prev_text_i:]
        # if self.filter_claim_texts(tail):
        #     claims.append(Claim(
        #         claim_text=tail.strip(),
        #         sentence=tail,
        #         aligned_token_ids=list(range(prev_token_i, token_i + 1))
        #     ))

        # ---------- POST-PROCESS: merge short claims (< min_chars_per_step characters) into the previous ----------
        merged: list[Claim] = []
        for c in claims:
            if self.min_chars_per_step > 0 and len(c.claim_text) < self.min_chars_per_step and merged:
                prev = merged[-1]
                # choose a clean separator between texts
                sep_text = "" if (prev.claim_text.endswith(("\n", " ")) or c.claim_text.startswith((" ", "\n"))) else " "
                sep_sent = "" if (prev.sentence.endswith(("\n", " ")) or c.sentence.startswith((" ", "\n"))) else " "
                merged[-1] = Claim(
                    claim_text=(prev.claim_text.rstrip() + sep_text + c.claim_text.lstrip()).strip(),
                    sentence=prev.sentence + sep_sent + c.sentence,
                    aligned_token_ids=prev.aligned_token_ids + c.aligned_token_ids,
                )
            elif self.thinking_prefix in c.claim_text:
                prefix_token_id = tokenizer.encode(self.thinking_prefix, add_special_tokens=False)[0]
                new_aligned_token_ids = c.aligned_token_ids
                for i, token_id in enumerate(c.aligned_token_ids):
                    if tokens[token_id] == prefix_token_id:
                        new_aligned_token_ids = c.aligned_token_ids[i + 1:]
                        break
                new_claim = Claim(
                    claim_text=c.claim_text.split(self.thinking_prefix, 1)[1],
                    sentence=c.sentence.split(self.thinking_prefix, 1)[1],
                    aligned_token_ids=new_aligned_token_ids,
                )
                merged.append(new_claim)
            elif self.thinking_suffix in c.claim_text:
                break
                # suffix_token_id = tokenizer.encode(self.thinking_suffix, add_special_tokens=False)[0]
                # new_aligned_token_ids = c.aligned_token_ids
                # for i, token_id in enumerate(c.aligned_token_ids):
                #     if token_id == suffix_token_id:
                #         new_aligned_token_ids = c.aligned_token_ids[:i]
                #         break
                # new_claim = Claim(
                #     claim_text=c.claim_text.split(self.thinking_suffix, 1)[0],
                #     sentence=c.sentence.split(self.thinking_suffix, 1)[0],
                #     aligned_token_ids=new_aligned_token_ids,
                # )
                # merged.append(new_claim)
            else:
                merged.append(c)

        return merged


def load_stat_calculator(config, builder):
    return StepsExtractorThinking(
        thinking_prefix=getattr(config, "thinking_prefix", "<think>"),
        thinking_suffix=getattr(config, "thinking_suffix", "</think>"),
        progress_bar=getattr(config, "progress_bar", False),
        min_chars_per_step=getattr(config, "min_chars_per_step", 20),
    )
