from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from synthetic_dataset_generation.utils.steps_extractor import StepsExtractor
from argparse import Namespace
from parse import parse
import torch


def load_manager(man_path: str):
    file_path = hf_hub_download(
        repo_id=man_path,
        filename="ue_manager.pth",
        repo_type="model"
    )
    man = torch.load(file_path, weights_only=False)
    return man


def extract_steps(man, base_model_path: str, hf_cache: str = None) -> list[list[str]]:
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True,
                                                   cache_dir=hf_cache)

    steps_extractor = StepsExtractor()
    steps = steps_extractor(man['stats'], man['stats']['input_texts'],
                            model=Namespace(tokenizer=base_tokenizer))["claims"]
    return [[step.claim_text for step in sample_steps] for sample_steps in steps]


def extract_questions(man, prompt_file: str) -> list[str]:
    with open(prompt_file, 'r') as f:
        prompt = f.read()
    input_texts = man['stats']['input_texts']
    return [parse(prompt, inp_text).named['q'] for inp_text in input_texts]
