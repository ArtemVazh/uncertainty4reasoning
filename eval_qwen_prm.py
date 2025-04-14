import torch
from transformers import AutoModel, AutoTokenizer

if __name__ == '__main__':
    hf_cache = '/cluster/project/sachan/ekaterina/.cache'
    model_name = "Qwen/Qwen2.5-Math-7B-PRM800K"
    device = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=hf_cache)
    model = AutoModel.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=hf_cache,
    ).eval()