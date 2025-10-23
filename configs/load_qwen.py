from transformers import AutoTokenizer, AutoModelForCausalLM


def load_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.chat_template = None
    return tokenizer


def load_model(model_path: str, device_map: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    return model