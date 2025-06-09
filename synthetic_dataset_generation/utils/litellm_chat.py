import os
import diskcache as dc
import time
import sys
from litellm import completion


MODEL_DICT = {
    'o1': 'o1',
    'o3-mini': 'o3-mini',
    'gpt-4o-mini': 'gpt-4o-mini',
    'gpt-4o': 'gpt-4o',
    'deepseek-chat': 'together_ai/deepseek-ai/DeepSeek-V3',
    'deepseek-reasoner': "together_ai/deepseek-ai/DeepSeek-R1",
    'qwq-32b': 'together_ai/Qwen/QwQ-32B',
    'sonnet-3.7-high': "anthropic/claude-3-7-sonnet-20250219",
    'gemini-2.5-pro': "gemini/gemini-2.5-pro-preview-03-25",
    'llama_405': "together_ai/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    'llama_70': "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    'llama4_maverick': "together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    'gemma3': "together_ai/google/gemma-3-12b-it",
}
INPUT_COST_DICT = {
    'o1': 15,
    'o3-mini': 1.1,
    'gpt-4o-mini': 0.15,
    'gpt-4o': 2.5,
    # 'deepseek-chat': 0.27,
    # 'deepseek-reasoner': 0.55,
    'deepseek-chat': 1.25,
    'deepseek-reasoner': 3,
    'qwq-32b': 1.2,
    'sonnet-3.7-high': 3,
    'gemini-2.5-pro': 1.25,
    'llama_405': 3.5,
    'llama_70': 0.88,
    'llama4_maverick': 0.27,
    'gemma3': 0.3,
}
OUTPUT_COST_DICT = {
    'o1': 60,
    'o3-mini': 4.4,
    'gpt-4o-mini': 0.6,
    'gpt-4o': 10,
    # 'deepseek-chat': 1.10,
    # 'deepseek-reasoner': 2.19,
    'deepseek-chat': 1.25,
    'deepseek-reasoner': 7,
    'qwq-32b': 1.2,
    'sonnet-3.7-high': 15,
    'gemini-2.5-pro': 2.5,
    'llama_405': 3.5,
    'llama_70': 0.88,
    'llama4_maverick': 0.85,
    'gemma3': 0.3,
}
GENE_ARGS_DICT = {
    'gpt-4o-mini': {'temperature': 0, 'max_tokens': 4096},
    'deepseek-reasoner': {'temperature': 0.6, 'max_tokens': 8192},
    'qwq-32b': {'temperature': 0.6, 'top_p': 0.95, 'max_tokens': 8192},
    'o3-mini': {'reasoning_effort': 'high', 'max_tokens': 8192},
    'sonnet-3.7-high': {'reasoning_effort': 'high', 'max_tokens': 8192},
    'gemini-2.5-pro': {'max_tokens': 8192},
    'deepseek-chat': {'temperature': 0, 'max_tokens': 4096},
    'llama_405': {'temperature': 0, 'max_tokens': 4096},
    'llama_70': {'temperature': 0, 'max_tokens': 4096},
    'llama4_maverick': {'temperature': 0, 'max_tokens': 4096},
    'gemma3': {'temperature': 0, 'max_tokens': 4096},
}


class LiteLLMChat:
    def __init__(
            self,
            model_name: str = "together_ai/deepseek-ai/DeepSeek-R1",
            cache_path: str = "litellm_cache",
            cache_name: str = "litellm",
            generation_args: dict = None,
    ):
        self.model_name = MODEL_DICT[model_name]
        self.cache_path = os.path.join(cache_path, f"{cache_name}.diskcache")
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        if generation_args:
            self.generation_args = generation_args
        else:
            self.generation_args = GENE_ARGS_DICT[model_name]

    def ask(self, message: str):
        cache_settings = dc.DEFAULT_SETTINGS.copy()
        cache_settings["eviction_policy"] = "none"
        cache_settings["size_limit"] = int(1e12)
        cache_settings["cull_limit"] = 0
        with dc.Cache(self.cache_path, **cache_settings) as litellm_responses:
            if (self.model_name, message) in litellm_responses:
                reply_content = litellm_responses[(self.model_name, message)]
                print("Loaded from cache")
                input_price, output_price, input_token_num, output_token_num = 0, 0, 0, 0
            else:
                messages = [{"role": "user", "content": message}]
                chat = self._send_request(messages)
                reply_content = {
                    'response': chat.choices[0].message.content,
                    'response_reasoning': chat.choices[0].message.reasoning_content,
                }
                litellm_responses[(self.model_name, message)] = reply_content

        return reply_content

    def _send_request(self, messages):
        sleep_time_values = (5, 10, 30, 60, 120)
        arg_dict = {
            'model': self.model_name,
            'messages': messages,
            **self.generation_args,
        }
        for i in range(len(sleep_time_values)):
            try:
                return completion(**arg_dict)
            except Exception as e:
                sleep_time = sleep_time_values[i]
                print(
                    f"Request to LiteLLM failed with exception: {e}. Retry #{i}/5 after {sleep_time} seconds."
                )
                time.sleep(sleep_time)
        try:
            return completion(**arg_dict)
        except Exception as e:
            sys.stderr.write(f'Error: {e}')
            return None
