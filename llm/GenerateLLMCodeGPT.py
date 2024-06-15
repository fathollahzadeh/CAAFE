import openai
import tiktoken
import os
import time


class GenerateLLMCodeGPT:
    @staticmethod
    def generate_code_OpenAI_LLM(messages: list):
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        number_of_tokens = GenerateLLMCodeGPT.get_number_tokens(messages=messages)
        time_start = time.time()
        code = GenerateLLMCodeGPT.__submit_Request_OpenAI_LLM( messages=messages)
        time_end = time.time()
        return code, number_of_tokens, time_end - time_start

    @staticmethod
    def __submit_Request_OpenAI_LLM(messages):
        from util.Config import _llm_model, _max_token_limit
        completion = openai.ChatCompletion.create(
            model=_llm_model,
            messages=messages,
            stop=["```end"],
            temperature=0.5,
            max_tokens=_max_token_limit,
        )
        code = completion["choices"][0]["message"]["content"]
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code


    @staticmethod
    def get_number_tokens(messages: list):
        prompt = ""
        for m in messages:
            d = m.keys()
            prompt = f"{prompt}\n{m[d]}"

        from util.Config import _llm_model
        enc = tiktoken.get_encoding("cl100k_base")
        enc = tiktoken.encoding_for_model(_llm_model)
        token_integers = enc.encode(prompt)
        num_tokens = len(token_integers)
        return num_tokens