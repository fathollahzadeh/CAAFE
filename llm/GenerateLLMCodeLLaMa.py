import re
from groq import Groq
import time
import tiktoken


class GenerateLLMCodeLLaMa:
    @staticmethod
    def generate_code_LLaMa_LLM(messages: list):
        time_start = time.time()
        from util.Config import _LLM_API_Key
        _, api_key = _LLM_API_Key.get_API_Key()
        client = Groq(api_key=api_key)
        time_end = time.time()
        wait_time_1 = time_end - time_start
        prompt = ""
        for m in messages:
            prompt = f"{prompt}\n{m['content']}"

        time_start = time.time()
        code, gen_time = GenerateLLMCodeLLaMa.__submit_Request_LLaMa_LLM(messages=prompt, client=client)
        time_end = time.time()
        wait_time_2 = time_end - time_start - gen_time
        return code, GenerateLLMCodeLLaMa.get_number_tokens(messages=prompt), wait_time_1 + wait_time_2

    @staticmethod
    def __submit_Request_LLaMa_LLM(messages, client):
        from util.Config import _llm_model, _LLM_API_Key
        try:
            time_start = time.time()
            completion = client.chat.completions.create(
                model=_llm_model,
                messages=messages,
                temperature=0.5
            )
            content = completion.choices[0].message.content
            content = GenerateLLMCodeLLaMa.__refine_text(content)
            codes = []
            code_blocks = GenerateLLMCodeLLaMa.__match_code_blocks(content)
            if len(code_blocks) > 0:
                for code in code_blocks:
                    codes.append(code)

                return "\n".join(codes), time.time() - time_start
            else:
                return content, time.time() - time_start
        except Exception:
            _, api_key = _LLM_API_Key.get_API_Key()
            client = Groq(api_key=api_key)
            return GenerateLLMCodeLLaMa.__submit_Request_LLaMa_LLM(messages, client)

    @staticmethod
    def __match_code_blocks(text):
        pattern = re.compile(r'```(?:python)?[\n\r](.*?)```', re.DOTALL)
        return pattern.findall(text)

    @staticmethod
    def __refine_text(text):
        ind1 = text.find('\n')
        ind2 = text.rfind('\n')

        begin_txt = text[0: ind1]
        end_text = text[ind2+1:len(text)]
        begin_index = 0
        end_index = len(text)
        if begin_txt == "<CODE>":
            begin_index = ind1+1

        if end_text == "</CODE>":
            end_index = ind2
        text = text[begin_index:end_index]
        text = text.replace("<CODE>", "# <CODE>")
        text = text.replace("</CODE>", "# </CODE>")
        text = text.replace("```", "@ ```")

        from .GenerateLLMCode import GenerateLLMCode
        text = GenerateLLMCode.refine_source_code(code=text)
        return text

    @staticmethod
    def get_number_tokens(messages: str):
        enc = tiktoken.get_encoding("cl100k_base")
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        token_integers = enc.encode(messages)
        num_tokens = len(token_integers)
        return num_tokens