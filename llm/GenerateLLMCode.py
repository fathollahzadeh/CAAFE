from .GenerateLLMCodeGPT import GenerateLLMCodeGPT
from .GenerateLLMCodeLLaMa import GenerateLLMCodeLLaMa
from .GenerateLLMGemini import GenerateLLMGemini


class GenerateLLMCode:

    @staticmethod
    def generate_llm_code(messages: list):
        from util.Config import _llm_platform, _OPENAI, _META, _GOOGLE

        if _llm_platform is None:
            raise Exception("Select a LLM Platform: OpenAI (GPT) or Meta (Lama)")
        elif _llm_platform == _OPENAI:
            return GenerateLLMCodeGPT.generate_code_OpenAI_LLM(messages=messages)
        elif _llm_platform == _META:
            return GenerateLLMCodeLLaMa.generate_code_LLaMa_LLM(messages=messages)
        elif _llm_platform == _GOOGLE:
            return GenerateLLMGemini.generate_code_Gemini_LLM(messages=messages)

        else:
            raise Exception(f"Model {_llm_platform} is not implemented yet!")

    @staticmethod
    def refine_source_code(code: str):
        final_code = []
        for line in code.splitlines():
            if not line.startswith('#'):
                final_code.append(line)
        final_code = "\n".join(final_code)
        return final_code.replace("@ ```", "# ```")

    def get_number_tokens(messages: list):
        from util.Config import _llm_platform, _OPENAI, _META, _GOOGLE

        if _llm_platform is None:
            raise Exception("Select a LLM Platform: OpenAI (GPT) or Meta (Lama)")
        elif _llm_platform == _OPENAI:
            return GenerateLLMCodeGPT.get_number_tokens(messages=messages)
        elif _llm_platform == _META:
            return 0
        elif _llm_platform == _GOOGLE:
            prompt = ""
            for m in messages:
                d = m.keys()
                prompt = f"{prompt}\n{m[d]}"
            return GenerateLLMGemini.get_number_tokens(messages=prompt)

        else:
            raise Exception(f"Model {_llm_platform} is not implemented yet!")
