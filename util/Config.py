from .LLM_API_Key import LLM_API_Key

__gen_run_mode = 'generate-and-run'
__validation_run_mode = 'validation'

__sub_task_data_preprocessing = "DataPreprocessing"
__sub_task_feature_engineering = "FeatureEngineering"
__sub_task_model_selection = "ModelSelection"

__GPT_4_Limit = 8192
__GPT_4_1106_Preview_Limit = 4096
__GPT_4_Turbo_Limit = 4096
__GPT_4o_Limit = 4096
__GPT_3_5_Turbo_limit = 4096
__Llama2_70b = 4096
__Llama3_70b_8192 = 8192
__Llama3_8b_8192 = 8192
__Mixtral_8x7b_32768 = 32768
__Gemma_7b_it = 8192
__Gemini = 8192

_OPENAI = "OpenAI"
__GPT_system_delimiter = "### "
__GPT_user_delimiter = "### "

_META = "Meta"
__Llama_system_delimiter = "### "
__Llama_user_delimiter = "### "

_GOOGLE = "Google"
__Gemini_system_delimiter = "### "
__Gemini_user_delimiter = "### "

_llm_model = None
_llm_platform = None
_system_delimiter = None
_user_delimiter = None
_max_token_limit = None
_delay = 75
_last_API_Key = None
_LLM_API_Key = None
_system_log_file = None

_df_train = None
_trainy = None
_df_test = None
_testy = None
_target_attribute = None


def set_dataset(df_train, trainy, df_test, testy, traget_attribute):
    global _df_train
    global _trainy
    global _df_test
    global _testy
    global _target_attribute

    _trainX = df_train
    _trainy = trainy
    _testX = df_test
    _testy = testy
    _target_attribute = traget_attribute


def set_config(model, delay, system_log):
    global _llm_model
    global _llm_platform
    global _system_delimiter
    global _user_delimiter
    global _max_token_limit
    global _delay
    global _last_API_Key
    global _LLM_API_Key
    global _system_log_file

    _llm_model = model
    _delay = delay
    _system_log_file = system_log

    if model == "gpt-4":
        _llm_platform = _OPENAI
        _max_token_limit = __GPT_4_Limit
        _user_delimiter = __GPT_user_delimiter
        _system_delimiter = __GPT_system_delimiter

    elif model == "gpt-4-1106-preview_":
        _llm_platform = _OPENAI
        _max_token_limit = __GPT_4_1106_Preview_Limit
        _user_delimiter = __GPT_user_delimiter
        _system_delimiter = __GPT_system_delimiter

    elif model == "gpt-4-turbo":
        _llm_platform = _OPENAI
        _max_token_limit = __GPT_4_Turbo_Limit
        _user_delimiter = __GPT_user_delimiter
        _system_delimiter = __GPT_system_delimiter

    elif model == "gpt-4o":
        _llm_platform = _OPENAI
        _max_token_limit = __GPT_4o_Limit
        _user_delimiter = __GPT_user_delimiter
        _system_delimiter = __GPT_system_delimiter

    elif model == "gpt-3.5-turbo":
        _llm_platform = _OPENAI
        _max_token_limit = __GPT_3_5_Turbo_limit
        _user_delimiter = __GPT_user_delimiter
        _system_delimiter = __GPT_system_delimiter

    elif model == "llama2-70b":
        _llm_platform = _META
        _max_token_limit = __Llama2_70b
        _user_delimiter = __Llama_user_delimiter
        _system_delimiter = __Llama_system_delimiter

    elif model == "llama3-70b-8192":
        _llm_platform = _META
        _max_token_limit = __Llama3_70b_8192
        _user_delimiter = __Llama_user_delimiter
        _system_delimiter = __Llama_system_delimiter

    elif model == "llama3-8b-8192":
        _llm_platform = _META
        _max_token_limit = __Llama3_8b_8192
        _user_delimiter = __Llama_user_delimiter
        _system_delimiter = __Llama_system_delimiter

    elif model == "mixtral-8x7b-32768":
        _llm_platform = _META
        _max_token_limit = __Mixtral_8x7b_32768
        _user_delimiter = __Llama_user_delimiter
        _system_delimiter = __Llama_system_delimiter

    elif model == "gemma-7b-it":
        _llm_platform = _GOOGLE
        _max_token_limit = __Gemma_7b_it
        _user_delimiter = __Gemini_user_delimiter
        _system_delimiter = __Gemini_system_delimiter

    elif model == "gemini-1.0-pro-latest":
        _llm_platform = _GOOGLE
        _max_token_limit = __Gemini
        _user_delimiter = __Gemini_user_delimiter
        _system_delimiter = __Gemini_system_delimiter

    elif model == "gemini-1.5-pro-latest":
        _llm_platform = _GOOGLE
        _max_token_limit = __Gemini
        _user_delimiter = __Gemini_user_delimiter
        _system_delimiter = __Gemini_system_delimiter

    else:
        raise Exception(f"Model {model} is not implemented yet!")

    _LLM_API_Key = LLM_API_Key()
