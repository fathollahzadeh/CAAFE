import pandas as pd


def read_text_file_line_by_line(fname: str):
    try:
        with open(fname) as f:
            lines = f.readlines()
            raw = "".join(lines)
            return raw
    except Exception as ex:
        raise Exception(f"Error in reading file:\n {ex}")


def save_text_file(fname: str, data):
    try:
        f = open(fname, 'w')
        f.write(data)
        f.close()
    except Exception as ex:
        raise Exception(f"Error in save file:\n {ex}")


def save_prompt(fname: str, system_message: str, user_message: str):
    prompt_items = ["SYSTEM MESSAGE:\n",
                    system_message,
                    "\n---------------------------------------\n",
                    "PROMPT TEXT:\n",
                    user_message]
    prompt_data = "".join(prompt_items)
    save_text_file(fname=fname, data=prompt_data)


def reader_CSV(filename):
    df = pd.read_csv(filename, low_memory=False)
    return df
