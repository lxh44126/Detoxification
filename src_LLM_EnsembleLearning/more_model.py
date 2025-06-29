# 包
import time
import requests
import json
import pandas as pd
import os
from tqdm import tqdm as dq
import time
import random
import re
import csv
import time
import tsv
from openai import OpenAI
from evaluate_4 import main_evaluate as evaluate


def upload_file_prompt(input_fold,rate = 0.1):
    examples_dict = {}
    file_name_list = []
    for root, dirs, files in os.walk(input_fold):
        for file in files:
            filepath = os.path.join(root, file)
            file_name_list.append(filepath)
            lang_type = str(file).split("-")[0]

            messages = "以下是可以参考的数据：\n"
            try:
                df = pd.read_parquet(str(filepath))

                required_columns = {"toxic_sentence", "neutral_sentence"}
                if not required_columns.issubset(df.columns):
                    raise ValueError(f"文件 {filepath} 缺少必要的列：{required_columns - set(df.columns)}")

                for _, row in df.iterrows():
                    toxic_sentence = row["toxic_sentence"]
                    neutral_sentence = row["neutral_sentence"]
                    if random.random() < rate:
                        messages += f"[<|toxic_sentence|>{toxic_sentence}<|toxic_sentence|>,<|neutral_sentence|>{neutral_sentence}<|neutral_sentence|>,<|lang|>{lang_type}<|lang|>"
                examples_dict[lang_type] = messages
            except Exception as e:
                pass
    return examples_dict


def create_vocabulary_list(input_fold):
    examples_dict = {}
    file_name_list = []
    for root, dirs, files in os.walk(input_fold):
        for file in files:
            filepath = os.path.join(root, file)
            file_name_list.append(filepath)
            lang_type = str(file).split(".")[0]

            try:
                df = pd.read_parquet(str(filepath),engine="pyarrow")
                required_columns = {"text"}
                if not required_columns.issubset(df.columns):
                    raise ValueError(f"文件 {filepath} 缺少必要的列：{required_columns - set(df.columns)}")
                one = []
                tow = []
                three = []
                more = []
                for _, row in df.iterrows():
                    length = 0
                    if lang_type in ["zh", "jz"]:
                        length = len(row["text"])
                    else:
                        length = len(row["text"].split(" "))
                    if length == 1:
                        one.append(row["text"])
                    elif length == 2:
                        tow.append(row["text"])
                    elif length == 3:
                        three.append(row["text"])
                    else:
                        more.append(row["text"])
                examples_dict[lang_type] = {
                    '1': one,
                    "2": tow,
                    "3": three,
                    "4": more
                }
            except Exception as e:
                print(str(e))
                pass
    return examples_dict


def generate_prompt():
    return """
Role:
Assume you are an expert in language processing.

Inputs:
I have received a batch of toxic sentences (sentences containing harmful text) <|toxic_sentence|> and their detoxified neutral counterparts (sentences with harmful text removed) <|neutral_sentence|> in the context object.

Steps:
You can detoxify the sentences by removing harmful keywords or directly optimizing the sentences to convert them into neutral versions.

Expectation of the result:
Return the data in JSON format as follows:
{
    "toxic_sentence": "",  
    "neutral_sentence": "",  
    "lang": ""  
}
Where:
lang is the language type: en, ru, uk, de, es, am, zh, ar, hi, it, fr, he, hin, tt, ja.
neutral_sentence is the detoxified neutral sentence, which should retain the original content while ensuring semantic similarity (measured by cosine similarity between LaBSE embeddings) and maintaining good fluency.
toxic_sentence is the original toxic sentence before detoxification.
"""


def generate_messages(data, mode, examples=None, lang=None):
    """
    :param lang:
    :param examples:
    :param data: 有毒文本
    :param mode: 提示词模式
    :return: message队列
    """

    prompt_text = None
    if mode == "co-star":
        prompt_text = generate_prompt()

    message_model = [{
        "role": "system",
        "content": prompt_text
    }, {
        "role": "system",
        "content": examples
    }, {
        "role": "system",
        "content": f"""I will convert the provided <|toxic_sentence|> into <|neutral_sentence|> and only output the results in JSON format as specified.：
            {{
                "toxic_sentence": "",
                "neutral_sentence": ""，
                “lang": "{lang}"

            }}
            """
    }, {
        "role": "user",
        "content": f"""
                <|toxic_sentence|>{data}<|toxic_sentence|>.
            """
    }]

    return message_model


def parse_response(response):
    """
    :param response: Json
    :return: [toxic_sentence, neutral_sentence, lang]
    """
    try:
        response_data = str(response["data"])
        json_data = json.loads(response_data)
        content = json_data["choices"]
        content = content[0]
        content = content["message"]
        content = content["content"]
        # 找到 JSON 部分的起始和结束位置
        start = content.find('{')
        end = content.rfind('}') + 1
        json_str = content[start:end]
        # 解析 JSON 字符串
        json_data = json.loads(json_str)
        toxic_sentence = json_data["toxic_sentence"]
        neutral_sentence = json_data["neutral_sentence"]
        lang = json_data["lang"]
        return [toxic_sentence, neutral_sentence, lang]
    except Exception as e:
        try:
            response_data = response["data"]
            content = response_data["choices"]
            content = content[0]
            content = content["message"]
            content = content["content"]
            # 找到 JSON 部分的起始和结束位置
            start = content.find('{')
            end = content.rfind('}') + 1
            json_str = content[start:end]
            # 解析 JSON 字符串
            json_data = json.loads(json_str)
            toxic_sentence = json_data["toxic_sentence"]
            neutral_sentence = json_data["neutral_sentence"]
            lang = json_data["lang"]
            return [toxic_sentence, neutral_sentence, lang]
        except Exception as e:
            return None


def deepseek_request(messages):
    url = "https://deepseek.fosu.edu.cn/v1/chat/completions"
    api_key = "sk-XXXX"

    data = {
        "model": "DeepSeek-R1-Int8",
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 1024
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        return {
            "status": "success",
            "message": "success",
            "data": response.json()
        }
    except Exception as e:
        print("ds error:",str(e))
        return {
            "status": "fail",
            "message": str(e),
            "data": messages
        }


def kimi_request(messages):
    try:
        # response = requests.request("POST", url, json=payload, headers=headers)
        client = OpenAI(
            api_key="sk-XXXXX",  # 在这里将 MOONSHOT_API_KEY 替换为你从 Kimi 开放平台申请的 API Key
            base_url="https://api.moonshot.cn/v1",
        )

        completion = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=messages,
            temperature=0.3,
        )

        return {
            "status": "success",
            "message": "success",
            "data": completion.model_dump_json()
        }

    except Exception as e:
        print("qwen error:",str(e))
        return {
            "status": "fail",
            "message": str(e),
            "data": messages
        }


def qwen3_request(messages):

    try:
        # response = requests.request("POST", url, json=payload, headers=headers)
        client = OpenAI(
        api_key="sk-XXXX",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
        model="qwen-plus", # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=messages
        )

        return {
            "status": "success",
            "message": "success",
            "data": completion.model_dump_json()
        }

    except Exception as e:
        print("qwen error:",str(e))
        return {
            "status": "fail",
            "message": str(e),
            "data": messages
        }


# 替换有毒词汇
def replace_bad_words(text, start, end, replacement="*"):
    """替换文本中从 start 到 end 的部分为指定字符（默认 *）"""
    return text[:start] + replacement * (end - start) + text[end:]


def replace_toxic(toxic_sentence, bad_words, lang=None):
    if not toxic_sentence or not bad_words:
        return toxic_sentence

    # 中文/日文直接按字符处理（不拆分）
    if lang in ["ja", "zh"]:
        text = toxic_sentence
        # 按短语长度从长到短检查（优先匹配更长的敏感词）
        for phrase_length in sorted(bad_words.keys(), key=lambda x: -int(x)):
            phrases = bad_words.get(phrase_length, [])
            if not phrases:
                continue
            phrase_len = int(phrase_length)
            for phrase in phrases:
                start = 0
                while True:
                    # 查找敏感词出现的位置
                    idx = text.find(phrase, start)
                    if idx == -1:
                        break
                    # 替换为等长 ***
                    text = replace_bad_words(text, idx, idx + len(phrase))
                    start = idx + 1  # 避免重复检查已替换部分
    else:
        # 英文等按空格分词处理
        words = toxic_sentence.split()
        for phrase_length in sorted(bad_words.keys(), key=lambda x: -int(x)):
            phrases = bad_words.get(phrase_length, [])
            if not phrases:
                continue
            phrase_len = int(phrase_length)
            for i in range(len(words) - phrase_len + 1):
                candidate = " ".join(words[i:i + phrase_len])
                if candidate in phrases:
                    words[i:i + phrase_len] = ["*"] * phrase_len
        text = " ".join(words)

    return text


def main_among_ai(filename, output_file, vocabulary_fold=None, examples_fold=None, prompt_mode="co-star"):
    with open(file=output_file, newline="", mode="w", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows([["toxic_sentence", "neutral_sentence", "lang"]])
        f.close()

    examples_messages = None
    if examples_fold is not None:
        examples_messages = upload_file_prompt(examples_fold,0.2)

    examples_messages_kimi = None
    if examples_fold is not None:
        examples_messages_kimi = upload_file_prompt(examples_fold,0.05)


    vocabulary_dict = None
    if vocabulary_fold is not None:
        vocabulary_dict = create_vocabulary_list(vocabulary_fold)

    df = pd.read_csv(filename, sep='\t', encoding='utf-8')
    toxic_sentences = df['toxic_sentence'].tolist()
    languages = df['lang'].tolist()
    with open(file=output_file, newline="", mode="a", encoding="utf-8") as output_file:
        writer = csv.writer(output_file, delimiter="\t")

        for toxic, lang in zip(toxic_sentences, languages):
            print(f"处理的{lang}句子:", toxic)

            messages = generate_messages(toxic, prompt_mode,
                                         examples=examples_messages[lang], lang=lang)

            messages_kimi = generate_messages(toxic, prompt_mode,
                                         examples=examples_messages_kimi[lang], lang=lang)

            response_list = [deepseek_request(messages),qwen3_request(messages=messages),kimi_request(messages_kimi)]

            neutral_sentence_list = []
            for response in response_list:
                if response["status"] == "success":
                    result = parse_response(response)
                    if result is not None:
                        neutral_sentence_list.append(result[1])
                    else:
                        neutral_text = replace_toxic(toxic_sentence=toxic, bad_words=vocabulary_dict[lang], lang=lang)
                        neutral_sentence_list.append(neutral_text)

                else:
                    neutral_text = replace_toxic(toxic_sentence=toxic, bad_words=vocabulary_dict[lang], lang=lang)
                    neutral_sentence_list.append(neutral_text)
            neutral_sentence_list.append(replace_toxic(toxic_sentence=toxic,bad_words=vocabulary_dict[lang],lang=lang))
            evaluate_result = evaluate(toxic,neutral_sentence_list,lang)

            writer.writerow(evaluate_result)


if __name__ == '__main__':
    main_among_ai(filename="./test_inputs_upd.tsv",
                  output_file="dev_outputs_upd_ar.tsv",
                  examples_fold="./examples_data",
                  vocabulary_fold="./toxi_text_list")
