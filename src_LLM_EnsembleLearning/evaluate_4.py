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

def generate_prompt():
    return """
任务说明：
我正在执行自然语言处理的数据去毒化任务，对输入的
{{
"toxic_sentence": "原始文本",
"neutral_sentence": ["解毒文本列表"], 
"lang": "语言代码"
}}
需要对输入的neutral_sentence进行最优选取。

工作要求：
1. 选择原则：
- 严格保持和toxic_sentence相同的语言和情感表达
- 仅进行毒性最小化选择
- 保留所有原始格式（包括标点、大小写、空格等）

2. 选取标准：
- 毒性程度最低（30%占比）
- 语义相似度最高（40%占比）
- 语句流畅自然(20%)
- 风格保持一致(10%)

3. 评审标准：
- 由专业语言学家评估
- 重点考察毒性降低程度和语义保真度

输出格式：
{{
    "toxic_sentence": "原始文本",
    "neutral_sentence": "较好的去毒化文本", 
    "lang": "语言代码"
}}
支持语言：en,zh,ru,uk,de,es,fr,it,ja,ar,hi等
"""

def generate_messages(data):

    message_model = []
    message_model.append({
        "role": "system",
        "content": generate_prompt()
    })

    message_model.append({
        "role": "user",
        "content": data
    })

    return message_model


# 加载模型
def create_model_request(messages):
    try:
        # response = requests.request("POST", url, json=payload, headers=headers)
        client = OpenAI(
            api_key="sk-19069cb47c2742ceb2e89d1b80521488",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model="qwen-plus",  # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=messages
        )

        return {
            "status": "success",
            "message": "success",
            "data": completion.model_dump_json()
        }

    except Exception as e:
        print("qwen error:", str(e))
        return {
            "status": "fail",
            "message": str(e),
            "data": messages
        }


def parse_response_deepseek__Si_CO_STAR(response):
    """
    :param response: Json
    :return: [toxic_sentence, neutral_sentence, lang]
    """
    try:
        response_data = response["data"]
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
            print("大模型评价:",str(e))
            return None


def main_evaluate(toxic_sentence,neutral_sentence_list,lang):
    data = f"""
        {{
        "toxic_sentence": "{toxic_sentence}",
        "neutral_sentence": ["{neutral_sentence_list[0]}","{neutral_sentence_list[1]}","{neutral_sentence_list[2]}","{neutral_sentence_list[3]}"],
        "lang": "{lang}"
        }}
        """
    print(data)
    prompt = generate_messages(data)
    try:
        resp = create_model_request(prompt)
        result = parse_response_deepseek__Si_CO_STAR(resp)

        result[0] = toxic_sentence
        if result[1] not in neutral_sentence_list:
            for i in range(2):
                resp = create_model_request(prompt)
                neutral_text = parse_response_deepseek__Si_CO_STAR(resp)[1]
                if neutral_text in neutral_sentence:
                    result[1] = neutral_text
                    break
                else:
                    if i == 2:
                        result[1] = neutral_sentence[2]

        return result
    except Exception as e:
        return [toxic_sentence,neutral_sentence_list[0],lang]

if __name__ == '__main__':
    result = main_evaluate(
        "Gegen die Dummheit dieser Frau gibts doch bestimmt was von Ratiopharm",
        ["Gegen das Problem dieser Frau gibt es doch bestimmt etwas von Ratiopharm","Gegen die Dummheit dieser * gibts doch * was von * ","Gegen die Dummheit dieser Frau gibts doch bestimmt was von Ratiopharm"],
       "de"

    )
    print(result)