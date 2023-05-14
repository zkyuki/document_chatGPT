import os
import time

import numpy as np
import openai
import pandas as pd
from PyPDF2 import PdfReader
from scipy.spatial import distance

# OpenAIのapiキー
openai.api_key = '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'

def get_summary(result):
    system = """与えられた論文の要点を3点以内でまとめ、以下のフォーマットで日本語で出力してください。```
    ・要点1
    ・要点2
    ・要点3
    ```"""

    text = f"{result}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': text}
        ],
        temperature=0.25,
    )
    summary = response['choices'][0]['message']['content']
    return summary


def make_answer(result, query):
    system = """与えられた文章を読んで、最後の行の質問に日本語で回答してください。
    """
    template = """
    # 問題文：
    {result}

    # 質問：{query}
    """
    text = template.format(result=result, query=query)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': text}
        ],
        temperature=0.25,
    )
    summary = response['choices'][0]['message']['content']
    return summary


def make_answer(result, query):
    system = """Read the given text and answer the question on the last line in Japanese.
    """
    template = """
    # TEXT
    {result}

    # QUESTION
    {query}
    """
    text = template.format(result=result, query=query)
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': text}
        ],
        temperature=0.25,
    )
    summary = response['choices'][0]['message']['content']
    return summary


with open("Stripe.pdf", "rb") as input:
    reader = PdfReader(input)
    print("GPT-4_Technical_Report has %d pages.\n" % len(reader.pages))

    df_embedding = pd.DataFrame(index=range(
        len(reader.pages)), columns=range(1536))
    for p_index in range(len(reader.pages)):
        page = reader.pages[p_index]
        text = page.extract_text()
        print(p_index, '---')
        print(get_summary(text))

        # ベクトル化
        response = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002"
        )
        df_embedding.loc[p_index, :] = response['data'][0]['embedding']
        time.sleep(1)

        df_embedding.to_csv('./page_embedding.csv', encoding='utf_8_sig')

df_embedding


statement = "2022年のワールドカップの結果を教えて"

input_vec = openai.Embedding.create(
    input=statement,
    model="text-embedding-ada-002"
)
input_vec = input_vec['data'][0]['embedding']

dist_M = distance.cdist(np.array(input_vec).reshape(
    1, -1), df_embedding, metric='cosine')
dist_M = pd.Series(dist_M[0], index=df_embedding.index)
dist_M = dist_M.sort_values(ascending=False)
close_ids = dist_M.index[:10]
close_ids

with open("Stripe.pdf", "rb") as input:
    reader = PdfReader(input)

    text = ""
    for p_index in close_ids:
        page = reader.pages[p_index]
        text += page.extract_text() + '\n\n'
print(statement)
print(make_answer(text, statement))

statement = "Please itemize the ethical concerns in using the GPT-4."

input_vec = openai.Embedding.create(
    input=statement,
    model="text-embedding-ada-002"
)
input_vec = input_vec['data'][0]['embedding']

dist_M = distance.cdist(np.array(input_vec).reshape(
    1, -1), df_embedding, metric='cosine')
dist_M = pd.Series(dist_M[0], index=df_embedding.index)
dist_M = dist_M.sort_values(ascending=False)
close_ids = dist_M.index[:3]
close_ids

with open("Stripe.pdf", "rb") as input:
    reader = PdfReader(input)

    text = ""
    for p_index in close_ids:
        page = reader.pages[p_index]
        text += page.extract_text() + '\n\n'
print(statement)
print(make_answer(text, statement))
