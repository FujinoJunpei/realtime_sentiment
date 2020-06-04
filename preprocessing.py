import pandas as pd
from neologdn import normalize
import emoji

def preprocess_text(df):
    text = df["text"]
    text = text.str.strip() # 余分な空白削除
    text = text.apply(normalize) # 正規化(半角/全角変換など)
    text = text.str.lower() # 大文字→小文字
    text= text.str.replace("\n", " ") # 改行コード削除
    
    # 絵文字を置換
    for i in range(len(text)):
        text[i] = emoji.demojize(text[i], delimiters=("",""))
    
    df["text"] = text
    
    return df

def make_jsonl(df):
    with open("input.jsonl", "w") as f:
        for _, row in df.iterrows():
            f.write(f'{{"id":{int(row[0])},"text":"{row[1]}"}}\n')
