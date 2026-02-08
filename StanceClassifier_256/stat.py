import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

CSV_PATH = "data/rumoureval_2019/rumoureval2019_train.csv"
MODEL_NAME = "GateNLP/stance-bertweet-target-aware"   # 改成你实际用的
COL_A = "source_text"             # 改成你实际列名
COL_B = "reply_text"              # 改成你实际列名

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

df = pd.read_csv(CSV_PATH)
# 可选：丢掉没label的行，和你训练保持一致
df = df.dropna(subset=["label"])

lengths = []
batch_size = 256

for i in tqdm(range(0, len(df), batch_size)):
    a = df[COL_A].iloc[i:i+batch_size].astype(str).tolist()
    b = df[COL_B].iloc[i:i+batch_size].astype(str).tolist()

    enc = tokenizer(
        a, b,
        add_special_tokens=True,
        truncation=False,   # 关键：不截断，测真实长度
        padding=False,
        return_attention_mask=False
    )
    # 每条样本的 token 数
    lens = [len(ids) for ids in enc["input_ids"]]
    lengths.extend(lens)

lengths = np.array(lengths)

for th in [128, 256, 512]:
    cnt = int((lengths > th).sum())
    print(f"> {th}: {cnt}/{len(lengths)} = {cnt/len(lengths):.2%}")

print("p50/p90/p95/p99 =", np.percentile(lengths, [50, 90, 95, 99]))
print("max =", lengths.max())
