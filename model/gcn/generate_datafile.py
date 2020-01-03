# coding=utf8
import pandas as pd
import jieba
import numpy as np

"""
Generate dataset_files

/data/dataset.txt 是数据的ID train/test标识 (多)类别
/data/corpus/dataset.txt 是数据的原始文本数据
"""

dataset = 'baidu_95'
dataset_file = f'../../data/{dataset}.csv'
stopwords_file = './data/stopwords.txt'

df = pd.read_csv(dataset_file, header=None, names=["labels", "item"], dtype=str)

# shuffle
# the baidu_95.csv is shuffled

# cut words
df['item'] = df.item.apply(lambda x: list(jieba.cut(x)))

# remove stopwords
# 哈工大停用词 https://github.com/goto456/stopwords
stopwords = open(stopwords_file, encoding='utf8').readlines()
stopwords = [i.strip() for i in stopwords]
df['item'] = df.item.apply(lambda x: [word for word in x if word not in stopwords])

# write to corpus/baidu_95.clean.txt corpus/baidu_95.txt ./baidu_95.txt
with open(f'./data/corpus/{dataset}.txt', 'w') as f:
    for line in df.item:
        f.write(' '.join(line)+'\n')

with open(f'./data/corpus/{dataset}.clean.txt', 'w', encoding='utf8') as f:
    for line in df.item:
        f.write(' '.join(line)+'\n')

# split dataset
# sublabel 先使用单标签分类验证模型效果
# sublabel = 2
index_split = len(df) * 0.9
with open(f'./data/{dataset}.txt', 'w', encoding='utf8') as f:
    for index, row in df.iterrows():
        category = 'train' if index <= index_split else 'test'
        # single lable
        # f.write(f'{index}\t{category}\t{row[0].split()[sublabel]}\n')
        # multi_label
        lables = '\t'.join(row[0].split())
        f.write(f"{index}\t{category}\t{lables}\n")


print(f'Dataset{dataset} file is generated, please use build_graph!')