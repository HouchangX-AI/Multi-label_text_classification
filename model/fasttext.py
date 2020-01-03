import fasttext
import logging
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from model.metrics import f1_np


if __name__ == '__main__':
    # 切分数据集
    # !head -n 19576 ./data/kkb/fasttext/baidu_95__label__.txt > ./data/kkb/fasttext/baidu_95__label__.train
    # !tail -n 3000 ./data/kkb/fasttext/baidu_95__label__.txt > ./data/kkb/fasttext/baidu_95__label__.valid

    train_file = r'./data/kkb/fasttext/baidu_95__label__.train'
    valid_file = r'./data/kkb/fasttext/baidu_95__label__.valid'

    model = fasttext.train_supervised(input=train_file, epoch=1000, wordNgrams=5, bucket=200000, dim=50, loss='ova')
    label_true, label_pred = [], []

    # 验证模型
    with open(valid_file) as f:
        for line in f.readlines():
            labels = line.split()[:-1]
            string = line.split()[-1]
            predicts = model.predict(string, k=2)
            label_true.append(set(labels))
            label_pred.append(set(predicts[0]))

    # 评估模型
    mlb = MultiLabelBinarizer()
    y_true = mlb.fit_transform(label_true)
    y_pred = mlb.transform(label_pred)
    f1_np(y_true, y_pred)