# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/7/25 9:30
# @author  : Mo
# @function: predict model, 预测模块
import json
# 适配linux
import sys
import os
import pandas as pd

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(path_root)
from tcConfig import model_config

os.environ["CUDA_VISIBLE_DEVICES"] = model_config.get("CUDA_VISIBLE_DEVICES", "0")
from tcPredict import TextClassificationPredict

if __name__ == "__main__":
    # BERT-base = 8109M
    path_config = "../output/text_classification/model_BERT/tc.config"
    # path_config = "../output/text_classification/model_ERNIE/tc.config"
    tcp = TextClassificationPredict(path_config)
    # texts = [{"text": "平乐县，古称昭州，隶属于广西壮族自治区桂林市，位于广西东北部，桂林市东南部，东临钟山县，南接昭平，西北毗邻阳朔，北连恭城，总面积1919.34平方公里。"},
    #          {"text": "平乐县主要旅游景点有榕津千年古榕、冷水石景苑、仙家温泉、桂江风景区、漓江风景区等，平乐县为漓江分界点，平乐以北称漓江，以南称桂江，是著名的大桂林旅游区之一。"},
    #          {"text": "印岭玲珑，昭水晶莹，环绕我平中。青年的乐园，多士受陶熔。生活自觉自治，学习自发自动。五育并重，手脑并用。迎接新潮流，建设新平中"},
    #          {"text": "桂林山水甲天下, 阳朔山水甲桂林"},
    #          ]
    train_texts = []
    test_texts = []
    file_train = open(os.path.join(path_root, 'Data', 'Data20230109', 'train.txt'), 'r', encoding='utf-8')
    file_test = open(os.path.join(path_root, 'Data', 'Data20230109', 'test.txt'), 'r', encoding='utf-8')

    for line in file_train:
        line_json = json.loads(line.strip())
        train_texts.append({'text': line_json["text"]})
    for line in file_test:
        line_json = json.loads(line.strip())
        test_texts.append({'text': line_json["text"]})

    train_res = tcp.predict(train_texts)
    test_res = tcp.predict(test_texts)

    train = {'text': [], 'report_cancer': [], 'report_benign': [], 'report_value': []}
    test = {'text': [], 'report_cancer': [], 'report_benign': [], 'report_value': []}
    for i in range(len(train_texts)):
        train['text'].append(train_texts[i]['text'])
        train['report_cancer'].append(train_res[i]['cancer'])
        train['report_benign'].append(train_res[i]['benign'])
        train['report_value'].append(train_res[i]['cancer'] - train_res[i]['benign'])
    for i in range(len(test_texts)):
        test['text'].append(test_texts[i]['text'])
        test['report_cancer'].append(test_res[i]['cancer'])
        test['report_benign'].append(test_res[i]['benign'])
        test['report_value'].append(test_res[i]['cancer'] - test_res[i]['benign'])

    train = pd.DataFrame(train)
    test = pd.DataFrame(test)
    train.to_excel(os.path.join(path_root, 'Data', 'Data20230109', 'train_nlp.xlsx'), index=False)
    test.to_excel(os.path.join(path_root, 'Data', 'Data20230109', 'test_nlp.xlsx'), index=False)
