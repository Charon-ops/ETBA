# !/usr/bin/python
# -*- coding: utf-8 -*-
# !/usr/bin/python
# -*- coding: utf-8 -*-
# @time    : 2021/2/23 21:34
# @author  : Mo
# @function: 多标签分类, 根据label是否有|myz|分隔符判断是多类分类, 还是多标签分类


# 适配linux
import platform
import json
import sys
import os

path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
path_sys = os.path.join(path_root, "Pytorch-NLU-main", "pytorch_nlu", "pytorch_textclassification")
print(path_root)
# 分类下的引入, pytorch_textclassification
from tcTools import get_current_time
from tcRun import TextClassification
from tcConfig import model_config


evaluate_steps = 320  # 评估步数
save_steps = 320  # 存储步数
# pytorch预训练模型目录, 必填
pretrained_model_name_or_path = "bert-base-chinese"
# 训练-验证语料地址, 可以只输入训练地址
path_corpus = os.path.join(path_root, "Data", "Data20230109")
path_train = os.path.join(path_corpus, 'train.txt')
path_dev = os.path.join(path_corpus, 'test.txt')
model_type = ["BERT", "ERNIE", "BERT_WWM", "ALBERT", "ROBERTA", "XLNET", "ELECTRA"]
idx = 0

if __name__ == "__main__":
    model_config["evaluate_steps"] = evaluate_steps  # 评估步数
    model_config["save_steps"] = save_steps  # 存储步数
    model_config["path_train"] = path_train  # 训练模语料, 必须
    model_config["path_dev"] = path_dev  # 验证语料, 可为None
    model_config["path_tet"] = None  # 测试语料, 可为None
    # 损失函数类型,
    # multi-class:  可选 None(BCE), BCE, BCE_LOGITS, MSE, FOCAL_LOSS, DICE_LOSS, LABEL_SMOOTH
    # multi-label:  SOFT_MARGIN_LOSS, PRIOR_MARGIN_LOSS, FOCAL_LOSS, CIRCLE_LOSS, DICE_LOSS等
    model_config["path_tet"] = None
    os.environ["CUDA_VISIBLE_DEVICES"] = str(model_config["CUDA_VISIBLE_DEVICES"])

    model_config["pretrained_model_name_or_path"] = pretrained_model_name_or_path
    model_config["model_save_path"] = "../output/text_classification/model_{}".format(model_type[idx])

    model_config["model_type"] = "BERT"
    # main
    lc = TextClassification(model_config)
    lc.process()
    lc.train()
