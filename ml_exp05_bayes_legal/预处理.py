import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']  #解决中文显示乱码问题
plt.rcParams['axes.unicode_minus']=False
import warnings
warnings.filterwarnings(action = 'ignore')
from scipy.stats import beta
from sklearn.naive_bayes import GaussianNB
import sklearn.linear_model as LM
from sklearn.model_selection import cross_val_score,cross_validate,train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc,accuracy_score,precision_recall_curve
#本节增加导入的模块
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import  TfidfVectorizer
import json
from sklearn.naive_bayes import MultinomialNB

documents = ["中国的发展是开放的发展",
    "中国经济发展的质量在稳步提升，人民生活在持续改善",
    "从集市、超市到网购，线上年货成为中国老百姓最便捷的硬核年货",
    "支付体验的优化以及物流配送效率的提升，线上购物变得越来越便利"]
documents = [" ".join(jieba.cut(item)) for item in documents]
print("文本分词结果：\n",documents)
vectorizer = TfidfVectorizer()  #定义TF-IDF对象
X = vectorizer.fit_transform(documents)

words=vectorizer.get_feature_names()
print("特征词表：\n",words)
print("idf:\n",vectorizer.idf_)  #idf
X=X.toarray() #print(X.toarray())   #文本-词的tf-idf矩阵
for i in range(len(X)): ##打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    for j in range(len(words)):
        print(words[j],X[i][j])