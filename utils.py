from transformers import BertTokenizer, AutoTokenizer
import torch
import pandas
import numpy as np

def flat_accuracy(preds, labels):
    '''
    计算准确率
    :param preds: 预测结果
    :param labels: 真实标签
    :return: 准确率
    '''
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def data_loader(path):
    data = pandas.read_csv(path, sep=",")
    data = data.dropna() # 去除缺失值
    data = data.drop_duplicates() # 去除重复值
    data = data.reset_index(drop=True) # 重置索引
    return data

def preprocess_data(txt_data, max_length=128):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')  # 加载中文BERT tokenizer
    
    input_ids = []
    attention_masks = []
    
    for text in txt_data:
        encoded_text = tokenizer.encode_plus(
            text,  # 输入文本
            add_special_tokens=True,  # 添加特殊标记（[CLS]和[SEP]）
            max_length=max_length,  # 设置最大长度
            padding='max_length',  # 填充到最大长度
            truncation=True,  # 截断文本
            return_tensors='pt',  # 返回PyTorch张量
            return_attention_mask=True  # 返回attention mask
        )
        
        input_ids.append(encoded_text['input_ids'])
        attention_masks.append(encoded_text['attention_mask'])
    
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    print('Data preprocessing completed.')
    return input_ids, attention_masks


import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

def ml_preprocess_data(txt_data, max_length=128):
    with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read()
    stopwords = stopwords.split('\n')
    stopwords = set(stopwords)

    tfidf_matrix = []
    words = []
    for text in txt_data:
        word_list = jieba.cut(text)
        word_list = [word for word in word_list if word not in stopwords]
            # 如果文本长度不够，用 0 填充
        if len(word_list) < max_length:
            word_list.extend(['0'] * (max_length - len(word_list)))
        words.append(word_list)
        tfidf_matrix.append(' '.join(word_list))
    tfidf = TfidfVectorizer(max_features=max_length)
    tfidf_matrix = tfidf.fit_transform(tfidf_matrix).toarray()
    print('Data preprocessing completed.')
    return tfidf_matrix

