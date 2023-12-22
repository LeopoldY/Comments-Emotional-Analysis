from transformers import BertTokenizer
import torch
import pandas

def data_loader(path):
    data = pandas.read_csv(path, sep=",")
    data = data.dropna() # 去除缺失值
    data = data.drop_duplicates() # 去除重复值
    data = data.reset_index(drop=True) # 重置索引
    return data

def preprocess_data(data, max_length=128):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  # 加载中文BERT tokenizer
    
    input_ids = []
    attention_masks = []
    
    for text in data.review:
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

def ml_preprocess_data(data):
    with open('data/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = f.read()
    stopwords = stopwords.split('\n')
    stopwords = set(stopwords)

    tfidf_matrix = []
    words = []
    for text in data.review:
        text = jieba.cut(text)
        text = [word for word in text if word not in stopwords]
        words.append(text)

    # 将每个句子的单词列表转换为字符串形式，以空格连接单词
    text_list = [' '.join(words) for words in words]

    # 初始化 TF-IDF 向量化器
    tfidf_vectorizer = TfidfVectorizer()

    # 将文本列表转换为 TF-IDF 特征矩阵
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_list)

    # 将 TF-IDF 特征矩阵转换为稀疏矩阵形式
    tfidf_matrix_array = tfidf_matrix.toarray()
    print('Data preprocessing completed.')
    return tfidf_matrix_array

