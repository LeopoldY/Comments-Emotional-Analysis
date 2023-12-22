import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from utils import ml_preprocess_data, data_loader

# 示例中文情感分类数据集（假设已准备好）
# 假设X为文本特征，y为对应的情感标签（0代表负面情感，1代表正面情感）

dataPath = "data/online_shopping_10_cats.csv"
data = data_loader(dataPath)

# 将分词后的文本列表转换为 TF-IDF 特征矩阵
tfidf_matrix = ml_preprocess_data(data)
y = data.label.values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.2, random_state=42)

# 初始化 LR 分类器
clf = LogisticRegression(verbose=1)

# 在训练集上训练 SVM 分类器
clf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = clf.predict(X_test)

# 输出模型评估指标
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
