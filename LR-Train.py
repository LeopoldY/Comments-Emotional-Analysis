import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_curve   

from utils import ml_preprocess_data, load_data
import matplotlib.pyplot as plt

dataPath = "data/online_shopping_10_cats.csv"
data = load_data(dataPath)
isPreprocess = False

# 将分词后的文本列表转换为 TF-IDF 特征矩阵
if isPreprocess:
    tfidf_matrix = ml_preprocess_data(data.review.values, max_length=64)
    # 保存 TF-IDF 特征矩阵
    pickle.dump(tfidf_matrix, open("data/tfidf_matrix.pkl", "wb"))
else:
    tfidf_matrix = pickle.load(open("data/tfidf_matrix.pkl", "rb"))
y = data.label.values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.2, random_state=42)

# 初始化 LR 分类器
clf = LogisticRegression(verbose=3)

# 在训练集上训练 LR 分类器
clf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = clf.predict(X_test)

# 输出模型评估指标
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr)
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

plt.savefig("output/roc_curve.png")

# 保存模型
pickle.dump(clf, open("output/lr.pkl", "wb"))
