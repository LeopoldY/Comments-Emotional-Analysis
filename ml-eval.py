import pickle
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from utils import ml_preprocess_data, load_data

# data = data_loader("data/online_shopping_10_cats.csv")
# # 随机选择 1000 条数据进行测试
# data = data.sample(1000)

with open('data/ml-eval-data.txt', 'r', encoding='utf-8') as f:
    data = f.read().split('\n')
    data = [line.split(' ') for line in data]

text = [line[1] for line in data]
label = [int(line[0]) for line in data]
text = ml_preprocess_data(text, max_length=64)

clf = pickle.load(open("output/lr.pkl", "rb"))
y_pred = clf.predict(text)

print("Accuracy:", accuracy_score(label, y_pred))
print("Classification Report:\n", classification_report(label, y_pred))
print("Predicted label:", y_pred)
print("True label:", label)