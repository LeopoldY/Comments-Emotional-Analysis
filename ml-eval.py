import pickle
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from utils import ml_preprocess_data, data_loader

data = data_loader("data/online_shopping_10_cats.csv")
# 随机选择 1000 条数据进行测试
data = data.sample(1000)

text = ml_preprocess_data(data.review.values)
label = data.label.values 

clf = pickle.load(open("output/lr.pkl", "rb"))
y_pred = clf.predict(text)

print("Accuracy:", accuracy_score(label, y_pred))
print("Classification Report:\n", classification_report(label, y_pred))