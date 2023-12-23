import pandas
import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from models import BertSentimentClassifier
from utils import preprocess_data, data_loader

import time

isPreprocess = False
BATCH_SIZE = 4
EPOCHS = 4
LEARNING_RATE = 2e-5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

dataPath = "data/online_shopping_10_cats.csv"
data = data_loader(dataPath)

# 数据预处理
if isPreprocess:
    input_ids, attention_masks = preprocess_data(data.review.values, max_length=512)
    # 保存预处理后的数据
    torch.save(input_ids, "data/input_ids.pt.data")
    torch.save(attention_masks, "data/attention_masks.pt.data")
else:
    input_ids = torch.load("data/input_ids.pt.data")
    attention_masks = torch.load("data/attention_masks.pt.data")

# 构建数据集
dataset = TensorDataset(input_ids, attention_masks, torch.tensor(data.label.values))
# 划分训练集、验证集和测试集
train_size = int(0.7 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# 构建数据加载器
batch_size = BATCH_SIZE
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

# 模型训练
device = torch.device(DEVICE)
model = BertSentimentClassifier(num_classes=2).to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
epochs = EPOCHS
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
loss = torch.nn.CrossEntropyLoss().to(device)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train():
    print(f"trainnig on {device}")
    print('Number of train dataset:', len(train_dataset))
    print('Number of validation dataset:', len(val_dataset))
    print('Number of test dataset:', len(test_dataset))
    model.train()
    total_train_loss = 0
    total_train_accuracy = 0
    for epoch in range(epochs):
        print("Epoch:", epoch + 1)
        start_time = time.time()
        for step, batch in enumerate(train_dataloader): 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            model.zero_grad() # 梯度清零
            outputs = model(b_input_ids, b_input_mask) # 前向传播
            loss_value = loss(outputs, b_labels) # 计算损失
            total_train_loss += loss_value.item() # 累计损失
            loss_value.backward() # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪
            optimizer.step() # 更新参数
            scheduler.step() # 更新学习率
            
            logits = outputs.detach().cpu().numpy() # 将输出转移到CPU上
            label_ids = b_labels.to('cpu').numpy() # 将标签转移到CPU上
            total_train_accuracy += flat_accuracy(logits, label_ids) # 累计准确率
            
            if step % 1000 == 0: 
                print("Step:", step, "Loss:", loss_value.item(), "Accuracy:", flat_accuracy(logits, label_ids))
        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_accuracy = total_train_accuracy / len(train_dataloader)
        print("Average training loss:", avg_train_loss)
        print("Average training accuracy:", avg_train_accuracy)
        print("Training epoch took:", time.time() - start_time, "s")
        torch.save(model.state_dict(), "output/BERT_model_epoch_" + str(epoch + 1) + ".pth")
    return avg_train_loss, avg_train_accuracy

train_loss, train_accuracy = train()
print("Training loss:", train_loss)
print("Training accuracy:", train_accuracy)
