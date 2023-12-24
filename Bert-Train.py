import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from models import BertSentimentClassifier
from utils import preprocess_data, load_data, flat_accuracy

import time

isPreprocess = False
BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 2e-5
DEVICE = 'cpu'
if DEVICE == 'cuda':
    torch.backends.cudnn.benchmark = True

dataPath = "data/online_shopping_10_cats.csv"
data = load_data(dataPath)

# 数据预处理
if isPreprocess:
    input_ids, attention_masks = preprocess_data(data.review.values, max_length=64)
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
model = BertSentimentClassifier(num_classes=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
epochs = EPOCHS
total_steps = len(train_dataloader) * epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps) # 余弦退火学习率
loss = torch.nn.BCELoss().to(device) # 二分类交叉熵损失函数

def train():
    print(f"trainnig on {device}")
    print('Number of train dataset:', len(train_dataset))
    print('Number of validation dataset:', len(val_dataset))
    print('Number of test dataset:', len(test_dataset))
    model.train()
    train_loss = []
    train_accuracy = []
    for epoch in range(epochs):
        print("Epoch:", epoch + 1)
        start_time = time.time()
        for step, batch in enumerate(train_dataloader): 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            model.zero_grad() # 梯度清零
            outputs = model(b_input_ids, b_input_mask) # 前向传播
            outputs = torch.squeeze(outputs, dim=1)
            loss_value = loss(outputs, b_labels.float()) # 计算损失
            train_loss.append(loss_value.item()) # 记录损失
            
            optimizer.zero_grad() # 梯度清零
            loss_value.backward() # 反向传播
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪

            optimizer.step() # 更新参数
            scheduler.step() # 更新学习率
            
            logits = outputs.detach().cpu().numpy() # 将输出转移到CPU上
            label_ids = b_labels.to('cpu').numpy() # 将标签转移到CPU上
            train_accuracy.append(flat_accuracy(logits, label_ids)) # 记录准确率
            
            print("Step:", step, "Loss:", loss_value.item(), "Accuracy:", train_accuracy[-1], end='\r')
        print()
        print("Training epoch took:", time.time() - start_time, "s")
        torch.save(model.state_dict(), "output/BERT_model_epoch_" + str(epoch + 1) + ".pth")
    return train_loss, train_accuracy

train_loss, train_accuracy = train()
avg_train_loss = sum(train_loss) / len(train_loss)
avg_train_accuracy = sum(train_accuracy) / len(train_accuracy)
print("Training loss:", avg_train_loss)
print("Training accuracy:", avg_train_accuracy)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(train_loss, label='train loss')
plt.plot(train_accuracy, label='train accuracy')
plt.legend()
plt.show()
plt.savefig("output/train_loss_accuracy.png")