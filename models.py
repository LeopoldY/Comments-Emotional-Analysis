import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F

class BertSentimentClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super(BertSentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = F.sigmoid(self.classifier(pooled_output))
        return logits