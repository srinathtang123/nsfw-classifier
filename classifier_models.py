import torch
import torch.nn as nn
from transformers import (AutoTokenizer,
    AutoModel,
    DistilBertForSequenceClassification,
    DistilBertConfig
    )

class MyClassifier(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device = device
        self.model_name = "ai4bharat/indic-bert"
        self.clf_layers = 2
        self.clf_heads = 4
        self.padding = True
        self.truncation = True
        self.max_length = 100

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.embedder = AutoModel.from_pretrained(self.model_name)
        self.clf_config = DistilBertConfig(n_layers=self.clf_layers, n_heads=self.clf_heads)
        self.clf = DistilBertForSequenceClassification(self.clf_config)
        self.rnn = torch.nn.LSTM(input_size=768, hidden_size=64, num_layers=1, batch_first=True)
        self.linear = nn.Linear(64,2)
 
    def forward(self,x):
        self.tokens = self.tokenizer(list(x), return_tensors='pt', padding=self.padding, truncation=self.truncation, max_length=self.max_length)
        # self.tokens.to(self.device)
        for key in self.tokens.keys():
            self.tokens[key] = self.tokens[key].to(self.device)
        self.embedding = self.embedder(**self.tokens)
        o,(h,c) = self.rnn(self.embedding.last_hidden_state)
        h = h.squeeze(0)
        out = self.linear(h)
        return torch.sigmoid(out)

class MyClassifierEmbed(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_size=768, hidden_size=64, num_layers=1, batch_first=True)
        self.linear = nn.Linear(64,2)

    def forward(self,x):
        o,(h,c) = self.rnn(x)
        h = h.squeeze(0)
        out = self.linear(h)
        return torch.sigmoid(out)

class MyClassifierEmbedTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.clf_layers = 2
        self.clf_heads = 4
        self.clf_config = DistilBertConfig(n_layers=self.clf_layers, n_heads=self.clf_heads)
        self.clf = DistilBertForSequenceClassification(self.clf_config)
    def forward(self,embedding):
        out = self.clf(inputs_embeds=embedding)
        return out.logits

