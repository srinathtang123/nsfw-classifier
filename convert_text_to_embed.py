import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import ProfilerActivity, record_function, profile
from transformers import (AutoTokenizer,
    AutoModel,
    DistilBertForSequenceClassification,
    DistilBertConfig
    )

import pandas as pd
from prettytable import PrettyTable
from tqdm import tqdm
import sys

from classifier import CreateDataset, split_train_val_dataloader
dataset = CreateDataset('data/train/bq-results-20210825-203004-swh711l21gv2.csv')
train_loader,val_loader = split_train_val_dataloader(dataset,train_batch_size=16,val_batch_size=8)
model_name = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
embedder = AutoModel.from_pretrained(model_name)
device = torch.device('cuda:0')
embedder.to(device)
cnt = 0
loop = tqdm(train_loader)
for x,y in loop:
    tokens = tokenizer(list(x), return_tensors='pt', padding='max_length', truncation=True, max_length=42)
    for key in tokens.keys():
        tokens[key] = tokens[key].to(device)
    embedding = embedder(**tokens)
    torch.save({
            'embeddings':embedding.last_hidden_state,
            'label':y
        }, f'embeddings/train/{cnt}.pth.tar'
        )
    cnt+=1
loop = tqdm(val_loader)
cnt = 0
for x,y in loop:
    tokens = tokenizer(list(x), return_tensors='pt', padding='max_length', truncation=True, max_length=42)
    for key in tokens.keys():
        tokens[key] = tokens[key].to(device)
    embedding = embedder(**tokens)
    torch.save({
            'embeddings':embedding.last_hidden_state,
            'label':y
        }, f'embeddings/val/{cnt}.pth.tar'
        )
    cnt+=1