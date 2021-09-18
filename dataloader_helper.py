import torch
from torch.utils.data import DataLoader, random_split, Dataset
import pandas as pd
import os

class CreateDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = pd.read_csv(data_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence = self.data.iloc[index,2]
        label = self.data.iloc[index,-2]
        return sentence,label

class CreateDataset_test(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = pd.read_csv(data_path)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sentence = self.data.iloc[index,3]
        label = self.data.iloc[index,0]
        return sentence,label

def split_train_val_dataloader(dataset, train_batch_size=32, val_batch_size=32, ratio=0.8):
    data_len = len(dataset)
    train_len = int(data_len*ratio)
    val_len = data_len-train_len
    train_set,val_set = random_split(dataset,[train_len,val_len])
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader,val_loader

class CreateDatasetEmbed(Dataset):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.length = len(os.listdir(f'embeddings/{mode}'))
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        data = torch.load(f'embeddings/{self.mode}/{index}.pth.tar')
        embeddings = data['embeddings']
        labels = data['label']
        return embeddings,labels 
