from math import log
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

class MyClassifier(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device = device
        self.model_name = "ai4bharat/indic-bert"   # name of the model for tokenizer and embedding model
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

    # def forward(self,x):
    #     self.tokens = self.tokenizer(x, return_tensors='pt', padding=self.padding, truncation=self.truncation, max_length=self.max_length)
    #     # self.tokens.to(self.device)
    #     for key in self.tokens.keys():
    #         self.tokens[key] = self.tokens[key].to(device)
    #     self.embedding = self.embedder(**self.tokens)
    #     self.out = self.clf(inputs_embeds=self.embedding.last_hidden_state)
    #     return self.out.logits
    
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
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=True)
    return train_loader,val_loader


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

class Trainer(object):
    def __init__(self):    
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_path = 'data/train/bq-results-20210825-203004-swh711l21gv2.csv'
        self.ratio=0.7 
        self.train_batch_size=64
        self.val_batch_size=64
        self.epochs = 1
        self.lr = 0.001
        self.log_dir='results/exp1/'
        self.model_save_path = 'results/exp1/'
        self.clf = MyClassifier(self.device).to(self.device)
        #freze embeeder and tokenizer
        for params in self.clf.embedder.parameters():
            params.requires_grad = False
        
        # count_parameters(clf)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.clf.parameters(), lr=self.lr)
        self.dataset = CreateDataset(data_path=self.data_path)
        self.train_loader,self.val_loader = split_train_val_dataloader(dataset=self.dataset, ratio=self.ratio,\
                                                     train_batch_size=self.train_batch_size, val_batch_size=self.val_batch_size)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.train_step = 0
        self.val_step = 0
    def train_loop(self):
        for epoch in range(self.epochs):
            self.clf.train()
            self.train_loss_epoch = 0
            cnt = 0
            loop = tqdm(self.train_loader)
            for x,y in loop:
                y = y.to(self.device)
                y_hat = self.clf(x)
                loss = self.criterion(y_hat,y.long())   
                self.train_loss_epoch+=loss
                cnt+=1
                self.writer.add_scalar("batch_loss/train", loss, global_step=self.train_step)
                self.train_step+=1         
                #backward
                self.optimizer.zero_grad()
                loss.backward()
                #step
                self.optimizer.step()
                loop.set_description(f'train [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss)
                break
            self.train_loss_epoch/=cnt
            self.writer.add_scalar("epoch_loss/train", self.train_loss_epoch, global_step=epoch)
            self.clf.eval()
            self.val_loss_epoc = 0
            cnt = 0
            loop = tqdm(self.val_loader)
            for x,y in loop:
                y = y.to(self.device)
                y_hat = self.clf(x)
                loss = self.criterion(y_hat,y.long())
                self.val_loss_epoc+=loss
                cnt+=1
                self.writer.add_scalar("batch_loss/val", loss, global_step=self.val_step)
                self.val_step+=1 
                loop.set_description(f'val [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss)
                break
            self.val_loss_epoc/=cnt
            self.writer.add_scalar("epoch_loss/val", self.val_loss_epoc, global_step=epoch)
            ckpt = {'epoch':epoch, 'state_dict':self.clf.state_dict(), 'optimizer':self.optimizer.state_dict(), 'loss':loss }
            torch.save(ckpt, self.model_save_path+f'epoch:{epoch}-val_loss:{self.val_loss_epoc}.pth.tar')
            self.writer.add_hparams(
                    { "lr":self.lr, "batch_size":self.train_batch_size},
                    { "train_loss":self.train_loss_epoch, "val_loss":self.val_loss_epoc },
            )
        self.writer.close()
    def test_loop(self,model_load_path):
        self.test_data_path = 'data/test/eam2021-test-set-public.csv'
        self.model_load_path = model_load_path
        self.test_batch_size = 32
        # self.test_data = pd.read_csv(self.test_data_path)
        self.test_set = CreateDataset_test(self.test_data_path)
        self.test_loader = DataLoader(self.test_set, batch_size=self.test_batch_size)
        self.state_dict = torch.load(self.model_load_path)['state_dict']
        self.clf.load_state_dict(self.state_dict)
        self.labels_pred = []
        self.Ids = []
        self.clf.eval()
        loop = tqdm(self.test_loader)
        for x,ids_batch in loop:
            y_hat = self.clf(x)
            labels_pred_batch = torch.argmax(y_hat, dim=1)
            # self.labels_pred.append(labels_pred_batch.detach())
            for elm,id in zip(labels_pred_batch.cpu().numpy(), ids_batch.numpy()):
                self.labels_pred.append(elm)
                self.Ids.append(id)
            loop.set_description(f'test')
            # break
        sub = pd.DataFrame()
        sub['Id'] = self.Ids
        sub['Expected'] = self.labels_pred
        sub.to_csv('submission.csv',index=False)


if __name__=='__main__':
    trainer = Trainer()
    # trainer.train_loop()
    trainer.test_loop('results/exp1/epoch:0-val_loss:0.6944313645362854.pth.tar')
    # with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         trainer.train_loop()
    # print(prof.key_averages().table(sort_by="cuda_time_total"))




