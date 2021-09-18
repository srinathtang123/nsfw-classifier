import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from tqdm import tqdm
import sys
from torch.utils.data import DataLoader
from classifier_models import MyClassifier
from dataloader_helper import CreateDataset, split_train_val_dataloader, CreateDataset_test

class Trainer(object):
    def __init__(self):    
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_path = 'data/train/bq-results-20210825-203004-swh711l21gv2.csv'
        self.ratio=0.7 
        self.train_batch_size=32
        self.val_batch_size=8
        self.epochs = 10
        self.lr = 0.001
        self.log_dir='results/exp1/'
        self.model_save_path = 'results/exp1/'

        self.model = MyClassifier(self.device).to(self.device)
        #freze embeeder and tokenizer
        for params in self.model.embedder.parameters():
            params.requires_grad = False
        
        # count_parameters(model)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.dataset = CreateDataset(data_path=self.data_path)
        self.train_loader,self.val_loader = split_train_val_dataloader(dataset=self.dataset, ratio=self.ratio,\
                                                     train_batch_size=self.train_batch_size, val_batch_size=self.val_batch_size)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.train_step = 0
        self.val_step = 0
        # self.model.to(self.device)

    def train(self,x,y):
        x = x.squeeze(0)
        y = y.squeeze(0)
        # x = x.to(self.device)
        y = y.to(self.device)
        y_hat = self.model(x)
        loss = self.criterion(y_hat,y.long())   
        self.train_loss_epoch+=float(loss)
        self.writer.add_scalar("batch_loss/train", loss, global_step=self.train_step)
        self.train_step+=1         
        #backward
        self.optimizer.zero_grad()
        loss.backward()
        #step
        self.optimizer.step()
        return loss
    
    def val(self,x,y):
        x = x.squeeze(0)
        y = y.squeeze(0)
        # x = x.to(self.device)
        y = y.to(self.device)
        y_hat = self.model(x)
        loss = self.criterion(y_hat,y.long())
        self.val_loss_epoc+=float(loss)
        self.writer.add_scalar("batch_loss/val", loss, global_step=self.val_step)
        self.val_step+=1 
        return loss


    def train_loop(self):
        self.model.to(self.device)
        for epoch in range(self.epochs):
            #train loop
            self.model.train()
            self.train_loss_epoch = 0
            cnt = 0
            loop = tqdm(self.train_loader)
            for x,y in loop:
                loss = self.train(x,y)
                loop.set_description(f'train [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=float(loss))
                cnt+=1
                # break
            self.train_loss_epoch/=cnt
            self.writer.add_scalar("epoch_loss/train", self.train_loss_epoch, global_step=epoch)
            # val loop
            self.model.eval()
            self.val_loss_epoc = 0
            cnt = 0
            loop = tqdm(self.val_loader)
            for x,y in loop:
                loss = self.val(x,y)
                loop.set_description(f'val [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=float(loss))
                cnt+=1
                # break
            self.val_loss_epoc/=cnt
            self.writer.add_scalar("epoch_loss/val", self.val_loss_epoc, global_step=epoch)
            #models saving
            ckpt = {'epoch':epoch, 'state_dict':self.model.state_dict(), 'optimizer':self.optimizer.state_dict(), 'loss':loss }
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
        self.model.load_state_dict(self.state_dict)
        self.labels_pred = []
        self.Ids = []
        self.model.eval()
        loop = tqdm(self.test_loader)
        for x,ids_batch in loop:
            y_hat = self.model(x)
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

