from trainer import Trainer
from torch.utils.data import DataLoader
from classifier_models import MyClassifierEmbed
from dataloader_helper import CreateDatasetEmbed

if __name__=='__main__':
    trainer = Trainer()
    trainer.model = MyClassifierEmbed()
    train_set = CreateDatasetEmbed('train')
    val_set = CreateDatasetEmbed('val')
    trainer.train_loader = DataLoader(train_set, batch_size=1)
    trainer.val_loader = DataLoader(val_set, batch_size=1)
    trainer.train_loop() 