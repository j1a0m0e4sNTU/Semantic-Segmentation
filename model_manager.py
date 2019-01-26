import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from util import *

def get_string(*args):
    string = ''
    for s in args:
        string = string + ' ' + str(s)
    return string

class Manaeger():
    def __init__(self, model, args):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.lr = args.lr
        self.metric = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr= self.lr)
        self.epoch_num = args.epoch_num
        self.batch_size = args.batch_size
        self.save_name = args.save
        self.log_file = open(args.log, 'w')
        self.check_batch_num = args.check_batch_num
        self.best = (0, 0) #(epoch num, validation acc)

        load_name = args.load
        if load_name != None:
            weight = torch.load(load_name)
            self.model.load_state_dict(weight)
    
    def load_data(self, train_loader, valid_loader):
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def record(self, message):
        self.log_file.write(message)
        print(message)

    def get_info(self):
        info = get_string('Model:', self.model.name(), '\n')
        info = get_string(info, 'Learning rate:', self.lr, '\n')
        info = get_string(info, 'Epoch number:', self.epoch_num, '\n')
        info = get_string(info, 'Batch size:', self.batch_size, '\n')
        info = get_string(info, 'Weight name:', self.save_name, '\n')
        info = get_string(info, 'Log file:', self.log_file, '\n')
        info = get_string(info, '=======================\n\n')
        return info

    def train(self):
        info = self.get_info()
        self.record(info)
        
        for epoch in range(self.epoch_num):
            self.model.train()
            for batch_id, (imgs, labels) in enumerate(self.train_loader):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
    
                out = self.model(imgs)
                loss = self.metric(out, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (batch_id % self.check_batch_num == 0):
                    result = get_string('Epoch',epoch, '| batch', batch_id, '| Training loss :', loss.item(),'\n')
                    self.record(result)

            self.validate(epoch)
        
        info = get_string('\n# The best model is at epoch', self.best[0], 'with mean IOU', self.best[1])
        self.record(info)

    def validate(self, epoch):
        self.model.eval()
        loss_total = 0
        total_iou = 0
        
        for batch_id, (imgs, labels) in enumerate(self.valid_loader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            out = self.model(imgs)
            
            loss = self.metric(out, labels)
            loss_total += loss.item()
            
            pred = out.max(1)[1]
            total_iou = total_iou + mean_iou_batch(pred, labels)

        loss_avg = loss_total / (batch_id + 1)
        mean_iou = total_iou / ((batch_id + 1) * self.batch_size)
        line = '\n----------------------------\n'
        info = get_string('Validation result for ', epoch, 'epoch\n')
        info = get_string(info,'Average loss:', loss_avg, '\n Mean IOU:', mean_iou)
        info = get_string(line, info, line)
        self.record(info)

        if mean_iou > self.best[1]:
            self.best = (epoch, mean_iou)
            torch.save(self.model.state_dict(), self.save_name)
            self.record('***** Saved best model! *****\n')

if __name__ == '__main__':
    pass