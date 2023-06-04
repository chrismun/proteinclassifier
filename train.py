from __future__ import print_function
import torch
#import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pyuul import utils
from sklearn.model_selection import train_test_split
import os
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
# import wandb
from sklearn.metrics import f1_score

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
num_epochs = 100
batch_size = 50
learning_rate = 0.01
feature_transform = True

# wandb.login()
# sweep_config = {}
# sweep_id= wandb.sweep(sweep_config,project="proteinclassifier")

# I think this function, one_hot, is no longer used.
def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def test(dataloader, classifier):
    total_correct = 0
    total_testset = 0
    preds = []
    targets = []

    with torch.no_grad():
        for _ ,data in tqdm(enumerate(dataloader, 0)):
            points, _, target = data
            points = points.permute(0, 2, 1).float()
            points, target = points.cuda(), target.cuda()
            pred, _, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]
            
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_testset += points.size()[0]
            # wandb.log(
            #     "test set acc": (total_correct/float(total_testset))
            # )
            
            preds.append(pred_choice.cpu())
            targets.append(target.cpu().data)
        
        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()

    return total_correct / float(total_testset), f1_score(targets, preds, average='weighted')

def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

# Dataset 
class ProteinDataset(Dataset):

    def __init__(self, fn):
        # data loading
        self.data = np.load(fn, 'rb')
        self.x = self.data['x']
        self.y = np.reshape(self.data['y'], (-1,))
        self.n_points = self.x.shape[1]

        self.num_classes = np.unique(self.y).shape[0]
        self.nsamples = self.x.shape[0]
        # self.x = torch.from_numpy(self.x)
        # self.y = torch.from_numpy(self.y)

    def __getitem__(self, index):
        point_data = self.x[index]
        # point_data = shuffle_along_axis(point_data, axis=1)
        theta1 = np.random.uniform(0,np.pi*2)
        theta2 = np.random.uniform(0,np.pi*2)
        theta3 = np.random.uniform(0,np.pi*2)
        trans_matrix1 = np.array([[np.cos(theta1),-np.sin(theta1)],
                                    [np.sin(theta1), np.cos(theta1)]])
        trans_matrix2 = np.array([[np.cos(theta2), -np.sin(theta2)],
                                    [np.sin(theta2), np.cos(theta2)]])
        trans_matrix3 = np.array([[np.cos(theta3),-np.sin(theta3)],
                                    [np.sin(theta3), np.cos(theta3)]])
        point_data[:,[0,1]] = point_data[:,[0,1]].dot(trans_matrix1)
        point_data[:,[0,2]] = point_data[:,[0,2]].dot(trans_matrix2)
        point_data[:,[1,2]] = point_data[:,[1,2]].dot(trans_matrix3)
        return point_data, one_hot(self.y[index], self.num_classes), self.y[index]

    def __len__(self):
        return self.nsamples

blue = lambda x: '\033[94m' + x + '\033[0m' 

# Dataset
dataset = ProteinDataset('protein-data2.npz')
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Weighted sampler
weight = [1./128, 1./269, 1./790, 1./476, 1./233, 1./168, 1./50, 1./26, 1./46]
samples_weight = np.array([weight[y] for _, _, y in train_dataset])
samples_weight = torch.tensor(samples_weight)
sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

# Dataloaders 
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8, sampler=sampler, drop_last=True) # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)

print(f'length of train: {len(train_dataset)}, length of test: {len(test_dataset)}')
print("number of classes = ", dataset.num_classes)

# Create model, optimizer, loss, scheduler
classifier = PointNetCls(k=dataset.num_classes, feature_transform=True).cuda()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.0)
# criterion = nn.CrossEntropyLoss(label_smoothing=0.6).cuda()
criterion = nn.SmoothL1Loss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

# wandb.watch(classifier)
classifier = nn.DataParallel(classifier, [0, 1, 2, 3])
classifier.train()

# Training Loop
num_batch = len(train_dataset) / batch_size

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        points, target, _ = data
        points = points.permute(0, 2, 1).float()
        points, target = points.cuda(), target.cuda()
        pred, trans, trans_feat = classifier(points)

        optimizer.zero_grad()
        loss = criterion(pred, target)
        # loss = F.nll_loss(pred, target)
        if feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print('[%d: %d/%d] train loss: %.2f' % (epoch, i, num_batch, loss.item()))

        if i % 20 == 0:
            accuracy, f1 = test(test_loader, classifier)
            # taccuracy, tf1 = test(train_loader, classifier)
            # wandb.log({"loss": loss.item(), "accuracy": correct.item()/float(batch_size)})
            # print('[%d: %d/%d] %s: train accuracy: %.5f, f1: %.5f' % (epoch, i, num_batch, blue('test'), taccuracy, tf1))
            print('[%d: %d/%d] %s: val accuracy: %.5f, f1: %.5f' % (epoch, i, num_batch, blue('test'), accuracy, f1))
    scheduler.step()
    # torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

classifier.eval()
accuracy, f1 = test(test_loader, classifier)
print('Test: accuracy: %.5f, f1: %.5f' % (accuracy, f1))

# wandb.agent(sweep_id,train,count=3)