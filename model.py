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

# wandb.login()

# Hyper-parameters 
num_epochs = 200
batch_size = 64
learning_rate = 0.03
rand_rotate = False
rand_jitter = True
feature_trans = False
lr_step_size = 50
gamma = .1
optimizer_fn = "adam"
loss_fn = "crossEntropy"

#min_samples = 100
#npoints = 2048
#subset = False

# sweep_config = {
#     'method': 'random',
#     'metric': {
#         'name': 'val_accuracy',
#         'goal': 'maximize'
#     },
#     'parameters': {
#         'batch_size': {
#             'values': [4,32]
#         },
#         'learning_rate': {
#             'values': [0.001,0.01,0.0001]
#         },
#         'num_epochs': {
#             'values': [10, 20] # number of epoch to run after resume epoch
#         },
#         'npoints': {
#             'values':[2500, 5000]
#         }
#     }
# }

# sweep_id= wandb.sweep(sweep_config,project="proteinclassifier")

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
        #  print(preds)

    return total_correct / float(total_testset), f1_score(targets, preds, average='weighted')
    # wandb.agent(sweep_id,train,count=3)

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
        
        # random rotation
        if rand_rotate:
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
        # jitter
        if rand_jitter:
            jitter = np.random.normal(0,0.02,size=point_data.shape)
            jitter[:,[0,1,2]] = 0
            point_data += jitter
        return point_data, one_hot(self.y[index], self.num_classes), self.y[index]

    def __len__(self):
        return self.nsamples

blue = lambda x: '\033[94m' + x + '\033[0m' 

# Dataset
dataset = ProteinDataset('protein-data.npz')
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Weighted sampler 
weights_dict = {4: 333, 8: 92, 3: 825, 7: 115, 5: 234, 2: 1600, 0: 148, 6: 206, 1: 371} # comes from preprocess
weight = [1 / value for _, value in sorted(weights_dict.items())]
samples_weight = np.array([weight[y] for _, _, y in train_dataset])
samples_weight = torch.tensor(samples_weight)
sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

# Dataloaders 
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8, sampler=sampler, drop_last=True) # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)

print(f'length of train: {len(train_dataset)}, length of test: {len(test_dataset)}')
print("number of classes = ", dataset.num_classes)

# Create model, optimizer, loss, scheduler
classifier = PointNetCls(k=dataset.num_classes, feature_transform=feature_trans).cuda()
if optimizer_fn == "sgd":
    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate, momentum=0.9)
elif optimizer_fn == "adam":
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0.0)

if loss_fn == "crossEntropy":
    criterion = nn.CrossEntropyLoss(label_smoothing=0.6).cuda()
elif loss_fn == "SmoothL1":
    criterion = nn.SmoothL1Loss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

# wandb.watch(classifier)
classifier = nn.DataParallel(classifier, [0, 1, 2, 3])
classifier.train()

accuracy_log = {}

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
        if feature_trans:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print('[%d: %d/%d] train loss: %.2f' % (epoch, i, num_batch, loss.item()))

        if i % 20 == 0:
            continue
            #accuracy, f1 = test(test_loader, classifier)
            #taccuracy, tf1 = test(train_loader, classifier)
            # wandb.log({"loss": loss.item(), "accuracy": correct.item()/float(batch_size)})
            #print('[%d: %d/%d] %s: train accuracy: %.5f, f1: %.5f' % (epoch, i, num_batch, blue('test'), taccuracy, tf1))
            #print('[%d: %d/%d] %s: val accuracy: %.5f, f1: %.5f' % (epoch, i, num_batch, blue('test'), accuracy, f1))
    scheduler.step()
    accuracy, f1 = test(test_loader, classifier)
    taccuracy, tf1 = test(train_loader, classifier)
    print('[%d: %d/%d] %s: val accuracy: %.5f, f1: %.5f' % (epoch, i, num_batch, blue('test'), accuracy, f1))
    print('[%d: %d/%d] %s: train accuracy: %.5f, f1: %.5f' % (epoch, i, num_batch, blue('test'), taccuracy, tf1))
    if epoch % 10 == 9:
        accuracy_log[epoch] = "val acc: {:.2f}, train acc: {:.2f}".format(accuracy, taccuracy)

    # torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

classifier.eval()
accuracy, f1 = test(test_loader, classifier)
taccuracy, tf1 = test(train_loader, classifier)
print('[%d: %d/%d] %s: val accuracy: %.5f, f1: %.5f' % (epoch, i, num_batch, blue('test'), accuracy, f1))
print('[%d: %d/%d] %s: train accuracy: %.5f, f1: %.5f' % (epoch, i, num_batch, blue('test'), taccuracy, tf1))
print(f'length of train: {len(train_dataset)}, length of test: {len(test_dataset)}')
print("number of classes = ", dataset.num_classes)

print("Hyperparameters:")
print("Number of epochs:", num_epochs)
print("Batch size:", batch_size)
print("Learning rate:", learning_rate)
print("Random rotation:", rand_rotate)
print("Random jitter:", rand_jitter)
print("Feature transformation:", feature_trans)
print("Learning rate step size:", lr_step_size)
print("Gamma:", gamma)
print("Optimizer fn:", optimizer_fn)
print("Loss fn:", loss_fn)

for k,v in accuracy_log.items():
    print("epoch ", k, ":", v)