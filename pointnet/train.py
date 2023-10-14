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
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics

# tensorboard 
write_to_tensorboard = False
if(write_to_tensorboard):
    writer = SummaryWriter()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyper-parameters 
num_epochs = 300
batch_size = 64
learning_rate = 0.003
rand_rotate = False
rand_jitter = True
jitter_amount = .02
feature_trans = False
lr_step_size = 150
gamma = .1
optimizer_fn = "adam"
loss_fn = "crossEntropy"
weight_decay = 0.00001 # L2 Regularization 

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def test(dataloader, classifier, print_metrics, criterion):
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
            loss = criterion(pred, target)
            correct = pred_choice.eq(target.data).cpu().sum()
            total_correct += correct.item()
            total_testset += points.size()[0]
            preds.append(pred_choice.cpu())
            targets.append(target.cpu().data)
        
        preds = torch.cat(preds).numpy()
        targets = torch.cat(targets).numpy()
        if print_metrics:
            cm = metrics.confusion_matrix(targets, preds)
            print(cm)
            print(metrics.classification_report(targets, preds))
        #  print(preds)

    return total_correct / float(total_testset), f1_score(targets, preds, average='weighted'), loss

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

        #self.num_classes = self.y.shape[0]
        self.num_classes = np.unique(self.y).shape[0]
        self.nsamples = self.x.shape[0]

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
            jitter = np.random.normal(0,jitter_amount,size=point_data.shape)
            #jitter[:,[0,1,2]] = 0
            point_data += jitter
        return point_data, one_hot(self.y[index], self.num_classes), self.y[index]

    def __len__(self):
        return self.nsamples

blue = lambda x: '\033[94m' + x + '\033[0m' 

# Dataset
dataset = ProteinDataset('protein-data.npz')
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Weighted sampler
# print(len(train_dataset[0]))
# print(train_dataset[:3][:3][:3])
# print(train_dataset[1])

# train_weights_dict = {}
# j = 0
# for prot in train_dataset[0]:
#     train_weights_dict[train_dataset[1][j][0]] = train_weights_dict.setdefault(train_dataset[1][j][0], 0) + 1
#     j += 1

# sorted_dict = dict(sorted(class_count.items(), key=lambda x: int(x[0])))

# print("print(class_count): ", class_count)
#train_weights_dict = {8: 333, 23: 92, 18: 75, 10: 1, 7: 825, 19: 115, 9: 234, 5: 1600, 0: 148, 13: 206, 3: 371, 14: 56, 17: 4, 21: 35, 6: 29, 22: 5, 20: 13, 11: 1, 4: 16, 12: 38, 1: 4, 15: 1, 16: 7}
old_weights_dict = {4: 333, 8: 92, 3: 825, 7: 115, 5: 234, 2: 1600, 0: 148, 6: 206, 1: 371} # comes from preprocess
weight = [1 / value for _, value in sorted(old_weights_dict.items())]
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
    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
elif optimizer_fn == "adam":
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)

if loss_fn == "crossEntropy":
    criterion = nn.CrossEntropyLoss(label_smoothing=0.6).cuda()
elif loss_fn == "SmoothL1":
    criterion = nn.SmoothL1Loss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=gamma)

classifier = nn.DataParallel(classifier, [0, 1, 2, 3])
# classifier = nn.DataParallel(classifier, [0])
classifier.train()

accuracy_log = {}

# Training Loop
num_batch = len(train_dataset) / batch_size

for epoch in range(num_epochs):
    for i, data in enumerate(train_loader):
        points, _, target = data
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

        if i % 10 == 0:
            print('[%d: %d/%d] train loss: %.2f' % (epoch, i, num_batch, loss.item()))
            if(write_to_tensorboard):
                writer.add_scalar("Loss/train", loss.item(), ((epoch - 1)*num_batch + i))
                _, _, val_loss = test(train_loader, classifier, False, criterion)
                writer.add_scalar("Loss/val", val_loss.item(), ((epoch - 1)*num_batch + i))


        if i % 20 == 0:
            continue
            #accuracy, f1 = test(test_loader, classifier)
            #taccuracy, tf1 = test(train_loader, classifier)
            # wandb.log({"loss": loss.item(), "accuracy": correct.item()/float(batch_size)})
            #print('[%d: %d/%d] %s: train accuracy: %.5f, f1: %.5f' % (epoch, i, num_batch, blue('test'), taccuracy, tf1))
            #print('[%d: %d/%d] %s: val accuracy: %.5f, f1: %.5f' % (epoch, i, num_batch, blue('test'), accuracy, f1))
    scheduler.step()
    accuracy, f1, _ = test(test_loader, classifier, False, criterion)
    taccuracy, tf1, _ = test(train_loader, classifier, False, criterion)
    print('[%d: %d/%d] %s: val accuracy: %.5f, f1: %.5f' % (epoch, i, num_batch, blue('test'), accuracy, f1))
    print('[%d: %d/%d] %s: train accuracy: %.5f, f1: %.5f' % (epoch, i, num_batch, blue('test'), taccuracy, tf1))
    if epoch % 10 == 9:
        accuracy_log[epoch] = "val acc: {:.2f}, train acc: {:.2f}".format(accuracy, taccuracy)

classifier.eval()
accuracy, f1, _ = test(test_loader, classifier, True, criterion)
taccuracy, tf1, _ = test(train_loader, classifier, True, criterion)
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
print("Jitter amount: ", jitter_amount)
print("Feature transformation:", feature_trans)
print("Learning rate step size:", lr_step_size)
print("Gamma:", gamma)
print("Optimizer fn:", optimizer_fn)
print("Loss fn:", loss_fn)
print('L2 weight decay: ', weight_decay)

for k,v in accuracy_log.items():
    print("epoch ", k, ":", v)
if(write_to_tensorboard):
    writer.flush() 