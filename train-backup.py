from __future__ import print_function
import torch
#import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import os
from pyuul import utils
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import h5py
import wandb

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# wandb.login()

# Hyper-parameters 
num_epochs = 10
batch_size = 8
learning_rate = 0.001
min_samples = 100
npoints = 2048
subset = True

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


# Dataset 
class ProteinDataset(Dataset):

    def __init__(self):
        # data loading
        full_protein_list = pd.read_csv("./data/proteins.csv")
        transMemProteins = full_protein_list[full_protein_list['type_id'] == 1]
        if subset:
            choice = np.random.choice(len(transMemProteins), 1000, replace=True)
            transMemProteins = transMemProteins.iloc[choice, :]
        transMemProteins['pdbid'] = transMemProteins['pdbid'].str.replace('[^\w]', '', regex=True)  # remove "=...." extra characters
        counts = transMemProteins['membrane_name_cache'].value_counts()
        label_dict_counts = {key: counts[key] for key in counts.index}
        selected_list = [key for key, num_samples in label_dict_counts.items() if num_samples > min_samples]
        k = len(selected_list)
        transMemProteins = transMemProteins[transMemProteins['membrane_name_cache'].isin(selected_list)]
        labels = transMemProteins['membrane_name_cache'].unique()
        label_dict = {key: value for value, key in enumerate(sorted(labels))}
        labels = list(label_dict.keys())
        # self.label_dict_counts = label_dict_counts
        # self.num_classes = len(labels)
        # self.counts = counts

        x = []
        y = []
        for i, pdb_id in enumerate(transMemProteins['pdbid']):
            file_name = pdb_id + '.pdb'
            file_path = os.path.join('./data/pdb', file_name)

            if os.path.exists(file_path):
                coords, atname = utils.parsePDB(file_path, keep_hetatm=False)
                radius = utils.atomlistToRadius(atname)
                atomType = utils.atomlistToChannels(atname)

                radius = radius.tolist()[0]
                atomType = atomType.tolist()[0]

                file_name = pdb_id + '.pdb'
                file_path = os.path.join('./data/pdb', file_name)

                coords, atname = utils.parsePDB(file_path, keep_hetatm=False)
                radius = utils.atomlistToRadius(atname)
                atomType = utils.atomlistToChannels(atname)

                radius = radius.tolist()[0]
                atomType = atomType.tolist()[0]

                coords = coords.tolist()
                coords = coords[0]
                i = 0

                for coord in coords:
                    coord.append(radius[i])
                    coord.append(atomType[i])

                a = transMemProteins.loc[transMemProteins['pdbid'] == pdb_id, 'membrane_name_cache'].iloc[0]

                x.append(coords)
                y.append([label_dict[a]])
            else:
                print(f"The file {file_path} does not exist.")

        # fixing varying size
        max_len = max(len(entry) for entry in x)
        # Pad shorter entries with all-zero entries
        for i, entry in enumerate(x):
            if len(entry) < max_len:
                num_padding = max_len - len(entry)
                for j in range(0, num_padding):
                    x[i].append(x[i][0])
        
        class_count = {}
        j = 0
        for prot in x:
            class_count[y[j][0]] = class_count.setdefault(y[j][0], 0) + 1
            j += 1

        for k, v in class_count.items():
            print("Label:", k, "count:", v)
        print("LENCLASSCT:", len(class_count))


        self.class_count = class_count
        self.num_classes = len(class_count)
        self.nsamples = len(x)
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __getitem__(self, index):
        point_set = self.x[index]
        choice = np.random.choice(len(point_set), npoints, replace=True)
        point_set = point_set[choice, :]
        return point_set, self.y[index]

    def __len__(self):
        return self.nsamples

blue = lambda x: '\033[94m' + x + '\033[0m'

# Instantiate data, and split
dataset = ProteinDataset()
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)


print("number of classes = ", dataset.num_classes)

label_weight = { l: 1/v for l,v in dataset.class_count.items() }
sample_weights = []
for sample in train_dataset:
    sample_weights.append(label_weight[sample[1][0].item()])
# print(sample_weights)
sample_weights = torch.tensor(sample_weights)

# sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight), replacement=True)
sampler = WeightedRandomSampler(sample_weights.type('torch.DoubleTensor'), len(dataset), replacement=True)


# Create the dataloaders for training and test sets
# weighted sampler data loader - not working properly 
# train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=1, sampler=sampler, drop_last=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=1, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True)

print(f'length of train: {len(train_dataset)}, length of test: {len(test_dataset)}')

# Create model, optimizer, scheduler
classifier = PointNetCls(k=dataset.num_classes, feature_transform=True)
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
classifier.cuda()
# wandb.watch(classifier)
num_batch = len(train_dataset) / batch_size

# Training Loop
for epoch in range(num_epochs):
    for i, data in enumerate(train_loader, 0):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, trans, trans_feat = classifier(points)
        loss = F.nll_loss(pred, target)
        if True:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(batch_size)))

        if i % 10 == 0:
            j, data = next(enumerate(test_loader, 0))
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            # wandb.log({"loss": loss.item(), "accuracy": correct.item()/float(batch_size)})
            print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(batch_size)))
    scheduler.step()
    # torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

total_correct = 0
total_testset = 0
for i,data in tqdm(enumerate(test_loader, 0)):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    classifier = classifier.eval()
    pred, _, _ = classifier(points)
    pred_choice = pred.data.max(1)[1]
    correct = pred_choice.eq(target.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]
    # wandb.log(
    #     "test set acc": (total_correct/float(total_testset))
    # )

print("final accuracy {}".format(total_correct / float(total_testset)))
# wandb.agent(sweep_id,train,count=3)