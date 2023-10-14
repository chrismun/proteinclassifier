#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import profiler
from data import ModelNet40
from model import PointNet, DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import StepLR

def collate_fn(batch):
    point_clouds, labels = zip(*batch)

    # convert point clouds and labels to tensors
    point_clouds = [torch.Tensor(pc) for pc in point_clouds]
    labels = torch.LongTensor(labels)

    # zero-pad the point clouds in the batch
    point_clouds_padded = pad_sequence(point_clouds, batch_first=True, padding_value=0.0)

    return point_clouds_padded, labels

def collate_fn_onehot(batch):
    point_clouds, labels = zip(*batch)

    # convert labels to tensors
    labels = torch.LongTensor(labels)

    # separate coordinates and features
    coordinates = [torch.Tensor(pc[:, :3]) for pc in point_clouds]
    features = [torch.Tensor(pc[:, 3:]) for pc in point_clouds]

    # pad coordinates with zeros in channels 1 to 3
    coordinates_padded = pad_sequence(coordinates, batch_first=True, padding_value=0.0)

    # pad features with zeros in channels 1 to 3 and a zero vector of length 21 in channel 4
    zero_vector = torch.zeros((len(features), features[0].size(0), 21))
    features_padded = torch.cat([torch.cat([f, zero_vector], dim=1) for f in features], dim=0)

    # concatenate coordinates and features along the second axis
    point_clouds_padded = torch.cat([coordinates_padded, features_padded], dim=2)

    return point_clouds_padded, labels



# Hyperparameters
rand_jitter = True
jitter_amount = 0.01

def normalize(point_data):
    # z-score normalization
    mean = np.mean(point_data, axis=0)
    std = np.std(point_data, axis=0)
    point_data_normalized = (point_data - mean) / std

    # min-max normalization (scale to range 0-1)
    # min_val = np.min(point_data, axis=0)
    # max_val = np.max(point_data, axis=0)
    # point_data_normalized = (point_data - min_val) / (max_val - min_val)

    return point_data_normalized


# My Dataset 
class ProteinDataset(Dataset):

    def __init__(self, fn):
        # data loading
        self.data = np.load(fn, 'rb', allow_pickle=True)
        self.x = self.data['x']
        self.y = np.reshape(self.data['y'], (-1,))
        self.num_classes = np.unique(self.y).shape[0]
        self.nsamples = self.x.shape[0]

        # self.x, self.y = zip(*[(x_i, y_i) for x_i, y_i in zip(self.x, self.y) if len(x_i) <= 2048 and len(x_i) >= 128])
        # print("maxlen: 8k")
        # self.x = list(self.x)
        # self.y = list(self.y)
        # self.num_classes = np.unique(self.y).shape[0]
        # self.nsamples = len(self.x)
        print("nsamples: ", self.nsamples)
        print("num_classes: ", self.num_classes)
   
    def __getitem__(self, index):
        point_data = np.array(self.x[index])

        # Normalize point data
        # point_data[:,:3] = normalize(point_data[:,:3])
        point_data = normalize(point_data)
        # # jitter
        # if rand_jitter:
        #     jitter = np.random.normal(0, jitter_amount, size=point_data[:,:3].shape)
        #     point_data[:,:3] += jitter
         # jitter
        if rand_jitter:
            for i in range(len(point_data)):
                # Generate a random jitter of appropriate size for this row
                jitter = np.random.normal(0, jitter_amount, size=point_data[i,:3].shape)
                
                # Add the jitter to the row
                point_data[i,:3] += jitter

        return point_data, self.y[index]





    def __len__(self):
        return self.nsamples

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')

def train(args, io):
    data_path = '../../data/downsampled/downsampled-5000x4.npz'
    print("data path: ", data_path)
    dataset = ProteinDataset(data_path)
    print("length: ", dataset.nsamples)
    train_dataset, test_dataset = train_test_split(dataset, test_size=0.20, random_state=17)

    # Weighted sampler
    train_class_counts = {}
    for _, label in train_dataset:
        if label in train_class_counts:
            train_class_counts[label] += 1
        else:
            train_class_counts[label] = 1

    train_samples_weight = np.array([1 / train_class_counts[y] for _, y in train_dataset])
    train_samples_weight = torch.tensor(train_samples_weight)
    train_sampler = WeightedRandomSampler(train_samples_weight.type('torch.DoubleTensor'), len(train_samples_weight))
    batch_size = 64
    print(f"batch size: {batch_size}")

    # Dataloaders 
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8, sampler=train_sampler, drop_last=True, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=False, collate_fn=collate_fn)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=8, sampler=train_sampler, drop_last=True)
    # test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)

    device = torch.device("cuda" if args.cuda else "cpu")

    # Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device, dtype=torch.float)
    elif args.model == 'dgcnn':
        model = DGCNN(args).to(device, dtype=torch.float)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model, [0,1,2,3])
    # model = nn.DataParallel(model)
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=.1, momentum=args.momentum, weight_decay=1e-3)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=.1, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=.001)

    #step scheduler
    # scheduler = StepLR(opt, step_size=150, gamma=0.1)

    # criterion = nn.CrossEntropyLoss()
    
    criterion = cal_loss

    best_test_acc = 0
    epoch_test_acc = []
    epoch_test_loss = []
    epoch_train_acc = []
    epoch_train_loss = []

    for epoch in range(args.epochs):
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        total_correct = 0
        total_testset = 0
        for data, label in train_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            # with profiler.profile(record_shapes=True, profile_memory=True, use_cuda=True) as prof:
            #     with profiler.record_function("forward pass"):
            #         logits = model(data)
            #     loss = criterion(logits, label)
            #     loss.backward()
            #     opt.step()
            #     preds = logits.max(dim=1)[1]

            
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]

            correct = preds.eq(label.data).cpu().sum()
            total_correct += correct.item()
            total_testset += data.size()[0]


            count += batch_size
            train_loss += loss.item() * batch_size
        train_acc = total_correct / float(total_testset)
        epoch_train_acc.append(train_acc)
        epoch_train_loss.append(train_loss*1.0/count)
        outstr = 'Train %d, loss: %.6f, train acc: %.6f' % (epoch, train_loss*1.0/count, train_acc)
        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
        io.cprint(outstr)

        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        total_correct = 0
        total_testset = 0
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]

            correct = preds.eq(label.data).cpu().sum()
            total_correct += correct.item()
            total_testset += data.size()[0]

            count += batch_size
            test_loss += loss.item() * batch_size
        test_acc = total_correct / float(total_testset)
        outstr = 'Test %d, loss: %.6f, test acc: %.6f' % (epoch, test_loss*1.0/count, test_acc)
        epoch_test_acc.append(test_acc)
        epoch_test_loss.append(test_loss*1.0/count)
        io.cprint(outstr)
        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
            print("NEW BEST TEST ACCURACY: ", best_test_acc)
    print("Epoch test acc: ", epoch_test_acc)
    print("Epoch train acc: ", epoch_train_acc)
    print("Epoch test loss: ", epoch_test_loss)
    print("Epoch train loss: ", epoch_train_loss)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=100, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=8, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    train(args, io)