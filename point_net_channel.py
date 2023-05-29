from torch.utils.data import Dataset,DataLoader
import torch
import pandas as pd
import os
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# input transform ; spatial transformation newtwork (now 5d)
class STN3d(nn.Module):
    """ First Trasformation applied on the input """
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(5, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 25)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)

        self.bn2 = nn.BatchNorm1d(128)

        self.bn3 = nn.BatchNorm1d(1024)

        self.bn4 = nn.BatchNorm1d(512)
        self.in4 = nn.InstanceNorm1d(512) # if batch size=1
        self.bn5 = nn.BatchNorm1d(256)
        self.in5 = nn.InstanceNorm1d(256)# if batch size=1


    def forward(self, x):
        batchsize = x.size()[0]
        # x=self.conv1(x)
        # x=self.bn1(self.conv1(x))
        x= F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1,1024)
        # x = self.fc1(x)

        x = F.relu(self.in4(self.fc1(x)))
        x = F.relu(self.in5(self.fc2(x)))
        x = self.fc3(x)

        iden =  Variable(torch.from_numpy(np.eye(5).flatten().astype(np.float32))).view(1,5*5).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 5, 5)


         ###  my
        # x = F.relu(self.bn1(self.conv1(x)))

        return x


class STNkd(nn.Module):
    """ Trasformation on the features vector"""
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.in4 = nn.InstanceNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.in5 = nn.InstanceNorm1d(256)

        self.k = k

    def forward(self, x) -> torch.Tensor:
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.in4(self.fc1(x)))
        x = F.relu(self.in5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    """ Instance normalisation replaces batch normalization"""
    def __init__(self, global_feat = True, feature_transform = False, point_transform = False):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d()
        self.point_transform=point_transform
        self.conv1 = torch.nn.Conv1d(5, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]

        if self.point_transform == True:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        else:
            trans = None
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetCls(nn.Module):

    """ After the max pooling form the pointfeat architecture and outputs the classification.
    Args:
        k (int): number of classifications/labels
        feature_transform (bool) : option to do the feature transform (STK3d)
        point_transform (bool) : option to do point transform (STN3d)
        batch_status_of_network (bool) : Option to do batch_normalization or instance normalization

        Returns :
           log softmax

    """
    def __init__(self, k=2, feature_transform=False,point_transform = False, batch_status_of_network=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.point_transform = point_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform, point_transform=point_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.in1 = nn.InstanceNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.in2 = nn.InstanceNorm1d(256)
        self.relu = nn.ReLU()
        self.batch_status_of_network=batch_status_of_network

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        if self.batch_status_of_network == False:
            x = F.relu(self.in1(self.fc1(x)))
            x = F.relu(self.in2(self.dropout(self.fc2(x))))
            print('instance normlaization')
        else :
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.dropout(self.fc2(x))))
            print('batch_nomalization')

        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss


if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,3500))
    trans = STN3d()
    out = trans(sim_data)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('stn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5,batch_status_of_network=True)
    out, _, _ = cls(sim_data)
    print('class', out.size())