
# import sys
# sys.path.append('/Users/bivekpokhrel/PycharmProjects/database/my_pointnet')

import random
import shutil
from pyuul import utils, VolumeMaker
from torch.utils.data import Dataset,DataLoader
import torch
import pandas as pd
import os
import timeit
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from my_pointnet.point_net_channel import *
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
import time
import wandb
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.login()





config_default = {

    'csv_path': '/Users/bivekpokhrel/PycharmProjects/database/data/proteins-2023-04-15.csv',
    'source_folder': '/Users/bivekpokhrel/PycharmProjects/database/data/pdb_3',
    'destination_folder': '/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder',
    'checkpoint_folder': '/Users/bivekpokhrel/PycharmProjects/database/my_pointnet/my_checkpoint',
    'load_previous_epoch': False, # True if want to load
    'resume_epoch':0, # check the last epoch from the saved checkpoint
    'num_protiens':75,
    'min_counts':3
                    }




sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'optimizer': {
            'values': ['sgd']
        },
        'batch_size': {
            'values': [1,10]
        },
        'learning_rate': {
            'values': [0.001,0.01,0.005]
        },
        'feature_transform': {
            'values': [True, False]
        },
        'point_transform': {
            'values': [True, False]
        },
        'num_epoch': {
            'value': 80 # number of epoch to run after resume epoch
        }
    }
}


sweep_id= wandb.sweep(sweep_config,project="Pointnet_training_allchannel5")
#






# initial_time = timeit.default_timer()


def check_matches(list_to_check,index_no):
    sample=list_to_check[index_no]
    return sample

def validating_it(trans_df,pdb_list,labels,label_dict):
    """ to validate if pdb_list form trans_df and labels ( not input as it has coordinates) are matching or not
     by accessing it's key value from the label_dictionary"""
    sample_position_list=random.sample(range(0,config_default['num_protiens']),k=1)

    for sample_position in sample_position_list:

        label_value = int(labels[sample_position])
        print(f' !!! open pdb file and check if the {pdb_list[sample_position]} has {get_key_by_value(label_dict, label_value)} localization !!!')


def get_key_by_value(dict_obj, value):
    for k, v in dict_obj.items():
        if v == value:
            return k
    return None

#
# def get_coords(protien_path):
#     return utils.parsePDB(protien_path, keep_hetatm=False)[0]


def get_coords(protien_path):
    coords,atname=utils.parsePDB(protien_path, keep_hetatm=False)[0],utils.parsePDB(protien_path, keep_hetatm=False)[1]
    radius=utils.atomlistToRadius(atname)
    atom_channel= utils.atomlistToChannels(atname)
    return coords, radius, atom_channel

# Calculate confusion matrix
def calulate_cm(predictions, labels):
    cm = confusion_matrix(predictions, labels)
    return cm



# Log metrics to wandb



def calculate_precision(predictions, labels):
    # Calculate precision
    precision = precision_score(labels, predictions, average='weighted',zero_division=1)
    return precision

def calculate_recall(predictions, labels):
    # Calculate recall
    recall = recall_score(labels, predictions, average='weighted',zero_division=1)
    return recall

def calculate_f1_score(predictions, labels):
    # Calculate F1 score
    f1 = f1_score(labels, predictions, average='weighted',zero_division=1)
    return f1



## to make use of the padding (equal size) from pyuul making folder
def create_destination_folder(source,destination,df,col):
    """ make a folder"""
    not_entered_protiens=[]
    try:
        if not os.path.exists(destination):
            print('No directory making it !!')
            os.mkdir(destination)

        if os.path.exists(destination):
            print('Directory there, deleting and making new')
            shutil.rmtree(destination)
            os.mkdir(destination)
    except OSError:
        print(f'Error: Could not create a destination folder {destination}')
        exit()

    for i, pdb_id in enumerate(df[col]):
        file_name = pdb_id + '.pdb'

        try:

            source_file = os.path.join(source, file_name)
            destination_file = os.path.join(destination, file_name)
            # print(source_file)
            # print(destination_file)
            shutil.copy2(source_file, destination_file)


        except IOError:

            not_entered_protiens.append(pdb_id)
            print(f'Could not copy file {file_name} to destination folder {config_default["destination_folder"]}')
    # To make the array of the coordinates from the destination folder
    # input_array=get_coords(destination_folder)
    # print(input_array.shape)

    return not_entered_protiens
# creating a new dataframe for type_id=1 (transmembrane)
def make_lists(csv_path, padding):
    """ read OPM database csv -> select transmembrane -> create labels based on localization ->
    create list of inputs (i.e list of tensor) (pdbid) and labels without padding i.e individual size feed in network"""
    # df = pd.read_csv(csv_path)
    #
    # trans_a_df = df[df['type_id'] == 1] # Selecting only the transmembrane protien
    # trans_df = trans_a_df.iloc[:config_default['num_protiens'], :].copy() #make a copy of 50 pdb to avoid error
    # trans_df['pdbid'] = trans_df['pdbid'].str.replace('[^\w]', '', regex=True) #remove "=...." extra charecters
    #
    # #make a dictionary from original df and sort them
    # location=df['membrane_name_cache'].unique()
    # label_dict={key:value for value,key in enumerate(sorted(location))}
    #
    # inputs=[] # coordinates -> list [  Num_protiens * tensor(3,NA) ]
    # labels=[] # store labels -> list [ Num_protiens * tensor(1)]
    # pdb_list=[]# also noting the pdbid in parallel and correct or not can be verified by checking labels and pdbid
    # num_classes=24
    # invalids = create_destination_folder(config_default['source_folder'], config_default['destination_folder'],trans_df,col='pdbid')

    ###
    df = pd.read_csv(csv_path)

    trans_a_df = df[df['type_id'] == 1]  # Selecting only the transmembrane protien
    trans_b_df = trans_a_df.iloc[:config_default['num_protiens'], :].copy()  # make a copy of 50 pdb to avoid error
    trans_b_df['pdbid'] = trans_b_df['pdbid'].str.replace('[^\w]', '', regex=True)  # remove "=...." extra charecters

    # make a dictionary from original df and sort them
    # location=trans_df['membrane_name_cache'].unique()
    # label_dict={key:value for value,key in enumerate(sorted(location))}
    # location = trans_b_df['membrane_name_cache'].unique()
    # label_dict = {key: value for value, key in enumerate(sorted(location))}
    counts = trans_b_df['membrane_name_cache'].value_counts()
    label_dict_counts = {key: counts[key] for key in counts.index}  # to make another dictionary to select labels

    selected_list = [key for key, value in label_dict_counts.items() if value > config_default['min_counts']]
    print(f'selected list : {selected_list}')
    k=len(selected_list)
    trans_df = trans_b_df[trans_b_df['membrane_name_cache'].isin(selected_list)]
    location = trans_df['membrane_name_cache'].unique()
    label_dict = {key: value for value, key in enumerate(sorted(location))}
    class_names = list(label_dict.keys())
    print(label_dict)
    print(class_names)

    inputs = []  # coordinates -> list [  Num_protiens * tensor(3,NA) ]
    labels = []  # store labels -> list [ Num_protiens * tensor(1)]
    pdb_list = []  # also noting the pdbid in parallel and correct or not can be verified by checking labels and pdbid
    num_classes = 24
    invalids = create_destination_folder(config_default['source_folder'], config_default['destination_folder'],
                                         trans_df, col='pdbid')

    if padding == False:
        for i, pdb_id in enumerate(trans_df['pdbid']):
            encoding_array = torch.zeros(num_classes)
            file_name=pdb_id+'.pdb'
            file_path=os.path.join(config_default['destination_folder'],file_name)
            coords, radius, at_channel = get_coords(file_path)

            inputs.append(torch.cat((coords.squeeze().permute(1,0),radius, at_channel), dim=0))
            a=trans_df.loc[trans_df['pdbid'] == pdb_id,'membrane_name_cache'].iloc[0]
            labels.append(torch.tensor(label_dict[a]))
            pdb_list.append(pdb_id)

            #one hot encoding
            # encoding_array[label_dict[a]]=1
            # labels.append(encoding_array)
        # validating_it(trans_df,pdb_list,labels,label_dict)
        return inputs,labels,pdb_list, k, class_names
    elif padding == True:
        # all_tensor=get_coords(config_default['destination_folder']).permute(0,2,1)
        coords, radius, at_channel=get_coords(config_default['destination_folder'])
        all_tensor=torch.cat((coords.permute(0,2,1),radius.unsqueeze(1), at_channel.unsqueeze(1)), dim=1)

        for i, pdb_id in enumerate(trans_df['pdbid']):
            encoding_array = torch.zeros(num_classes)
            inputs.append(all_tensor[i,:,:])
            file_name=pdb_id+'.pdb'
            file_path=os.path.join(config_default['destination_folder'],file_name)
            a=trans_df.loc[trans_df['pdbid']==pdb_id,'membrane_name_cache'].iloc[0]
            labels.append(torch.tensor(label_dict[a]))
            pdb_list.append(pdb_id)
        # validating_it(trans_df,pdb_list,labels,label_dict)
        return inputs, labels, pdb_list, k, class_names
    else:
        print('Check the trans_folder')
        pass
    #


# inputs, labels, pdb_list = make_lists(config_default['csv_path'], padding=True)
# print(len(inputs))
# print(inputs[0].shape)
# print(inputs[1].shape)
# print(labels)
# print(inputs[0][3:,:])


def collate_fn(batch):
    inputs, labels, indices = zip(*batch)
    return torch.stack(inputs), torch.stack(labels), indices








class Mydataset(Dataset):
    def __init__(self, images, classes):
        self.images=images
        self.classes=classes

    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        a=self.images[index]
        b=self.classes[index]
        a1=a.clone().detach()
        b1=b.clone().detach()
        return a1,b1,index
#create folder
# print('********')
#
#
def build_set(csv_path,padding):

    inputs, labels, pdbs, k, class_names = make_lists(csv_path, padding=padding)

    train_data, val_test_data, train_labels, val_test_labels = train_test_split(inputs, labels, test_size=0.2)
    val_data,test_data, val_labels,test_labels=train_test_split(val_test_data,val_test_labels,test_size=0.5)
    train_set=Mydataset(train_data,train_labels)
    val_set=Mydataset(val_data,val_labels)
    test_set=Mydataset(test_data,test_labels)

    return train_set, val_set, test_set, k, class_names


#
def build_optimizer(network,learning_rate):

    optimizer=torch.optim.SGD(network.parameters(),lr=learning_rate)

    return optimizer


def calculate_accuracy(outputs, labels):
    _, predictions = torch.max(outputs, 1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = 100 * correct / total
    return accuracy

def train_epoch(network, train_loader, val_loader, optimizer, feature_transform,num_class,class_names):
    running_loss = 0.0
    total_samples = 0
    total_correct = 0

    for inputs, labels, index in tqdm(train_loader, leave=True, ncols=80):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs, trans, trans_feat = network(inputs)
            loss = F.nll_loss(outputs, labels)
            if feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        _, predictions = torch.max(outputs, 1) # same as the calculate_accuracy function
        correct = (predictions == labels).sum().item()
        total_samples += labels.size(0)
        total_correct += correct

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * total_correct / total_samples

    network.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0.0
        cm= torch.zeros(num_class,num_class)
        predicted_labels = []
        true_labels = []

        for inputs, labels, _ in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = network(inputs)[0]
            loss = F.nll_loss(outputs, labels)
            accuracy = calculate_accuracy(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            total_accuracy += accuracy * inputs.size(0)
            total_samples += inputs.size(0)

            predicted_labels_batch = torch.argmax(outputs, dim=1).cpu().numpy()
            true_labels_batch = labels.cpu().numpy()

            predicted_labels.extend(predicted_labels_batch)
            true_labels.extend(true_labels_batch)

            # Calculate confusion matrix
            # predicted_labels = torch.argmax(outputs, dim=1)
            for i in range(len(labels)):
                true_label = labels[i].item()
                predicted_label = predicted_labels[i].item()
                cm[true_label][predicted_label] += 1
                print(f'this is cm {cm}')
            print(f'this is final cm {cm}')


        val_loss = total_loss / total_samples
        val_accuracy = total_accuracy / total_samples
        f1 = f1_score(true_labels, predicted_labels, average='weighted',zero_division=1)
        precision = precision_score(true_labels, predicted_labels, average='weighted',zero_division=1)
        recall = recall_score(true_labels, predicted_labels, average='weighted',zero_division=1)
        print(f'this is the size of the predicted labels {len(true_labels)}')
        print(true_labels)
        print(predicted_labels)

        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            y_true=true_labels,
            preds=predicted_labels,
            class_names=class_names,

            title="Confusion Matrix")})

    return train_loss, val_loss, train_accuracy, val_accuracy, f1, precision, recall, cm














def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        batch_size = config.batch_size
        feature_transform=config.feature_transform
        point_transform = config.point_transform




        # num_epoch =config.num_epoch

        num_epoch=2
        resume_epoch=config_default['resume_epoch']
        print(config)
        if batch_size > 1:
            batch_status_of_network = True
            padding=True
        elif batch_size==1:
            batch_status_of_network = False
            padding = False
        trainset, valset, testset, k, class_names  = build_set(config_default['csv_path'], padding)
        print(f'THis is k {k}')
        model=PointNetCls(k=k,point_transform=point_transform, feature_transform=feature_transform, batch_status_of_network=batch_status_of_network).to(device)
        optimizer = build_optimizer(model, learning_rate=config.learning_rate)

        if resume_epoch >= 1 and config_default['load_previous_epoch']:
            checkpoint = torch.load(config_default['checkpoint_folder'] + '_epoch_' + str(resume_epoch) + '.pth')
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']


        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
        # optimizer=build_optimizer(model,learning_rate=config.learning_rate)
        print(config)
        wandb.watch(model)
        for epoch in range(num_epoch+resume_epoch):
            train_loss,val_loss, acc, acc2, f1, precision, recall,epoch_cm= train_epoch(model,train_loader,val_loader,optimizer,feature_transform=feature_transform,num_class=k,class_names = class_names )

            # # Log the metrics to WandB
            # wandb.log({"F1 Score": , "Precision": precision, "Recall": recall})

            # return train_loss,val_loss, acc, acc2
            wandb.log({'epoch_train_loss': train_loss ,'epoch_val_loss': val_loss,'train_accuracy' : acc, 'val_accuracy' : acc2, 'epoch': epoch + resume_epoch + 1})
            wandb.log({"Confusion Matrix":epoch_cm,
                       "F1 Score": f1,
                       "Precision": precision,
                       "Recall": recall})


            checkpoint= {
                'epoch' : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'train_loss' : train_loss,
                'val_accuracy' : acc2

            }
            # torch.save(checkpoint, config_default['checkpoint_folder'] + '_epoch_' + str(epoch) + '.pth') # saving the checkpoint per epoch


wandb.agent(sweep_id,train,count=3)





