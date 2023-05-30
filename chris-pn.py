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
# from sklearn.model_selection import train_test_split
# from point_net_channel import *
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
import time
# import wandb
# import wandb
import numpy as np
# from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
num_epochs = 100
batch_size = 32
learning_rate = 0.0001

# Data preparation
# 1. Load data
full_protein_list = "./data/proteins.csv"
full_protein_list = pd.read_csv(full_protein_list)

# 2. Filter for only transmembrane proteins
transMemProteins = full_protein_list[full_protein_list['type_id'] == 1]

# 3. Make dict of labels and counts
counts = transMemProteins['membrane_name_cache'].value_counts()
label_dict_counts = {key: counts[key] for key in counts.index} 

# 4. Filter for only proteins with more than 4 samples
selected_list = [key for key, value in label_dict_counts.items() if value > 4]
k=len(selected_list)
transMemProteins = transMemProteins[transMemProteins['membrane_name_cache'].isin(selected_list)]
labels = transMemProteins['membrane_name_cache'].unique()
label_dict = {key: value for value, key in enumerate(sorted(labels))}
class_names = list(label_dict.keys())

# 5. build input data
inputs = []
labels = []
pdb_list = [] 
for i, pdb_id in enumerate(transMemProteins['pdbid']):
    
    file_name = pdb_id + ".pdb"
    file_path = os.path.join("./data/pdb", file_name)
    
    coords, atname = utils.parsePDB(file_path, keep_hetatm=False)
    radius = utils.atomlistToRadius(atname)
    atom_channel = utils.atomlistToChannels(atname)
    
    inputs.append(torch.cat((coords.squeeze().permute(1,0),radius, at_channel), dim=0))
    localization = transMemProteins.loc[transMemProteins['pdbid'] == pdb_id,'membrane_name_cache'].iloc[0]
    labels.append(torch.tensor(label_dict[localization]))
    pdb_list.append(pdb_id)

print(inputs[0].shape)
print(labels[0].shape)
print(pdb_list[0])