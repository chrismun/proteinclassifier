from pyuul import utils, VolumeMaker
from torch.utils.data import Dataset,DataLoader
import torch
import pandas as pd
import os
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

full_protein_list = pd.read_csv("./data/proteins.csv")
transMemProteins = full_protein_list[full_protein_list['type_id'] == 1]
transMemProteins['pdbid'] = transMemProteins['pdbid'].str.replace('[^\w]', '', regex=True)  # remove "=...." extra charecters
counts = transMemProteins['membrane_name_cache'].value_counts()
label_dict_counts = {key: counts[key] for key in counts.index}
selected_list = [key for key, value in label_dict_counts.items() if value > 4]
k=len(selected_list)
print(k)
