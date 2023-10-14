# parse pdb to numpy array 
# x = np.array(n_samples, n_points, 4 channels)
# .npz

import os
import numpy as np
import pandas as pd
from pyuul import utils, VolumeMaker
import torch

min_samples = 100

full_protein_list = pd.read_csv("/jet/home/munley/proteinclassifier/data/proteins.csv")
transMemProteins = full_protein_list[full_protein_list['type_id'] == 1]
transMemProteins['pdbid'] = transMemProteins['pdbid'].str.replace('[^\w]', '', regex=True)  # remove "=...." extra characters
counts = transMemProteins['membrane_name_cache'].value_counts()
label_dict_counts = {key: counts[key] for key in counts.index}
selected_list = [key for key, num_samples in label_dict_counts.items() if num_samples > min_samples]
k = len(selected_list)
transMemProteins = transMemProteins[transMemProteins['membrane_name_cache'].isin(selected_list)]
labels = transMemProteins['membrane_name_cache'].unique()
label_dict = {key: value for value, key in enumerate(sorted(labels))}
labels = list(label_dict.keys())

x = []
y = []
for i, pdb_id in enumerate(transMemProteins['pdbid']):
    file_name = pdb_id + '.pdb'
    file_path = os.path.join('/jet/home/munley/proteinclassifier/data/pdb', file_name)

    if os.path.exists(file_path):
        coords, atname = utils.parsePDB(file_path, keep_hetatm=False)
        radius = utils.atomlistToRadius(atname)
        #atomType = utils.atomlistToChannels(atname)

        surface_maker = VolumeMaker.PointCloudSurface(device='cpu')
        surface_point_cloud = surface_maker.forward(coords, radius, maxpoints=1024, external_radius_factor=1.4)
        surface_point_cloud = surface_point_cloud.tolist()

        sample_class = transMemProteins.loc[transMemProteins['pdbid'] == pdb_id, 'membrane_name_cache'].iloc[0]

        x.append(surface_point_cloud)
        y.append([label_dict[sample_class]])
    else:
        print(f"The file {file_path} does not exist.")

class_count = {}
j = 0
for prot in x:
    class_count[y[j][0]] = class_count.setdefault(y[j][0], 0) + 1
    j += 1

sorted_dict = dict(sorted(class_count.items(), key=lambda x: int(x[0])))

print("print(class_count): ", class_count)
print("len(class_count):", len(class_count))

for k, v in class_count.items():
    print("Label:", k, "count:", v)

class_count = class_count
num_classes = len(class_count)
nsamples = len(x)
x = np.array(x)
y = np.array(y)
print(x.shape, y.shape)
np.savez('protein-data.npz', x=x, y=y)