# parse pdb to numpy array 
# x = np.array(n_samples, n_points, 4 channels)
# .npz

import os
import numpy as np
import pandas as pd
from pyuul import utils
import torch

min_samples = 100
# median = 8514

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
all_coord_lengths = []
for i, pdb_id in enumerate(transMemProteins['pdbid']):
    file_name = pdb_id + '.pdb'
    file_path = os.path.join('/jet/home/munley/proteinclassifier/data/pdb', file_name)

    if os.path.exists(file_path):
        coords, atname = utils.parsePDB(file_path, keep_hetatm=False)
        radius = utils.atomlistToRadius(atname)
        atomType = utils.atomlistToChannels(atname)

        radius = radius.tolist()[0]
        atomType = atomType.tolist()[0]

        file_name = pdb_id + '.pdb'
        file_path = os.path.join('/jet/home/munley/proteinclassifier/data/pdb', file_name)

        coords, atname = utils.parsePDB(file_path, keep_hetatm=False)
        # radius = utils.atomlistToRadius(atname)
        atomType = utils.atomlistToChannels(atname)

        # radius = radius.tolist()[0]
        atomType = atomType.tolist()[0]

        coords = coords.tolist()
        coords = coords[0]
        all_coord_lengths.append(len(coords))
        # filter
        if len(coords) < 1000 or len(coords) > 8192:
            continue

        i = 0

        for coord in coords:
            # coord.append(radius[i])
            coord.append(atomType[i])
            i += 1

        a = transMemProteins.loc[transMemProteins['pdbid'] == pdb_id, 'membrane_name_cache'].iloc[0]

        x.append(coords)
        y.append([label_dict[a]])
    else:
        print(f"The file {file_path} does not exist.")


max_len = max(len(entry) for entry in x)
min_len = min(len(entry) for entry in x)
print(max_len, min_len)

max_len = 8192

for i, entry in enumerate(x):
    if len(entry) < max_len:
        orig_len = len(entry)
        num_padding = max_len - len(entry)
        for j in range(0, num_padding):
            # append a point until theyre all the same length
            x[i].append(x[i][j % orig_len])

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

# coord len data info
# print()
mean_value = np.mean(all_coord_lengths)
median_value = np.median(all_coord_lengths)
std_dev = np.std(all_coord_lengths)
q1, q3 = np.percentile(all_coord_lengths, [25, 75])
print("Min: ", min(all_coord_lengths))
print("Max: ", max(all_coord_lengths))
print("Mean:", mean_value)
print("Median:", median_value)
print("Standard Deviation:", std_dev)
print("Q1:", q1)
print("Q3:", q3)
np.savez('protein-data.npz', x=x, y=y)
