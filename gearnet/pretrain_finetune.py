import os 
import pandas as pd
import torch
import numpy as np
import random
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils import data as torch_data
from torchdrug import models, transforms, data, layers, tasks, core
from torchdrug.layers import geometry
import torchdrug

def compute_sample_weights(targets):
    class_counts = np.bincount(targets)
    class_weights = 1. / (class_counts)
    sample_weights = class_weights[targets]
    return sample_weights

truncate_transform = transforms.TruncateProtein(max_length=3, random=False)
protein_view_transform = transforms.ProteinView(view="residue")
transform = transforms.Compose([truncate_transform, protein_view_transform])
# transform = transforms.Compose([protein_view_transform])
# full_ec_dataset = datasets.EnzymeCommission("./protein-datasets/", transform=transform, atom_feature=None,bond_feature=None)

### OPM Dataset ###
class OPM(data.ProteinDataset):

    splits = ["train", "valid"]
    
    def __init__(self, test_split="valid", verbose=1, transform=None, **kwargs):

        self.test_split = self.splits[-1]

        self.transform = transform

        min_samples = 100

        # Read the CSV file
        full_protein_list = pd.read_csv("/ocean/projects/bio230029p/chrismun/data/proteins.csv")

        # Filter the transMemProteins
        transMemProteins = full_protein_list[full_protein_list['type_id'] == 1]
        transMemProteins['pdbid'] = transMemProteins['pdbid'].str.replace('[^\w]', '', regex=True)

        # Get list of files in the directory
        directory_path = "/ocean/projects/bio230029p/chrismun/data/pdb/pdb_ac_only"
        print(f"directory path : {directory_path}")
        all_files = os.listdir(directory_path)
        all_files = [file.replace('.pdb', '') for file in all_files if file.endswith('.pdb')]

        # Filter transMemProteins based on the files in the directory
        transMemProteins = transMemProteins[transMemProteins['pdbid'].isin(all_files)]

        # Get counts and filter based on min_samples
        counts = transMemProteins['membrane_name_cache'].value_counts()
        selected_list = [key for key, num_samples in counts.items() if num_samples >= min_samples]
        transMemProteins = transMemProteins[transMemProteins['membrane_name_cache'].isin(selected_list)]

        # Create label dictionary
        labels = transMemProteins['membrane_name_cache'].unique()
        label_dict = {key: value for value, key in enumerate(sorted(labels))}

        x = []
        y = []
        self.data = []
        self.sequences = []

        for pdb_id in transMemProteins['pdbid']:
            file_name = pdb_id + '.pdb'
            file_path = os.path.join(directory_path, file_name)

            if os.path.exists(file_path):
                protein = data.Protein.from_pdb(file_path)
                if protein is None:
                    print("protein none")
                    continue

                sample_class = transMemProteins.loc[transMemProteins['pdbid'] == pdb_id, 'membrane_name_cache'].iloc[0]

                x.append(protein)
                y.append(label_dict[sample_class])
                self.sequences.append(protein.to_sequence())
            else:
                print(f"The file {file_path} does not exist.")

        target_dict = {"localization": y}

        self.data = x
        self.targets = target_dict

        class_count = {}
        j = 0
        for _ in x:
            class_count[y[j]] = class_count.setdefault(y[j], 0) + 1
            j += 1

        print("print(class_count): ", class_count)
        print("len(class_count):", len(class_count))

        for k, v in class_count.items():
            print("Label:", k, "count:", v)

        self.num_samples = len(x)
        self.num_classes = len(class_count)
    
    def split(self):
        # Initialize empty lists to store index subsets for training and validation
        train_indices = []
        val_indices = []

        # Loop over each class and perform stratified split
        for class_label in range(self.num_classes):
            class_specific_indices = [i for i, target in enumerate(self.targets["localization"]) if target == class_label]

            train_class_indices, val_class_indices = train_test_split(
                class_specific_indices,
                test_size=0.3,  # 30% of each class will go to validation set
                random_state=42  # Random seed for reproducibility
            )

            train_indices.extend(train_class_indices)
            val_indices.extend(val_class_indices)

        # Shuffle the indices for randomness
        random.shuffle(train_indices)
        random.shuffle(val_indices)

        # Create Subset objects for training and validation datasets
        train_split = torch_data.Subset(self, train_indices)
        val_split = torch_data.Subset(self, val_indices)

        return [train_split, val_split]

    # def get_sample_weights(self):
    #     labels = self.targets["localization"]
    #     class_counts = np.bincount(labels)
    #     num_samples = len(labels)
        
    #     weights_per_class = num_samples / (len(class_counts) * class_counts)
    #     sample_weights = weights_per_class[labels]
    #     return sample_weights

    # def split(self):
    #     sample_weights = self.get_sample_weights()
    #     sample_weights /= sample_weights.sum()
    #     num_samples = len(self.data)
    #     num_val_samples = int(0.3 * num_samples)  # Assuming 30% validation split
        
    #     # Randomly select samples for the validation set based on their weights
    #     val_indices = np.random.choice(
    #         np.arange(num_samples),
    #         size=num_val_samples,
    #         replace=False,
    #         p=sample_weights
    #     )

    #     # Update weights for the remaining indices so they sum to 1
    #     remaining_weights = np.delete(sample_weights, val_indices)
    #     remaining_weights /= remaining_weights.sum()
    #     remaining_indices = np.delete(np.arange(num_samples), val_indices)

    #     # Randomly select samples for the training set based on their weights
    #     train_indices = np.random.choice(
    #         remaining_indices,
    #         size=num_samples - num_val_samples,
    #         replace=False,
    #         p=remaining_weights
    #     )
        
    #     # Create Subset objects for training and validation datasets
    #     train_split = torch_data.Subset(self, train_indices)
    #     val_split = torch_data.Subset(self, val_indices)

    #     return [train_split, val_split]

    def stratified_k_fold_split(self, n_splits=5):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        folds = []
        labels = self.targets["localization"]

        for train_indices, val_indices in skf.split(range(len(self.data)), labels):
            train_subset = torch_data.Subset(self, train_indices)
            val_subset = torch_data.Subset(self, val_indices)

            folds.append((train_subset, val_subset))

        return folds

    def get_item(self, index):
        protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein, "localization": self.targets["localization"][index]}
        if self.transform:
            item = self.transform(item)
        return item
    
    def __getitem__(self, index):
        return self.get_item(index)


# Instantiate dataset 
dataset = OPM(transform=transform, atom_feature=None, bond_feature=None)
train_set, test_set = dataset.split()
# sample_weights = compute_sample_weights(np.array(dataset.targets["localization"])[train_set.indices])
# sample_weights = compute_sample_weights(train_set.targets["localization"])
# sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
# weighted_train_loader = torch_data.DataLoader(train_set, batch_size=16, sampler=sampler)
# Pretrain 
gearnet_edge = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512], 
                                  num_relation=7, edge_input_dim=59, num_angle_bin=8,
                                  batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")

graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()], 
                                                        edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                     geometry.KNNEdge(k=10, min_distance=5),
                                                                     geometry.SequentialEdge(max_distance=2)],
                                                        edge_feature="gearnet")
 
task = tasks.PropertyPrediction(gearnet_edge, graph_construction_model=graph_construction_model, num_mlp_layer=3, num_class=9,
                                              task=['localization'], criterion="ce", metric=["acc"])

optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)

# Create a solver engine
solver = core.Engine(task, train_set, test_set, None, optimizer,
                    gpus=[0], batch_size=16)

# Load pretrained weights
_checkpoint = torch.load("model_weights/mc_gearnet_edge.pth")
task.load_state_dict(_checkpoint, strict=False)

# Fine-tune
solver.train(num_epoch=25)

# Validate
solver.evaluate("valid")
