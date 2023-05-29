from pyuul import utils
import numpy as np
import torch 
import tensorflow as tf

coords, atname= utils.parsePDB('pdb/1uaz.pdb')

print(coords)

channels = utils.atomlistToChannels(atname)

print(channels)

channels = channels.unsqueeze(0)

print(channels)

channels = tf.expand_dims(channels, 0)

print(channels)

#combined = torch.cat([coords, channels], dim=1)

#print(combined)
