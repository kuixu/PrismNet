from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)
import os, sys, h5py
import pandas as pd
import numpy as np
np.random.seed(100)

from prismnet.utils import datautils

max_length = 101
only_pos   = False
binary     = True
nega_zeros = False
cat        = False
header     = 0

name = sys.argv[1]
is_bin = sys.argv[2]
in_ver = int(sys.argv[3])
data_path =sys.argv[4]

print(name)

if is_bin=="0":
    binary = False
    cat = False
    outfile = name+'.h5'
elif is_bin=="1":
    binary = True
    cat = False
    outfile = name+'.h5'
elif is_bin=="2":
    binary = False
    cat = True
    outfile = name+'.h5'
else:
    raise "error bin parameter."

# load sequences
df = pd.read_csv(os.path.join(data_path, name+'.tsv'), sep='\t', header=header)

if header is None or 'Type' not in df.keys():
    Type  = 0
    loc   = 1
    Seq   = 2
    Str   = 3
    Score = 4
    label = 5
else:
    Type  = 'Type'
    loc   = 'name'
    Seq   = 'Seq'
    Str   = 'icshape'
    Score = 'Score'
    label = 'label'

rnac_set  = df[Type].to_numpy()
sequences = df[Seq].to_numpy()
icshapes  = df[Str].to_numpy()
targets   = df[Score].to_numpy().reshape(-1,1)
if cat:
    targets = df[label].to_numpy().reshape(-1,1)

if only_pos:
    ind01 = np.where(targets>0)[0]
    rnac_set = rnac_set[ind01]
    sequences = sequences[ind01]
    icshapes = icshapes[ind01]
    targets = targets[ind01]

if nega_zeros:
    targets[targets<0] = 0

if binary:
    targets[targets<0] = 0
    targets[targets>0] = 1
else:
    targets = datautils.rescale(targets)

if cat:
    # print("targets:",np.unique(targets,return_counts=True))
    targets = datautils.convert_cat_one_hot(targets)

one_hot = datautils.convert_one_hot(sequences, max_length)
structure = np.zeros((len(icshapes), in_ver-4, max_length))
for i in range(len(icshapes)):
    icshape = icshapes[i].split(',')
    ti = [float(t) for t in icshape]
    ti = np.array(ti).reshape(1,-1)
    pu = np.concatenate([ti], axis=0)
    structure[i] = pu

# merge sequences and structural profiles
data = np.concatenate([one_hot, structure], axis=1)

# split dataset into train, cross-validation, and test set
train, test = datautils.split_dataset(data, targets, rnac_set, valid_frac=0.2)

target_data_type = np.int32 if binary else np.float32
# save dataset
save_path = os.path.join(data_path, outfile)
print(name, data.shape, len(train[0]), len(test[0]), test[1].max(), test[1].min())
# print('saving dataset: ', save_path)
with h5py.File(save_path, "w") as f:
    dset = f.create_dataset("X_train", data=train[0].astype(np.float32), compression="gzip")
    dset = f.create_dataset("Y_train", data=train[1].astype(target_data_type), compression="gzip")
    dset = f.create_dataset("X_test", data=test[0].astype(np.float32), compression="gzip")
    dset = f.create_dataset("Y_test", data=test[1].astype(target_data_type), compression="gzip")
