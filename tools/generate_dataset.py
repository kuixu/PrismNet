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

def read_csv(path):
    # load sequences
    df = pd.read_csv(path, sep='\t', header=None)
    df = df.loc[df[0]!="Type"]

    Type  = 0
    loc   = 1
    Seq   = 2
    Str   = 3
    Score = 4
    label = 5

    rnac_set  = df[Type].to_numpy()
    sequences = df[Seq].to_numpy()
    structs  = df[Str].to_numpy()
    targets   = df[Score].to_numpy().astype(np.float32).reshape(-1,1)
    return sequences, structs, targets

max_length = 101
only_pos   = False
binary     = True

name       = sys.argv[1]
is_bin     = sys.argv[2]
in_ver     = int(sys.argv[3])
data_path  = sys.argv[4]

print(name)



outfile = name+'.h5'
sequences, structs, targets = read_csv(os.path.join(data_path, name+'.tsv'))

# combine inpute data
one_hot = datautils.convert_one_hot(sequences, max_length)
structure = np.zeros((len(icshapes), in_ver-4, max_length))
for i in range(len(icshapes)):
    icshape = icshapes[i].split(',')
    ti = [float(t) for t in icshape]
    ti = np.array(ti).reshape(1,-1)
    pu = np.concatenate([ti], axis=0)
    structure[i] = pu
data = np.concatenate([one_hot, structure], axis=1)

# preprare targets
if is_bin=="0":
    targets = datautils.rescale(targets)
elif is_bin=="1":
    targets[targets<0] = 0
    targets[targets>0] = 1


# split dataset into train, cross-validation, and test set
train, test = datautils.split_dataset(data, targets, rnac_set, valid_frac=0.2)

target_data_type = np.int32 if is_bin=="1" else np.float32
# save dataset
save_path = os.path.join(data_path, outfile)
print(name, data.shape, len(train[0]), len(test[0]), test[1].max(), test[1].min())
# print('saving dataset: ', save_path)
with h5py.File(save_path, "w") as f:
    dset = f.create_dataset("X_train", data=train[0].astype(np.float32), compression="gzip")
    dset = f.create_dataset("Y_train", data=train[1].astype(target_data_type), compression="gzip")
    dset = f.create_dataset("X_test", data=test[0].astype(np.float32), compression="gzip")
    dset = f.create_dataset("Y_test", data=test[1].astype(target_data_type), compression="gzip")
