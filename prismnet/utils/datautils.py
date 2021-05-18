from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, h5py
import numpy as np
from copy import deepcopy



def make_directory(path, foldername, verbose=1):
    """make a directory"""

    if not os.path.isdir(path):
        os.mkdir(path)
        print("making directory: " + path)

    outdir = os.path.join(path, foldername)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
        print("making directory: " + outdir)
    return outdir

def finished(path, line_num):
    """check a results file is finished or not

    Args:
        path ([str]): [results file path]
        line_num ([int]): [target line number]
    """

    if os.path.exists(path):
        with open(path, "r") as f:
            if line_num == len(f.readlines()):
                return True
            else:
                return False
    else:
        return False

def get_file_names(dataset_path):
    file_names = []
    for file_name in os.listdir(dataset_path):
        if os.path.splitext(file_name)[1] == '.h5':
            file_names.append(file_name)
    return file_names

def md5(string):
    return hashlib.md5(string.encode('utf-8')).hexdigest()

def mat2str(m):
    string=""
    if len(m.shape)==1:
        for j in range(m.shape[0]):
            string+= "%.3f," % m[j]
    else:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                string+= "%.3f," % m[i,j]
    return string

def rescale(vec, thr=0.0):
    ind0 = np.where(vec>=thr)[0]
    u_norm = 0.5 * (vec[ind0]-thr)/(vec[ind0].max()) + 0.5
    ind2 = np.where(vec<0)[0]
    vec_norm = vec.copy()
    vec_norm[ind0] = u_norm
    vec_norm[ind2] = 0.0
    return vec_norm

    
def decodeDNA(m):
    na=["A","C","G","U"]
    var,inds=np.where(m==1)
    seq=""
    for i in inds:
        seq=seq+na[i]
    return seq

def str_onehot(vec):
    thr=0.15
    mask_str = np.zeros((2,vec.shape[-1]))
    ind =np.where(vec >= thr)[1]
    mask_str[1,ind]=1
    ind =np.where(vec < thr)[1]
    mask_str[0,ind]=1
    ind =np.where(vec == -1)[1]
    mask_str[0,ind]=0.5
    mask_str[1,ind]=0.5
    return mask_str
    
def convert_one_hot(sequence, max_length=None):
    """convert DNA/RNA sequences to a one-hot representation"""

    one_hot_seq = []
    for seq in sequence:
        seq = seq.upper()
        seq_length = len(seq)
        one_hot = np.zeros((4,seq_length))
        index = [j for j in range(seq_length) if seq[j] == 'A']
        one_hot[0,index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'C']
        one_hot[1,index] = 1
        index = [j for j in range(seq_length) if seq[j] == 'G']
        one_hot[2,index] = 1
        index = [j for j in range(seq_length) if (seq[j] == 'U') | (seq[j] == 'T')]
        one_hot[3,index] = 1

        # handle boundary conditions with zero-padding
        if max_length:
            offset1 = int((max_length - seq_length)/2)
            offset2 = max_length - seq_length - offset1

            if offset1:
                one_hot = np.hstack([np.zeros((4,offset1)), one_hot])
            if offset2:
                one_hot = np.hstack([one_hot, np.zeros((4,offset2))])

        one_hot_seq.append(one_hot)

    # convert to numpy array
    one_hot_seq = np.array(one_hot_seq)

    return one_hot_seq

def convert_cat_one_hot(targets):
    """convert DNA/RNA sequences to a one-hot representation"""
    t_length = len(targets)
    cat_num  = len(np.unique(targets))
    one_hot = np.zeros((t_length, cat_num))
    for i in range(cat_num):
        index = np.where(targets==i)[0]
        one_hot[index,i]= 1
    return one_hot

def seq_mutate(seq):
    mut_seq = []
    for i in range(len(seq)):
        if seq[i] == "A" :
            mut_seq.extend([seq[0:i] + "C" + seq[(i+1):], seq[0:i] + "G" + seq[i+1:], seq[0:i] + "T" + seq[i+1:]])
        elif seq[i] == "C" :
            mut_seq.extend([seq[0:i] + "A" + seq[i+1:], seq[0:i] + "G" + seq[i+1:], seq[0:i] + "T" + seq[i+1:]])
        elif seq[i] == "G" :
            mut_seq.extend([seq[0:i] + "A" + seq[i+1:], seq[0:i] + "C" + seq[i+1:], seq[0:i] + "T" + seq[i+1:]])
        else:
            mut_seq.extend([seq[0:i] + "A" + seq[i+1:], seq[0:i] + "C" + seq[i+1:], seq[0:i] + "G" + seq[i+1:]])
    return mut_seq


def load_dataset_hdf5(file_path, ss_type='seq'):

    def prepare_data(train, ss_type=None):
        if ss_type == 'struct':
            structure = train['inputs'][:,:,:,4:9]
            paired = np.expand_dims(structure[:,:,:,0], axis=3)
            train['inputs']  = paired
            return train

        seq = train['inputs'][:,:,:,:4]

        if ss_type == 'pu':
            structure = train['inputs'][:,:,:,4:9]
            paired = np.expand_dims(structure[:,:,:,0], axis=3)

            if structure.shape[-1]>3:
                unpaired = np.expand_dims(np.sum(structure[:,:,:,1:], axis=3), axis=3)
                seq = np.concatenate([seq, paired, unpaired], axis=3)
            elif structure.shape[-1]==1:
                seq = np.concatenate([seq, paired], axis=3)
            elif structure.shape[-1]==2:
                unpaired = np.expand_dims(structure[:,:,:,1], axis=3)
                seq = np.concatenate([seq, paired, unpaired], axis=3)
            elif structure.shape[-1]==3:
                unpaired = np.expand_dims(structure[:,:,:,1], axis=3)
                other = np.expand_dims(structure[:,:,:,2], axis=3)
                seq = np.concatenate([seq, paired, unpaired, other], axis=3)
        elif ss_type == 'p':
            structure = train['inputs'][:,:,:,4:9]
            paired = np.expand_dims(structure[:,:,:,0], axis=3)
            seq = np.concatenate([seq, paired], axis=3)
        elif ss_type == 'struct':
            structure = train['inputs'][:,:,:,4:9]
            paired = np.expand_dims(structure[:,:,:,0], axis=3)
            HIME = structure[:,:,:,1:]
            seq = np.concatenate([seq, paired, HIME], axis=3)
        train['inputs']  = seq
        return train

    # open dataset
    with h5py.File(file_path, 'r') as f:
        # load set A data
        X_train = np.array(f['X_train'])
        Y_train = np.array(f['Y_train'])
        X_test  = np.array(f['X_test'])
        Y_test  = np.array(f['Y_test'])

  

    # expand dims of targets
    if len(Y_train.shape) == 1:
        Y_train = np.expand_dims(Y_train, axis=1)
        Y_test  = np.expand_dims(Y_test, axis=1)

    # add another dimension to make a 4d tensor
    X_train = np.expand_dims(X_train, axis=3).transpose([0, 2, 3, 1])
    X_test  = np.expand_dims(X_test,  axis=3).transpose([0, 2, 3, 1])
    
    # dictionary for each dataset
    train = {'inputs': X_train, 'targets': Y_train}
    test  = {'inputs': X_test, 'targets': Y_test}
    

    # parse secondary structure profiles
    train = prepare_data(train, ss_type)
    test  = prepare_data(test, ss_type)

    print("train:",train['inputs'].shape)
    print("test:",test['inputs'].shape)

    return train, test


def process_data(train, test, method='log_norm'):
    """get the results for a single experiment specified by rbp_index.
    Then, preprocess the binding affinity intensities according to method.
    method:
        clip_norm - clip datapoints larger than 4 standard deviations from the mean
        log_norm - log transcormation
        both - perform clip and log normalization as separate targets (expands dimensions of targets)
    """

    def normalize_data(data, method):
        if method == 'standard':
            MIN = np.min(data)
            data = np.log(data-MIN+1)
            sigma = np.mean(data)
            data_norm = (data)/sigma
            params = sigma
        if method == 'clip_norm':
            # standard-normal transformation
            significance = 4
            std = np.std(data)
            index = np.where(data > std*significance)[0]
            data[index] = std*significance
            mu = np.mean(data)
            sigma = np.std(data)
            data_norm = (data-mu)/sigma
            params = [mu, sigma]

        elif method == 'log_norm':
            # log-standard-normal transformation
            MIN = np.min(data)
            data = np.log(data-MIN+1)
            mu = np.mean(data)
            sigma = np.std(data)
            data_norm = (data-mu)/sigma
            params = [MIN, mu, sigma]

        elif method == 'both':
            data_norm1, params = normalize_data(data, 'clip_norm')
            data_norm2, params = normalize_data(data, 'log_norm')
            data_norm = np.hstack([data_norm1, data_norm2])
        return data_norm, params


    # get binding affinities for a given rbp experiment
    Y_train = train['targets']
    Y_test = test['targets']
    #import pdb; pdb.set_trace()

    if len(Y_train.shape)==1:
        # filter NaN
        train_index = np.where(np.isnan(Y_train) == False)[0]
        test_index = np.where(np.isnan(Y_test) == False)[0]
        Y_train = Y_train[train_index]
        Y_test = Y_test[test_index]
        X_train = train['inputs'][train_index]
        X_test = test['inputs'][test_index]
    else:
        X_train = train['inputs']
        X_test = test['inputs']

    # normalize intenensities
    if method:
        Y_train, params_train = normalize_data(Y_train, method)
        Y_test, params_test = normalize_data(Y_test, method)

    # store sequences and intensities
    train = {'inputs': X_train, 'targets': Y_train}
    test = {'inputs': X_test, 'targets': Y_test}

    return train, test


def down_negative_samples(train, test, ratio=0.0):
    """get the results for a single experiment specified by rbp_index.
    Then, preprocess the binding affinity intensities according to method.
    method:
        clip_norm - clip datapoints larger than 4 standard deviations from the mean
        log_norm - log transcormation
        both - perform clip and log normalization as separate targets (expands dimensions of targets)
    """
    if ratio==0.0:
        print("No negative down-sampling ratio.")
        return train, test

    X_train = train['inputs']
    X_test  = test['inputs']

    Y_train = train['targets']#.astype(np.int32)
    Y_test  = test['targets']#.astype(np.int32)

    pos_index_tr = np.where(Y_train==1)[0]
    pos_index_te = np.where(Y_test==1)[0]

    neg_index_tr = np.where(Y_train==0)[0]
    neg_index_te = np.where(Y_test==0)[0]

    n_down_neg_tr = int(ratio * (len(Y_train) - len(neg_index_tr)))
    n_down_neg_te = int(ratio * (len(Y_test) -  len(neg_index_te)))

    dw_neg_index_tr = np.random.choice(neg_index_tr, size=n_down_neg_tr)
    dw_neg_index_te = np.random.choice(neg_index_te, size=n_down_neg_te)

    pos_neg_tr =np.concatenate((dw_neg_index_tr,    pos_index_tr))
    pos_neg_te =np.concatenate((dw_neg_index_te,    pos_index_te))

    train = {'inputs': X_train[pos_neg_tr], 'targets': Y_train[pos_neg_tr]}
    test = {'inputs': X_test[pos_neg_te], 'targets': Y_test[pos_neg_te]}

    return train, test


def load_testset_txt_only_seq(filepath, test, return_trans_id=False, seq_length=101):
    print("Reading inference file(only seq):", filepath)
    if os.path.exists(filepath+"_test.h5"):
        print("loading from h5.")        
        with h5py.File(filepath+"_test.h5", 'r') as f:
            # load set A data
            test['inputs']  = f['inputs']
            test['targets'] = f['targets']
     
        
        if return_trans_id:
            blob = np.load(filepath+"_tran.npz")
            trans_ids = blob['trans_ids']
            return test, trans_ids
        else:
            return test

    seqs = []
    trans_ids = []
    with open(filepath,"r") as f:
        for line in f.readlines():
            line=line.strip('\n').split('\t')
            if len(line[2])!=seq_length:
                continue
            trans_ids.append(line[0])
            seqs.append(line[1])
    print("Converting.")        
    input = convert_one_hot(seqs, seq_length)
    print("Converted.")        

    inputs = np.expand_dims(input, axis=3).transpose([0, 2, 3, 1])
    targets = np.ones((inputs.shape[0],1))
    targets[inputs.shape[0]-1]=0

    test['inputs'] =inputs
    test['targets'] =targets
    
    print("Saving into h5.")
    with h5py.File(filepath+"_test.h5", "w") as f:
        dset = f.create_dataset("inputs", data=inputs, compression="gzip")
        dset = f.create_dataset("targets", data=targets, compression="gzip")
    print("Saved.")

    if return_trans_id:
        trans_ids = np.array(trans_ids)
        return test, trans_ids
    else:
        return test



def load_testset_txt(filepath, use_structure=True, seq_length=101):
    test = {}

    print("Reading inference file:", filepath)
    if os.path.exists(filepath+"_test.npz"):
        print("loading from npz.")        
       
        f = np.load(filepath+"_test.npz", allow_pickle=True)
        test['inputs']  = f['inputs']
        test['targets'] = f['targets']

        return test

    in_ver = 5
    seqs = []
    strs = []
    with open(filepath,"r") as f:
        for line in f.readlines():
            line=line.strip('\n').split('\t')
            if len(line[2])!=seq_length:
                continue
            seqs.append(line[2])
            if use_structure:
                strs.append(line[3])
    in_seq = convert_one_hot(seqs, seq_length)
    
    if use_structure:
        structure = np.zeros((len(seqs), in_ver-4, seq_length))
        for i in range(len(seqs)):
            icshape = strs[i].strip(',').split(',')
            ti = [float(t) for t in icshape]
            ti = np.array(ti).reshape(1,-1)
            structure[i] = np.concatenate([ti], axis=0)
        input = np.concatenate([in_seq, structure], axis=1)
    else:
        input = in_seq

    inputs = np.expand_dims(input, axis=3).transpose([0, 3, 2, 1])
    targets = np.ones((in_seq.shape[0],1))

    targets[in_seq.shape[0]-1]=0

    test['inputs']  = inputs
    test['targets'] = targets
    print("Saving into npz.")
    np.savez_compressed(filepath+"_test.npz", inputs=inputs, targets=targets)
    print("Saved.")

    return test



def load_testset_txt_mu(filepath, test, seq_length=101):
    print("Reading test file:", filepath)
    f_mu = open(filepath,"r")
    seqs = []
    strs = []
    use_pu = True
    if test['inputs'].shape[-1]==4:
        use_pu = False
    nf=0
    for line in f_mu.readlines():
        nf+=1
        line=line.strip('\n').split('\t')
        if len(line[2])!=seq_length:
            continue
        seqs.append(line[2])
        mut_seq=seq_mutate(line[2])
        seqs.extend(mut_seq)
        if use_pu:
            strs.extend([line[3]] * len(seqs))
    print("file line num:",nf)
    print("mut seq num:",len(seqs))
    in_seq = munge.convert_one_hot(seqs, seq_length)
    in_ver = 5
    if use_pu:
        structure = np.zeros((len(seqs), in_ver-4, seq_length))
        for i in range(len(seqs)):
            struct_list = strs[i].strip(',').split(',')
            ti = np.array([float(t) for t in struct_list]).reshape(1,-1)
            structure[i] = np.concatenate([ti], axis=0)
        input = np.concatenate([in_seq, structure], axis=1)
    else:
        input = in_seq

    inputs = np.expand_dims(input, axis=3).transpose([0, 2, 3, 1])
    targets = np.ones((in_seq.shape[0],1))

    targets[in_seq.shape[0]-1]=0

    test['inputs'] =inputs
    test['targets'] =targets
    return test


def split_dataset(data, targets, valid_frac=0.2):
    
    ind0 = np.where(targets<0.5)[0]
    ind1 = np.where(targets>=0.5)[0]
    
    n_neg = int(len(ind0)*valid_frac)
    n_pos = int(len(ind1)*valid_frac)

    shuf_neg = np.random.permutation(len(ind0))
    shuf_pos = np.random.permutation(len(ind1))

    X_train = np.concatenate((data[ind1[shuf_pos[n_pos:]]], data[ind0[shuf_neg[n_neg:]]]))
    Y_train = np.concatenate((targets[ind1[shuf_pos[n_pos:]]], targets[ind0[shuf_neg[n_neg:]]]))
    train = (X_train, Y_train)

    X_test = np.concatenate((data[ind1[shuf_pos[:n_pos]]], data[ind0[shuf_neg[:n_neg]]]))
    Y_test = np.concatenate((targets[ind1[shuf_pos[:n_pos]]], targets[ind0[shuf_neg[:n_neg]]]))
    test = (X_test, Y_test)

    return train, test
