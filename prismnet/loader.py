import os, sys, pdb, h5py
import os.path
import numpy as np
import torch

class SeqicSHAPE(torch.utils.data.Dataset):
    def __init__(self, data_path, is_test=False):
        """data loader
        
        Args:
            data_path ([str]): h5 file path
            is_test (bool, optional): testset or not. Defaults to False.
        """
        dataset = h5py.File(data_path, 'r')
        X_train = np.array(dataset['X_train']).astype(np.float32)
        Y_train = np.array(dataset['Y_train']).astype(np.int32)
        X_test  = np.array(dataset['X_test']).astype(np.float32)
        Y_test  = np.array(dataset['Y_test']).astype(np.int32)
        if len(Y_train.shape) == 1:
            Y_train = np.expand_dims(Y_train, axis=1)
            Y_test  = np.expand_dims(Y_test, axis=1)
        X_train = np.expand_dims(X_train, axis=3).transpose([0, 3, 2, 1])
        X_test  = np.expand_dims(X_test,  axis=3).transpose([0, 3, 2, 1])

        train = {'inputs': X_train, 'targets': Y_train}
        test  = {'inputs': X_test,  'targets': Y_test}

        train = self.__prepare_data__(train)
        test  = self.__prepare_data__(test)

        if is_test:
            self.dataset = test
        else:
            self.dataset = train
    
    def __prepare_data__(self, data):
        inputs    = data['inputs'][:,:,:,:4]
        structure = data['inputs'][:,:,:,4:]
        structure = np.expand_dims(structure[:,:,:,0], axis=3)
        inputs    = np.concatenate([inputs, structure], axis=3)
        data['inputs']  = inputs
        return data

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        x = self.dataset['inputs'][index]
        y = self.dataset['targets'][index]
        return x, y


    def __len__(self):
        return len(self.dataset['inputs'])

