import os, sys, pdb, h5py
import os.path
import numpy as np
import torch
import torch.utils.data

class SeqicSHAPE(torch.utils.data.Dataset):
    def __init__(self, data_path, is_test=False, is_infer=False, use_structure=True):
        """data loader
        
        Args:
            data_path ([str]): h5 file path
            is_test (bool, optional): testset or not. Defaults to False.
        """
        if is_infer:
            self.dataset = self.__load_infer_data__(data_path, use_structure=use_structure)
            print("infer data: ", self.__len__()," use_structure: ", use_structure)
        else:
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

            labels, nums = np.unique(Y_train,return_counts=True)
            print("train:", labels, nums)
            labels, nums = np.unique(Y_test,return_counts=True)
            print("test:", labels, nums)

            train = self.__prepare_data__(train)
            test  = self.__prepare_data__(test)

            if is_test:
                self.dataset = test
            else:
                self.dataset = train

        

    def __load_infer_data__(self, data_path, use_structure=True):
        from prismnet.utils import datautils
        dataset = datautils.load_testset_txt(data_path, use_structure=use_structure, seq_length=101)
        return dataset
       
    
    def __prepare_data__(self, data):
        inputs    = data['inputs'][:,:,:,:4]
        structure = data['inputs'][:,:,:,4:]
        structure = np.expand_dims(structure[:,:,:,0], axis=3)
        inputs    = np.concatenate([inputs, structure], axis=3)
        data['inputs']  = inputs
        return data

    def __to_sequence__(self, x):
        x1 = np.zeros_like(x[0,:,:1])
        for i in range(x1.shape[0]):
            # import pdb; pdb.set_trace()
            x1[i] = np.argmax(x[0,i,:4])
            # import pdb; pdb.set_trace()
        return x1

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        x = self.dataset['inputs'][index]
        # x = self.__to_sequence__(x)
        y = self.dataset['targets'][index]
        return x, y


    def __len__(self):
        return len(self.dataset['inputs'])

