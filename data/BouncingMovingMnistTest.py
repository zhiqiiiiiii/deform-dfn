# adapted from https://github.com/emansim/unsupervised-videos
# DataHandler for different types of datasets

from __future__ import division
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class BouncingMnistTestDataset(Dataset):
    """Data Handler that creates Bouncing MNIST dataset on the fly."""
    def __init__(self, dataset='./data/mnist_test_seq.npy'):
        self.data_ = np.load(dataset)[..., np.newaxis].transpose(1,4,2,3,0)
        self.data_ = self.data_[0:1000,:,:,:,:]
        self.data_ = self.data_.astype(np.float32) / 255

        self.dataset_size_ = self.data_.shape[0]
        self.num_channels_ = self.data_.shape[1]
        self.image_size_ = self.data_.shape[2]
        self.frame_size_ = self.image_size_ ** 2

        
    def __len__(self):
        return self.dataset_size_
    
    def __getitem__(self, idx):
        data = self.data_[idx]
        return data