from modulefinder import test
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pyshred import DataManager, SHRED, SHREDEngine, DeviceConfig
import sys
import os
import pyshred
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def trajectory_gen(data_list,lags,sensors):
    if not isinstance(data_list, list):
        raise TypeError('data type of data_list must be a list')
    t_dim = data_list[0].shape[0]
    x_dim = data_list[0].shape[1]
    trajectories = np.zeros((t_dim-lags,len(sensors),lags))
    full_states = np.zeros((t_dim-lags,x_dim))
    
    for n, data_temp in enumerate(data_list):       
        for i in range(t_dim-lags):
            for j in range(len(sensors)):
                trajectories[i,j,:] = data_temp[i:i+lags,sensors[j]]
            full_states[i,:] = data_temp[i+lags,:]

        if n < 1:
            X = trajectories.copy()
            y = full_states.copy()
        
        else:
            X = np.vstack((X,trajectories))
            y = np.vstack((y,full_states))
    return X, y

def get_sensor_data(data_list, sensor_locations):
    if not isinstance(data_list, list):
        raise TypeError('data type of data_list must be a list')
    
    t_dim = data_list[0].shape[0]
    sensor_measurements_temp = np.zeros((t_dim, len(sensor_locations)))
    for n, data_temp in enumerate(data_list):
        for j, sensor in enumerate(sensor_locations):
            sensor_measurements_temp[:, j] = data_temp[:, sensor]
        if n < 1:
           sensor_measurements = sensor_measurements_temp.copy()
        else:
           sensor_measurements = np.vstack((sensor_measurements, sensor_measurements_temp))
    return sensor_measurements


def data_prepare(data_list, lags, sensors):
    sensor_measurements = get_sensor_data(data_list, sensors)
    scaler = MinMaxScaler()
    scaler = scaler.fit(sensor_measurements)

    X, y = trajectory_gen(data_list, lags, sensors)
    for i in range(X.shape[0]):
        X[i, :, :] = scaler.transform(X[i, :, :].T).T
    return X, y, scaler
    

     
    
    
class SHREDdata(torch.utils.data.Dataset):
    """
    PyTorch Dataset for time series sensor data and corresponding full-state measurements.

    Parameters
    ----------
    DATA: dictionary containg X and y
    
    X : torch.Tensor
        Input sensor sequences of shape (batch_size, lags, num_sensors).
    Y : torch.Tensor
        Target full-state measurements of shape (batch_size, state_dim).

    Attributes
    ----------
    X : torch.Tensor
        Sensor measurement sequences.
    Y : torch.Tensor
        Full-state target measurements.
    len : int
        Number of samples in the dataset.
    """

    def __init__(self, DATA):
        """
        Initialize the TimeSeriesDataset.

        Parameters
        ----------
        X : torch.Tensor
            Input sensor sequences of shape (batch_size, lags, num_sensors).
        Y : torch.Tensor
            Target full-state measurements of shape (batch_size, state_dim).
        """
        self.X = DATA['X']
        self.Y = DATA['y']
        self.len = self.X.shape[0]
        
    def __getitem__(self, index):
        """
        Get a single sample from the dataset.

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        tuple
            (sensor_sequence, target_state) pair.
        """
        return self.X[index], self.Y[index]

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples.
        """
        return self.len
     
    def split_data(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        generator = torch.Generator().manual_seed(seed)
        train, val, test = random_split(self, [train_ratio, val_ratio, test_ratio], generator=generator)

        train_data = {
            'X':self.X[train.indices],
            'y':self.Y[train.indices]
        }
        val_data = {
            'X':self.X[val.indices],
            'y':self.Y[val.indices]
        }
        test_data = {
            'X':self.X[test.indices],
            'y':self.Y[test.indices]
        }
        train_dataset = SHREDdata(train_data)
        val_dataset = SHREDdata(val_data)
        test_dataset = SHREDdata(test_data)

        return train_dataset,val_dataset,test_dataset
    
    
    