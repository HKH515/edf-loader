import os
import keras
import re
import random
import numpy as np
from sklearn.model_selection import train_test_split
from pyedflib import EdfReader

def samplify(arr, timestep):
    """
    gathers samples at timestep intervals, cuts off the last segment if it does not get filled
    """
    samples = []
    for i in range(0, arr.shape[0], timestep):
        sample = arr[i:i+timestep]
        if len(sample) == timestep:
            samples.append(sample)

    return np.array(samples)

class Loader:
    def __init__(self, path, x_channels, y_channels):
        """
        Parameters
        ----------
        path : str
            Path to edf folder
        x_channels : list of str
            List of channel names to deliver as part of the x_train and x_test portion of the dataset
        x_channels : list of str
            List of channel names to deliver as part of the y_train and y_test portion of the dataset
        """
        self.path = path
        if not os.path.exists(self.path):
            raise FileNotFoundError
        self.walk = [(r, d, f) for r, d, f in os.walk(path)]
        self.x_channels = x_channels
        self.y_channels = y_channels

    def load_1d(onehot=False):
        """
        Parameters
        ----------
        onehot : bool
                whether y_train and y_test should be converted to onehot encoded numpy vectors
        """
        pass

    def load_3d(timestep, onehot=False):
        """
        x_train and x_test arrays are split into 3 dimensions, along a batch, along a timestep, and along sample
        batch (is an ndarray): selection of timesteps, meant to be used for training a neural network.
        timestep (is an ndarray): number of samples to be gathered together to form a single timestep
        sample (is a number): a single datapoint

        segment_size is the number of seconds allocated to each segment of the signal, in seconds

        further explanation of the terms batch, and timestep, can be found here: https://machinelearningmastery.com/use-timesteps-lstm-networks-time-series-forecasting/

        Parameters
        ----------
        timestep : int
        onehot : bool
                whether y_train and y_test should be converted to onehot encoded numpy vectors
        """
        for root, dirs, files in self.walk:
            for f in files:
                try:
                    edf_path = os.path.join(root, f)
                    x, y = self.load_segments_from_file(edf_path)


                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
                    # remove dimensions that have the length 1
                    x_train = np.squeeze(x_train)
                    x_test = np.squeeze(x_test)
                    y_train = np.squeeze(y_train)
                    y_test = np.squeeze(y_test)

                    # for better labels, convert to int
                    y_train = y_train.astype(int)
                    y_test = y_test.astype(int)
                    #print(x_train.shape)
                    #print("----------------------------------------")
                    x_train = samplify(x_train, self.timestep)
                    x_test = samplify(x_test, self.timestep)
                    y_train = samplify(y_train, self.timestep)
                    y_test = samplify(y_test, self.timestep)
                    #print(x_train.shape)
                    #exit()
                    #x_test = self._batchify(x_test)
                    #y_train = self._batchify(y_train)
                    #y_test = self._batchify(y_test)
                    #x_train = np.squeeze(x_train)
                    #x_test = np.squeeze(x_test)
                    #y_train = np.squeeze(y_train)
                    #y_test = np.squeeze(y_test)
                    yield (x_train, x_test, y_train, y_test)
                except OSError:
                    continue

    def _train_test_dict(self, loaded_tuple):
        """
        Parameters
        ----------
        loaded_tuple : tuple
            output of _load_file()
        Returns
        -------
        dict
            {
                "x_train":
                {
                    "channel_name" : np.ndarray,
                    "other_channel_name": np.ndarray,
                    ...
                },
                "y_train":
                {
                    "label_channel_name" : np.ndarray,
                    "other_label_channel_name" : np.ndarray,
                    ... 
                },
                "x_test":
                {
                    "channel_name": np.ndarray,
                    "other_channel_name": np.ndarray,
                    ...
                },
                "y_test":
                {
                    "label_channel_name" : np.ndarray,
                    "other_label_channel_name" : np.ndarray,
                    ... 
                },
            }
        """

        x, y = loaded_tuple
        #todo: split x, y into the dict described in the docstring




    def _load_file(self, input_file):
        """
        Parameters
        ----------
        input_file : str
                    path of the file which to load
        Returns
        -------
        tuple
            (
                {
                    "channel_name" : np.ndarray,
                    "other_channel_name": np.ndarray,
                    ...
                },
                {
                    "label_channel_name" : np.ndarray,
                    "other_label_channel_name" : np.ndarray,
                    ... 
                },
            )
        """

        reader = EdfReader(input_file)
        channel_names = reader.getSignalLabels()
        channel_names_dict = {channel_names[i]:i for i in range(len(channel_names))}
        x_channels_to_process = set(channel_names).intersection(set(self.x_channels))
        y_channels_to_process = set(channel_names).intersection(set(self.y_channels))


        x_channel_data_dict = {channel: reader.readSignal(channel_names_dict[channel]) for channel in x_channels_to_process}
        y_channel_data_dict = {channel: reader.readSignal(channel_names_dict[channel]) for channel in y_channels_to_process}
        print(x_channel_data_dict)
        print(y_channel_data_dict)

        return (x_channel_data_dict, y_channel_data_dict)




if __name__ == "__main__":
    # this is a fabricated example and the variables and channels used do not reflect real world usage
    loader = Loader("/home/hannes/repos/EEG_classification/output", ["C3-A2", "O1-A2"], ["nFlow"])
    loader._load_file("/home/hannes/datasets/stanford_edfs/IS-RC/AL_10_021708.edf")
    #for i in loader:
    #    print([x.shape for x in i])