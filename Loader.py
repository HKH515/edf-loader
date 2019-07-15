import os
import keras
import re
import random
import numpy as np
from sklearn.model_selection import train_test_split
from pyedflib import EdfReader
import pandas as pd
pd.set_option('display.max_rows', 500)
import matplotlib.pyplot as plt

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

def samplify_df(df, timestep):
    """
    returns df, but with each column samplified
    """
    new_df = pd.DataFrame()
    for col in df:
        new_df[col] = samplify(df[col], timestep)
    return new_df

class Loader:
    def __init__(self, path, x_channels, y_channels):
        """
        Parameters
        ----------
        path : str
            Path to edf folder
        x_channels : list of str
            List of channel names to deliver as part of the x_train and x_test portion of the dataset
        y_channels : list of str
            List of channel names to deliver as part of the y_train and y_test portion of the dataset

        Note that the length of the arrays (i.e. same sampling rate and time) must be consistend across all channels!
        """
        self.path = path
        if not os.path.exists(self.path):
            raise FileNotFoundError
        self.walk = [(r, d, f) for r, d, f in os.walk(path)]
        self.x_channels = x_channels
        self.y_channels = y_channels

    def load(self, test_size=0.2):
        """
        Parameters
        ----------
        test_size : 
            either a floating point from 0..1, describing the percentage of samples to use for testing, or the number of samples to use for testing.
        """
        for root, dirs, files in self.walk:
            for f in files:
                try:
                    edf_path = os.path.join(root, f)
                    x, y = self._load_file(edf_path)
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
                    yield (x_train, x_test, y_train, y_test)
                except OSError as e:
                    print("Loader.py error: %s" % e)
                    continue

    #def load_3d(self, timestep, onehot=False):
    #    """
    #    x_train and x_test arrays are split into 3 dimensions, along a batch, along a timestep, and along sample
    #    batch (is an ndarray): selection of timesteps, meant to be used for training a neural network.
    #    timestep (is an ndarray): number of samples to be gathered together to form a single timestep
    #    sample (is a number): a single datapoint
    #
    #    segment_size is the number of seconds allocated to each segment of the signal, in seconds

    #    further explanation of the terms batch, and timestep, can be found here: https://machinelearningmastery.com/use-timesteps-lstm-networks-time-series-forecasting/

    #    Parameters
    #    ----------
    #    timestep : int
    #    onehot : bool
    #            whether y_train and y_test should be converted to onehot encoded numpy vectors
    #    """
    #    for root, dirs, files in self.walk:
    #        for f in files:
    #            try:
    #                edf_path = os.path.join(root, f)
    #                x, y = self._load_file(edf_path)
    #
    #                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    #                x_train = samplify_df(x_train, timestep)
    #                print(x_train)
    #                print("exiting...")
    #                exit()
    #                x_test = samplify_df(x_test, timestep)
    #                y_train = samplify_df(y_train, timestep)
    #                y_test = samplify_df(y_test, timestep)
    #                yield (x_train, x_test, y_train, y_test)
    #            except OSError as e:
    #                print(e)
    #                continue



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
                pd.DataFrame,
                pd.DataFrame
            )
        """

        reader = EdfReader(input_file)
        channel_names = reader.getSignalLabels()
        channel_names_dict = {channel_names[i]:i for i in range(len(channel_names))}
        x_channels_to_process = set(channel_names).intersection(set(self.x_channels))
        y_channels_to_process = set(channel_names).intersection(set(self.y_channels))


        x_channel_data_dict = {channel: reader.readSignal(channel_names_dict[channel]) for channel in x_channels_to_process}
        y_channel_data_dict = {channel: reader.readSignal(channel_names_dict[channel]) for channel in y_channels_to_process}

        x_df = pd.DataFrame(x_channel_data_dict)
        y_df = pd.DataFrame(y_channel_data_dict)

        return (x_df, y_df)

def max_item(items):
    maxv = 0
    maxk = None
    for k,v in items.items():
        if len(v) > maxv:
            maxv = len(v)
            maxk = k
    return maxk


if __name__ == "__main__":
    # this is a fabricated example and the variables and channels used do not reflect real world usage
    loader = Loader("/home/hannes/repos/edf-consister/output/", ["eog_l"], ["chin"])
    #ret = loader._load_file("/home/hannes/datasets/stanford_edfs/IS-RC/AL_10_021708.edf")
    #ret = loader._load_file("/home/hannes/repos/edf-consister/output/al_10_021708.edf")
    for x_train, x_test, y_train, y_test in loader.load(0.45):
        print(len(x_train))
        print(len(x_test))
        print(len(y_train))
        print(len(y_test))