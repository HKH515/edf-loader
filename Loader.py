import os
import re
import random
import numpy as np
from sklearn.model_selection import train_test_split
from pyedflib import EdfReader
import pandas as pd
pd.set_option('display.max_rows', 500)
import sys
import matplotlib.pyplot as plt

def samplify(channel_dict, seconds):
    """
    Parameters
    ----
        channel_dict : Similar to what is returned from load
        seconds : int 
    Returns
        Channel dict similar to what is returned from load except
        the data fields are split up into arrays of length sampling_frequency * seconds
    ---- 
    This function drops the last (data % step) segments ! 
    ---
    """
    ret_dict = {}
    for i in channel_dict:

        #find how many datapoints are in a <seconds> duration recording
        step = channel_dict[i]['sampling_frequency'] * seconds
        data = channel_dict[i]['data']
        # drop the last 30 seconds...
        data = data[: len(data) - (len(data) % step )   ]
        data = np.split(data, len(data)/step )
        ret_dict[i] = {'sampling_frequency': channel_dict[i]['sampling_frequency'], 'data': data}

    return ret_dict


def samplify_df(df, timestep):
    """ 
    This has been deprecated, as wer are no longer using dataframes
    ---
    """
    raise(Exception('Deprecated'))

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

    def load(self, return_missing=False):
        """
        Parameters
        ----------
            return_missing : bool
                    if true, returns triple (x, y, missing), 
                    where missing is true if the returned 
                    batch is loaded from a file that is 
                    missing one or more of the requested
                    channels.
        Returns
        ----------
            tuple = (
                x data, y data
            )
        """
        for root, dirs, files in self.walk:
            for f in files:
                try:
                    edf_path = os.path.join(root, f)
                    if return_missing:
                        x, y, missing = self._load_file(edf_path, return_missing=return_missing)
                        yield (x, y, missing)
                    else:
                        x, y = self._load_file(edf_path)
                        yield (x, y)
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



    def _load_file(self, input_file, return_missing=False):
        """
        Parameters
        ----------
        input_file : str
                    path of the file which to load
        return_missing : bool
                    if true, returns triple (x, y, missing)
                    where missing is true if input_file does 
                    contain one or more of the requested channels.
        Returns
        -------
        tuple (
            x : {'channelname': {'sampling_frequency': 100, 'data': np.array}, ... }
            y : {'channelname': {'sampling_frequency': 100, 'data': np.array}, ... }
        )
        """

        reader = EdfReader(input_file)
        channel_names = reader.getSignalLabels()
        channel_names_dict = {channel_names[i]:i for i in range(len(channel_names))}
        x_channels_to_process = set(channel_names).intersection(set(self.x_channels))
        y_channels_to_process = set(channel_names).intersection(set(self.y_channels))

        x_not_present = set(self.x_channels).difference(set(self.x_channels).intersection(set(channel_names)))
        y_not_present = set(self.y_channels).difference(set(self.y_channels).intersection(set(channel_names)))
        not_present = x_not_present.union(y_not_present)

        #return this with x and y to detect missing channels.
        missing = len(not_present) != 0
        #report missing channels 
        if missing:
            not_present = [i for i in not_present]
            print(f'File \'{input_file}\' does not have channels: {not_present}')
        
        
        # Create data dictionaries 
        x_channel_data_dict = {} 
        for i in x_channels_to_process:
            x_channel_data_dict[i] = {
                "sampling_frequency": reader.getSampleFrequency(channel_names_dict[i]),
                "data": reader.readSignal(channel_names_dict[i])
            }
        
        y_channel_data_dict = {} 
        for i in y_channels_to_process:
            y_channel_data_dict[i] = {
                "sampling_frequency": reader.getSampleFrequency(channel_names_dict[i]),
                "data": reader.readSignal(channel_names_dict[i])
            }
        
        if return_missing: 
            return x_channel_data_dict, y_channel_data_dict, missing
        else:
            return x_channel_data_dict, y_channel_data_dict

        # create data dictionaries 
        # x_channel_data_dict = {
        #     channel: reader.readSignal(channel_names_dict[channel]) 
        #               for channel in x_channels_to_process}
        # y_channel_data_dict = {
        #     channel: reader.readSignal(channel_names_dict[channel])
        #               for channel in y_channels_to_process}
        
        # # extract sampling frequencies.
        # x_sfs = {f"{channel}_sf": [reader.getSampleFrequency(channel_names_dict[channel])] for channel in x_channels_to_process }
        # y_sfs = {f"{channel}_sf": [reader.getSampleFrequency(channel_names_dict[channel])] for channel in y_channels_to_process }

        # # concat the data streams and the sampling frequencies
        # x_df = pd.concat([pd.DataFrame(x_channel_data_dict), pd.DataFrame(x_sfs)], axis=1)
        # y_df = pd.concat([pd.DataFrame(y_channel_data_dict), pd.DataFrame(y_sfs)], axis=1)

        # print(x_df)

        # return (x_df, y_df)


def max_item(items):
    maxv = 0
    maxk = None
    for k,v in items.items():
        if len(v) > maxv:
            maxv = len(v)
            maxk = k
    return maxk


if __name__ == "__main__":
    location = sys.argv[1]
    # this is a fabricated example and the variables and channels used do not reflect real world usage
    loader = Loader(location, ['EEG Fpz-Cz' ], ['EEG Fpz-Cz'])
    #ret = loader._load_file("/home/hannes/datasets/stanford_edfs/IS-RC/AL_10_021708.edf")
    #ret = loader._load_file("/home/hannes/repos/edf-consister/output/al_10_021708.edf")


    fig = plt.figure() 
    ax = fig.add_subplot(111)
    

    for x_train, x_test in loader.load():
        x_train = samplify(x_train, 30)
        for i in x_train: 
            print(f'{i}:\t{len(x_train[i]["data"])}\t{x_train[i]["sampling_frequency"]}')