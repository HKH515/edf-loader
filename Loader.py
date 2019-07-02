import os
import keras
import re
import random
import numpy as np
from sklearn.model_selection import train_test_split

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

class Loader(keras.utils.Sequence):

    def __init__(self, path, timestep=50):
        """
        Reads in a folder `path`, and loads all files that have the extension .npz, these files are read as numpy arrays using np.load().
        These arrays will be split into x_train, x_test, y_train, y_test.

        x_train and x_test arrays are split into 3 dimensions, along a batch, along a timestep, and along sample
        batch (is an ndarray): selection of timesteps, meant to be used for training a neural network.
        timestep (is an ndarray): number of samples to be gathered together to form a single timestep
        sample (is a number): a single datapoint

        segment_size is the number of seconds allocated to each segment of the signal, in seconds

        further explanation of the terms batch, and timestep, can be found here: https://machinelearningmastery.com/use-timesteps-lstm-networks-time-series-forecasting/
        """
        self.path = path
        if not os.path.exists(self.path):
            raise FileNotFoundError
        self.walk = [(r, d, f) for r, d, f in os.walk(path)]
        self.timestep = timestep

    def __iter__(self):
        for root, dirs, files in self.walk:
            for f in files:
                if re.match(".*\.npz$", f):
                    npz_path = os.path.join(root, f)
                    x, y = self.load_segments_from_file(npz_path)


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

    def get_all(self):
        return [i for i in self]

    def __len__(self):
        return len(self.walk)

    def load_segments_from_file(self, input_file):
        # dict of signals, each signal is a list of numpy arrays, each numpy array is one segment (i.e. 30s of signal)
        data = np.load(input_file)
        x = data['x']
        y = data['y']
        return (x, y)
    




if __name__ == "__main__":
    loader = Loader("/home/hannes/repos/EEG_classification/output")
    for i in loader:
        print([x.shape for x in i])