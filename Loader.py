import os
from pyedflib import EdfReader
import keras
import re
import random
import numpy as np
from sklearn.model_selection import train_test_split

class Loader(keras.utils.Sequence):

    def __init__(self, path):
        """
        segment_size is the number of seconds allocated to each segment of the signal, in seconds
        """
        self.path = path
        self.shuffled_walk = [(r, d, f) for r, d, f in os.walk(path)]
        #random.shuffle(self.shuffled_walk)

    def __iter__(self):
        for root, dirs, files in self.shuffled_walk:
            for f in files:
                if re.match(".*\.npz$", f):
                    npz_path = os.path.join(root, f)
                    x, y = self.load_segments_from_file(npz_path)
                    x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.2)
                    yield (x_train, y_train, x_test, y_test)
    

    


    def __len__(self):
        return len(self.shuffled_walk)

    def split_signal(self, signal, sampling_frequency):
        return np.split(signal, self.segment_size*sampling_frequency)

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