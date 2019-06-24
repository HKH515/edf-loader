import os
from pyedflib import EdfReader
import keras
import numpy
import re
import random

class Loader(keras.utils.Sequence):

    def __init__(self, path, batch_size, segment_size=30):
        """
        segment_size is the number of seconds allocated to each segment of the signal, in seconds
        """
        self.path = path
        self.batch_size = batch_size
        self.segment_size = segment_size
        self.shuffled_walk = [(r, d, f) for r, d, f in os.walk(path)]
        random.shuffle(self.shuffled_walk)

    def __iter__(self):
        batch_x = []
        batch_y = []
        for root, dirs, files in self.shuffled_walk:
            for f in files:
                if re.match(".*\.edf$", f):
                    csv_filename = os.path.join(os.path.dirname(f), ".".join([os.path.splitext(os.path.basename(f))[0], "csv"]))
                    print(csv_filename)
                    if os.path.exists(os.path.join(root, csv_filename)):
                        # if both a .edf and a .csv file exist, then process
                        edf_path = os.path.join(root, f)
                        csv_path = os.path.join(root, csv_filename)

                        edf_signal_segments = self.load_segments_from_file(edf_path)
                        i = 0
                        for signal in edf_signal_segments:
                            print(signal.shape)
                        yield 1

    def __len__(self):
        assert len(self.shuffled_walk) % self.batch_size == 0
        # number of segments over all files, rounded upwards
        samples_per_segment = -(-len(self.shuffled_walk)//self.batch_size)
        print("samples in segment:" % samples_per_segment)
        return len(self.shuffled_walk) / samples_per_segment

    def split_signal(self, signal, sampling_frequency):
        return numpy.split(signal, self.segment_size*sampling_frequency)

    def load_segments_from_file(self, input_file):
        # dict of signals, each signal is a list of numpy arrays, each numpy array is one segment (i.e. 30s of signal)
        signals_segments = {}
        reader = EdfReader(input_file)
        signal_count = reader.signals_in_file
        for signal_i in range(signal_count):
            signal = reader.readSignal(signal_i)
            split = self.split_signal(signal, reader.getSampleFrequency(signal_i))
            signals_segments[reader.getLabel(signal_i)] = split
            print(split)
        return signals_segments
    

    



if __name__ == "__main__":
    loader = Loader("/home/hannes/Downloads/", 1, 30)
    for i in loader:
        print(i)