import os
from pyedflib import EdfReader
import re
import random

class Loader:
    def __init__(self, segment_size, path):
        """
        segment_size is the number of seconds allocated to each segment of the signal
        """
        self.files = []
        self.segment_size = segment_size
        if os.path.isdir(path):
            self.get_pairs(path)
        else:
            raise FileNotFoundError()

    def get_pairs(self, path):
        for root, dirs, files in os.walk(path):
            for f in files:
                if re.match(".*\.edf$", f):
                    csv_filename = os.path.join(os.path.dirname(f), ".".join([os.path.splitext(os.path.basename(f))[0], "csv"]))
                    print(csv_filename)
                    if os.path.exists(os.path.join(root, csv_filename)):
                        self.files.append((os.path.join(root, f), os.path.join(root, csv_filename)))

    def split_signal(self, signal, sampling_frequency):

        return numpy.split(signal, self._no_of_samples_per_segment(sampling_frequency))

    def load_segments_from_file(self, input_file):
        # dict of signals, each signal is a list of numpy arrays, each numpy array is one segment (i.e. 30s of signal)
        signals_segments = {}
        reader = EdfReader(input_file)
        signal_count = reader.signals_in_file
        for signal_i in range(signal_count):
            signal = reader.readSignal(signal_i)
            split = self.split_signal(signal, reader.getSampleFrequency(signal_i))
            print(split)
    
    def generate_x_y_pairs(self):
        # dict of signals, each signal is a list of numpy arrays, where each numpy array is a segment
        files_signals_segments = {}
        for edf_file, csv_file in self.files:
            self.load_segments_from_file(edf_file)


    def train_test_split(self, test_size=0.2):
        random.shuffle(self.files)

    def _number_of_segments(self, signal, sampling_frequency):
        pass

    def _number_of_samples_per_segment(self, sampling_frequency):
        pass

    



if __name__ == "__main__":
    loader = Loader(30, "/home/hannes/Downloads/")
    loader.generate_x_y_pairs()