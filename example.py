from Loader import Loader, samplify


loader = Loader("path/to/edf/folder", ["O1-A2"], ["SAO2"])


for x, y in loader.load():
    # do something with data here