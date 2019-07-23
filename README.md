<p align="center">
    <img src="https://user-images.githubusercontent.com/5413669/61218523-baa9b200-a701-11e9-9856-fcfa39474466.png"/>
</p>

Python library to load channels in EDF files as a pandas dataframe, with support for train/test set splitting.

# Setup
```bash
pip3 install -r requirements.txt
```

# Usage
```python
import sys
sys.path.append("path_to_edfloader_folder")
from Loader import Loader, samplify
loader = Loader("/path/to/folder/of/edf/files/", ["eog_l", "eog_r"], ["spo2"])
x, y = loader.load()
```
**x** and **y** will be dictionaries with channel names as the keys. and dictionaries with sampling frequency and data key-value pairs.
```
x = {
    "eog_l": {"sampling_frequency": 100, "data":[...]},
    "eog_r": {"sampling_frequency": 100, "data":[...]}
}

y = {
    "spo2": {"sampling_frequency": 1, "data": [...]}
}
```

## Samplify:
```python
data = { 
    'eog':{
        'sampling_frequency':2, 
        'data':[1, 2 ,3 ,4, 5, 6]
    }
}
#data contains the channel 'eog' which has a
# 3-second recording of a 2Hz signal(6 samples)
#samplify will split the data into 1 second 
# rercordings
sampled_data = samplify(data, 1)
# sampled_data = {
#     'eog':{
#         'sampling_frequency':2,
#         'data':[[1,2],[3,4],[5,6]]
#     }
# }