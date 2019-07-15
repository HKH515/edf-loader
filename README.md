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
from Loader import Loader
loader = Loader("/path/to/folder/of/edf/files/", ["eog_l", "eog_r"], ["spo2"])
for x_train, x_test, y_train, y_test in loader.load(test_size=0.3):
    # use data for ML, etc.
```