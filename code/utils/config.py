# Path to the data directory
from pathlib import Path

import colorama

data_dir = Path("../../images/train")
data_dir_test = Path("../../images/test")
img_type = "*.jpeg"  # "*.png"
characters = ['z', 's', 'h', 'q', 'd', 'v', '2', '7', '8', 'x', 'y', '5', 'e', 'a', 'u', '4', 'k', 'n', 'm', 'c', 'p']
# Desired image dimensions
img_width = 130
img_height = 50
# Maximum length of any captcha in the dataset
max_length = 7

MODEL_FNAME_TRANING = "../output.traning2.model"
MODEL_FNAME = "../output2.model"

# Training config
batch_size = 16

epochs = 100
early_stopping_patience = 10

# we all love colored output, aren't we?
colorama.init()
