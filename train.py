import os
import pandas as pd
from PIL import Image
from logger import logger
import utils as utils

import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
import torchvision.transforms as transforms


# Neural Networ Predefined Parameters
params_model = {
    "shape_in": (3, 96, 96),
    "initial_filters": 8,
    "num_fc1": 100,
    "dropout_rate": 0.25,
    "num_classes": 2}

# fix torch random seed
torch.manual_seed(0)

logger.info(f"Creating A Custom Dataset")
class pytorch_data(Dataset):

    def __init__(self, data_dir, transform, data_type="train"):

        # Get Image File Names
        cdm_data = os.path.join(data_dir, data_type)  # directory of files
        # get list of images in that directory
        file_names = os.listdir(cdm_data)
        # get the full path to images
        self.full_filenames = [os.path.join(cdm_data, f) for f in file_names]

        # Get Labels
        labels_data = os.path.join(data_dir, "train_labels.csv")
        labels_df = pd.read_csv(labels_data)
        labels_df.set_index("id", inplace=True)  # set data frame index to id
        # obtained labels from df
        self.labels = [labels_df.loc[filename[:-4]].values[0]
                       for filename in file_names]
        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)  # size of dataset

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        image = Image.open(self.full_filenames[idx])  # Open Image with PIL
        image = self.transform(image)  # Apply Specific Transformation to Image
        return image, self.labels[idx]

# Creating transformer to convert a PIL image into PyTorch tensor
data_transformer = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((46,46))])

# dataset directory to create a custom dataset
data_dir = os.path.join(
    os.path.abspath(
        os.getcwd()), 'histopathologic-cancer-detection')

img_dataset = pytorch_data(data_dir, data_transformer, "train") # Histopathalogic images

# load an example tensor
img, label = img_dataset[10]

logger.info(f"Splitting The Dataset")

len_img=len(img_dataset)
len_train=int(0.8*len_img)
len_val=len_img-len_train

# Split Pytorch tensor
train_ts, val_ts = random_split(img_dataset,
                                [len_train, len_val])  # random split 80/20

logger.info(f"Transforming The Data")
