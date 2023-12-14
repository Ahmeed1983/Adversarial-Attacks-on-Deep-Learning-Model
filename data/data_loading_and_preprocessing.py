import os
import tarfile
import urllib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torchmetrics import Accuracy, Precision, F1Score, ConfusionMatrix


# Function to download and extract the dataset
def download_and_extract_dataset(dataset_url, dataset_path, tar_file_path):
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)
        print(f"Downloading the dataset from {dataset_url}...")
        urllib.request.urlretrieve(dataset_url, tar_file_path)
        print("Download complete!")
        print("Extracting the dataset...")
        with tarfile.open(tar_file_path, 'r') as tar_ref:
            tar_ref.extractall(path=dataset_path)
        print("Extraction complete!")
    else:
        print("Dataset already downloaded and extracted.")

# Data Preparation
dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
dataset_path = "data/images"
tar_file_path = "data/images.tar"
download_and_extract_dataset(dataset_url, dataset_path, tar_file_path)

# Define the augmentation and transformation steps
augmentations = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the dataset from the 'Images' directory and apply the transformations
images_dir = os.path.join(dataset_path, 'Images')
dataset = ImageFolder(root=images_dir, transform=augmentations)

# Split the dataset into a train and validation set
targets = [label for _, label in dataset.samples]
train_idx, val_idx = train_test_split(
    range(len(targets)),
    test_size=0.2,  # 20% for validation
    random_state=42,  # Seed for reproducibility
    stratify=targets
)

# Create Subset objects to be used as the train and validation datasets
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

# Create DataLoaders for the train and validation datasets
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
class_names = dataset.classes
