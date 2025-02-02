import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import random
from data_augmentation import apply_data_augmentation
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from config import IMAGE_DIR, ROI_DIR

class LiverDataset(Dataset):
    '''
    The class load, normalize and create dataset 
    '''
    def __init__(self, image_paths, roi_paths, transform=None):
        self.image_paths = image_paths
        self.roi_paths = roi_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        roi = self.load_roi(self.roi_paths[idx])

        if self.transform:
            image, roi = self.transform(image, roi)

        # Normalisation
        image = image / 255.0
        roi = roi / 255.0

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        roi = torch.tensor(roi, dtype=torch.float32).unsqueeze(0)
        return image, roi

    def load_roi(self, roi_path):
        # transform into binary mask the roi path (grayscale image)
        roi = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
        return roi

def load_image_paths():
    image_paths = [os.path.join(IMAGE_DIR, img_name) for img_name in os.listdir(image_dir)]
    return image_paths

def load_roi_paths():
    roi_paths = [os.path.join(ROI_DIR, roi_name) for roi_name in os.listdir(roi_dir)]
    return roi_paths

def split_dataset(image_paths, roi_paths, test_size=0.2):
    train_img_paths, test_img_paths, train_roi_paths, test_roi_paths = train_test_split(
        image_paths, roi_paths, test_size=test_size, random_state=42)
    return train_img_paths, test_img_paths, train_roi_paths, test_roi_paths


def get_dataloader(image_paths, roi_paths, batch_size, apply_augmentation=False):
    if apply_augmentation:
        transform = apply_data_augmentation
    else:
        transform = None

    dataset = LiverDataset(image_paths, roi_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def get_dataset(test_size=0.2, batch_size=32, apply_augmentation=False):
    image_paths = load_image_paths()
    roi_paths = load_roi_paths()

    train_img_paths, test_img_paths, train_roi_paths, test_roi_paths = split_dataset(
        image_paths, roi_paths, test_size=test_size)

    train_loader = get_dataloader(train_img_paths, train_roi_paths, batch_size, apply_augmentation)
    test_loader = get_dataloader(test_img_paths, test_roi_paths, batch_size, False)

    return train_loader, test_loader

if __name__ == "__main__":
    
    train_loader, test_loader = get_dataset(test_size=0.2, batch_size=32, apply_augmentation=True)
    
    for images, rois in train_loader:
        print(images.shape, rois.shape)
        # Place here the code to visualize or use the batches
