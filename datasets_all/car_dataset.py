import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import glob
import os
import numpy as np
import cv2

class CarDataset(Dataset):
    def __init__(self, dataset_path):
        self.images = dataset_path + "/training_images/"
        self.label_file = dataset_path + "/training_labels.csv"
        self.labels = np.loadtxt(self.label_file, delimiter=",", skiprows=1, usecols=(1, 2, 3, 4))
        self.image_names = np.loadtxt(self.label_file, delimiter=",", skiprows=1, usecols=(0), dtype=str)
        self.bbox_max = 20
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.images + self.image_names[idx]
        bounding_box = self.labels[idx]
        img = cv2.imread(img_path)
        h, w = img.shape[0], img.shape[1]
        img = cv2.resize(img, (224, 224))
        img = img / 255
        img = img.transpose(2, 0, 1).astype(np.float32)
        bounding_box[0] = bounding_box[0] / w
        bounding_box[1] = bounding_box[1] / h
        bounding_box[2] = bounding_box[2] / w
        bounding_box[3] = bounding_box[3] / h
        bounding_box = np.array(self.convert_to_center(bounding_box)).astype(np.float32)
        
        elem = {"image": img, "bounding_box": bounding_box}
        return elem
    
    def convert_to_center(self, bounding_box):
        x1, y1, x2, y2 = bounding_box
        return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    
if __name__ == "__main__":
    # Example usage
    dataset = CarDataset("/ssd_scratch/cvit/keshav/data/")
    image, mask = dataset[0]
    print(image)
    print(mask)
