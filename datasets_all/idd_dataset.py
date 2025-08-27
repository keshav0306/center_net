import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
import glob
import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET

class IDDDataset(Dataset):
    def __init__(self, dataset_path, is_train=True):
        self.dataset_path = dataset_path
        if(is_train):
            self.train_path = dataset_path + "/train.txt"
        else:
            self.train_path = dataset_path + "/val.txt"
        self.filenames = open(self.train_path, "r").readlines()
        self.filenames = [f.strip() for f in self.filenames]
        self.bbox_max = 50
        
    def __len__(self):
        return len(self.filenames)
    
    def get_labels(self, idx):
        gt_file = self.dataset_path + "Annotations/" + self.filenames[idx] + ".xml"
        tree = ET.parse(gt_file)
        root = tree.getroot()
        gt_obs = []
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            cls = obj.find("name").text
            if(cls == "motorcycle"):
                cls = 1
            elif(cls == "rider"):
                cls = 2
            else:
                continue
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            gt_obs.append([cls, xmin, ymin, xmax, ymax])
            if(len(gt_obs) == self.bbox_max):
                break

        return gt_obs

    def __getitem__(self, idx):
        img_path = self.dataset_path + "/JPEGImages/" + self.filenames[idx] + ".jpg"
        gt_labels = self.get_labels(idx)
        # print(bounding_box)
        # exit(0)
        img = cv2.imread(img_path)
        h, w = img.shape[0], img.shape[1]
        img = cv2.resize(img, (384, 384))
        img = img / 255
        img = img.transpose(2, 0, 1).astype(np.float32)
        for label in gt_labels:
            label[1] = label[1] / w
            label[2] = label[2] / h
            label[3] = label[3] / w
            label[4] = label[4] / h
            label[1:] = self.convert_to_center(label[1:])
        
        bbox_comp = np.zeros((self.bbox_max, 5))
        if(len(gt_labels)):
            bbox_comp[:len(gt_labels), :] = np.array(gt_labels)
        
        mask = np.zeros((self.bbox_max)).astype(np.int32)
        mask[:len(gt_labels)] = 1
        
        elem = {"image": img, "bounding_box": bbox_comp, "mask": mask}
        return elem
    
    def convert_to_center(self, bounding_box):
        x1, y1, x2, y2 = bounding_box
        return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
    
if __name__ == "__main__":
    # Example usage
    dataset = IDDDataset("/ssd_scratch/cvit/keshav/IDD_Detection/")
    idx = 1
    image, mask = dataset[idx]["image"], dataset[idx]["bounding_box"]
    print(image)
    print(mask)