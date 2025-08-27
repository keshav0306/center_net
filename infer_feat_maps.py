import torch
import numpy as np
import cv2
import lightning as L
from tqdm import tqdm
from datasets_all.fs_dataset import FSDataset
from datasets_all.nusc_dataset import NuScenesDataset
import yaml
from fs_model import LITFSModel

class Config:
    def __init__(self, config):
        self.config = config
        for k, v in self.config.items():
            self.__setattr__(k,  v)

config = "configs/red.yaml"

with open(config, "r") as f:
    config = Config(yaml.safe_load(f))
    
dataset_class = {"nuscenes" : NuScenesDataset, "carla" : FSDataset}

val_dataset = dataset_class[config.dataset_type](config.dataset_config, is_train=True)
valloader = torch.utils.data.dataloader.DataLoader(val_dataset, batch_size=128, num_workers=4)
model = LITFSModel.load_from_checkpoint("/ssd_scratch/cvit/keshav/fs_ckpts/best_loss_resnet18_nuscenes.ckpt", config=config)
count = 0

with torch.no_grad():
    for batch in tqdm(valloader):
        # img = val_dataset[i][0].permute(1, 2, 0).cpu().numpy()
        data = batch
        seg, img = data['contour'], data['img'].cuda()
        pred_feats = model.model.get_feat_maps(img)
        for i in range(len(img)):
            path = data['img_path'][i].replace('/samples/CAM_FRONT/', '/feat_maps/').replace(".jpg", ".npz")
            np.savez(path, feat=pred_feats[i].cpu().numpy())