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

val_dataset = dataset_class[config.dataset_type](config.dataset_config, is_train=False)
valloader = torch.utils.data.dataloader.DataLoader(val_dataset, batch_size=32, num_workers=4, shuffle=True)
model = LITFSModel.load_from_checkpoint("/ssd_scratch/cvit/keshav/fs_ckpts/last_epoch.ckpt", config=config)
count = 0

import shutil
shutil.rmtree("/ssd_scratch/cvit/keshav/carla_val_vis/")

with torch.no_grad():
    for batch in tqdm(valloader):
        # img = val_dataset[i][0].permute(1, 2, 0).cpu().numpy()
        data = batch
        seg, img = data['contour'], data['img']
        for i in range(len(img)):
            img_ = img[i][None].repeat(64, 1, 1, 1).cuda()
            pred_segs = model.model.infer(img_)
            vis_imgs = model.visualize(pred_segs, img_)
            for j, img_ in enumerate(vis_imgs):
                cv2.imwrite(f"/ssd_scratch/cvit/keshav/carla_val_vis/{count}_{j}.png", img_)
            count += 1

