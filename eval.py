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
valloader = torch.utils.data.dataloader.DataLoader(val_dataset, batch_size=128, num_workers=16, shuffle=True)
model = LITFSModel.load_from_checkpoint("/ssd_scratch/cvit/keshav/fs_ckpts/last_epoch.ckpt", config=config)
count = 0
final_metrics = {"iou": 0, "count": 0}

with torch.no_grad():
    for batch in tqdm(valloader):
        # img = val_dataset[i][0].permute(1, 2, 0).cpu().numpy()
        data = batch
        seg, img = data['mask'], data['img'].cuda()
        valid = data['valid']
        seg = seg[valid]
        img = img[valid]
        pred_contours = model.model.infer(img).cpu().numpy()
        pred_contours = ((pred_contours + 1)/2 * 256).astype(np.int32)
        masks = []
        for i in range(len(img)):
            mask = np.zeros((img[i].shape[1], img[i].shape[2]))
            cv2.drawContours(mask, [pred_contours[i].squeeze()], -1, 1, -1)
            masks.append(mask)
            # cv2.imwrite(f"masks/{i}.png", mask)
        masks = np.array(masks)
        metrics = model.compute_metrics(masks, seg.cpu().numpy())
        for k, v in final_metrics.items():
            if(k == "count"):
                continue
            final_metrics[k] = (final_metrics[k] * final_metrics['count'] + metrics[k]) / (final_metrics['count'] + len(masks))
        final_metrics["count"] += len(masks)
        print(metrics)
        print(final_metrics)

print(final_metrics)