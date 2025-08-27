import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from datasets_all.car_dataset import CarDataset
from datasets_all.idd_dataset import IDDDataset
from fs_model import LITFSModel
import yaml
from torch.utils.data import WeightedRandomSampler
import numpy as np

L.seed_everything(2024)
class Config:
    def __init__(self, config):
        self.config = config
        for k, v in self.config.items():
            self.__setattr__(k,  v)

config = "configs/red.yaml"
with open(config, "r") as f:
    config = Config(yaml.safe_load(f))

dataset_class = {"car" : CarDataset, "idd" : IDDDataset}

train_dataset = dataset_class[config.dataset_type](config.dataset_config)
val_dataset = dataset_class[config.dataset_type](config.dataset_config, is_train=False)


trainloader = torch.utils.data.dataloader.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=4)
valloader = torch.utils.data.dataloader.DataLoader(val_dataset, batch_size=config.batch_size, num_workers=4, shuffle=True)

model = LITFSModel(config)
# logger = WandbLogger(name="contour_run_nusc", project="fs_carla_diff", config=config)
logger = CSVLogger("logs", name=f"fs_{config.dataset_type}_{config.backbone}")

checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=config.ckpt_dir, filename=f"best_loss_{config.backbone}_{config.dataset_type}", mode='min')
checkpoint_callback2 = ModelCheckpoint(dirpath=config.ckpt_dir, filename=f'last_{config.backbone}_{config.dataset_type}', save_last=True, )

devices = [0, 1, 2, 3]
# devices = [0]
trainer = L.Trainer(devices=devices, max_epochs=100, callbacks=[checkpoint_callback, checkpoint_callback2], logger=[logger],\
    strategy='ddp_find_unused_parameters_true', detect_anomaly=True)
trainer.fit(model, trainloader, valloader)

# outputs = trainer.test(ckpt_path="last", dataloaders=valloader)
# outputs = trainer.test(ckpt_path="/ssd_scratch/cvit/keshav/fs_ckpts/last_epoch.ckpt", dataloaders=valloader)
# test_iou = outputs[0]['test_iou']