import torch
import lightning as L
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from diffusers import DDPMScheduler
import argparse
from lightning.pytorch.loggers import WandbLogger
from diffusers import UNet2DModel
from model import Model
from scipy.optimize import linear_sum_assignment
from transformers import  AutoImageProcessor, \
    EfficientNetModel, SwinModel, \
         ViTModel, MobileNetV2Model
    
class GeneralEncoder(nn.Module):
    def __init__(self, backbone = 'resnet18', pretrained = True, num_images=1, init_ch=3):
        super(GeneralEncoder, self).__init__()
        print("inside general encoder class")
        self.backbone = backbone
        # breakpoint()
        if 'resnet' in backbone:
            self.img_preprocessor = None
            self.encoder = ResNetEncoder(backbone=backbone,
                                         pretrained=pretrained,
                                         num_images = num_images,
                                         init_ch=init_ch)
            self.encoder_dims = 512
        elif backbone == 'efficientnet':
            self.img_preprocessor = AutoImageProcessor.from_pretrained("google/efficientnet-b0")
            self.encoder = EfficientNetModel.from_pretrained("google/efficientnet-b0") 
            self.encoder_dims = 1280
        elif backbone == 'swinmodel':
            self.img_preprocessor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            self.encoder = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            self.encoder_dims = 768
        elif backbone == 'vit':
            self.img_preprocessor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            self.encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
            self.encoder_dims = 768
        elif backbone == 'mobilenet':
            self.encoder_dims = 1280
            self.img_preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
            self.encoder = MobileNetV2Model.from_pretrained("google/mobilenet_v2_1.0_224")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def forward(self, x):
        if 'resnet' in self.backbone:
            # print("in enc forward")
            # breakpoint()
            return self.encoder(x)
        # breakpoint()
        device = x.device
        x = self.img_preprocessor(x, return_tensors = 'pt')
        pixel_values = x['pixel_values'].to(device)
        enc_output = self.encoder(pixel_values=pixel_values)
        outputs = enc_output.last_hidden_state
        
        if self.backbone == 'vit':
            # reshaped_tensor.permute(0, 2, 1)[:,:,1:].reshape(-1, 768, 7, 7)
            reshaped_tensor = outputs.permute(0, 2, 1)[:, :, 1:].reshape(-1, 768, 14, 14)
            return reshaped_tensor
        
        if self.backbone == 'swinmodel':
            # breakpoint()
            reshaped_tensor = outputs.permute(0, 2, 1).reshape(-1, 768, 7, 7)
            return reshaped_tensor
        
        return outputs
            
        
class ResNetEncoder(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, num_images=1, init_ch=3):
        super(ResNetEncoder, self).__init__()
        
        # Load the pre-trained ResNet model
        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
        elif backbone == 'resnet152':
            self.model = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        if(num_images > 1):
            self.model.conv1 = nn.Conv2d(init_ch*num_images, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(self.model.conv1.weight.device)
        self.layer0 = nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool)
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4

    def forward(self, x):
        # Forward pass through each ResNet block
        outputs = {}
        x0 = self.layer0(x)  # First downsample: output after conv1, bn1, relu, and maxpool
        x1 = self.layer1(x0)  # Second downsample: layer1
        x2 = self.layer2(x1)  # Third downsample: layer2
        x3 = self.layer3(x2)  # Fourth downsample: layer3
        x4 = self.layer4(x3)  # Final downsample: layer4

        outputs[0], outputs[1], outputs[2], outputs[3], outputs[4] = x0, x1, x2, x3, x4
        # Return intermediate feature maps
        # breakpoint()
        return outputs[3] #downstream, only 4 is being used
    
class ResNetEncoderModified(nn.Module):
    def __init__(self, backbone='resnet18', pretrained=True, num_images=1, init_ch=3):
        super(ResNetEncoderModified, self).__init__()
        
        # Load the pre-trained ResNet model
        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.model = models.resnet101(pretrained=pretrained)
        elif backbone == 'resnet152':
            self.model = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        if(num_images > 1):
            self.model.conv1 = nn.Conv2d(init_ch*num_images, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(self.model.conv1.weight.device)
        self.layer0 = nn.Sequential(self.model.conv1, self.model.bn1, self.model.relu, self.model.maxpool)
        self.layer1 = self.model.layer1
        self.layer2 = self.model.layer2
        self.layer3 = self.model.layer3
        self.layer4 = self.model.layer4
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.deform_conv = DeformConv(128, k=3, padding=1, spatial_size=48)
        self.deform_conv2 = DeformConv(128, k=3, padding=1, spatial_size=48)

    def forward(self, x):
        # Forward pass through each ResNet block
        outputs = {}
        x0 = self.layer0(x)  # First downsample: output after conv1, bn1, relu, and maxpool
        x1 = self.layer1(x0)  # Second downsample: layer1
        x2 = self.layer2(x1)  # Third downsample: layer2
        x3 = self.layer3(x2)  # Fourth downsample: layer3
        x4 = self.layer4(x3)  # Final downsample: layer4

        outputs[0], outputs[1], outputs[2], outputs[3], outputs[4] = x0, x1, x2, x3, x4
        # x5 = self.up(x4)
        # x5 = F.relu(self.conv(x5) + x3)
        # x6 = self.up(x5)
        # x6 = self.conv2(x6) + x2
        
        x2 = self.deform_conv(x2)
        x2 = F.relu(x2)
        x2 = self.deform_conv2(x2)
        
        # Return intermediate feature maps
        # breakpoint()
        return x2 #downstream, only 4 is being used
    
class DeformConv(nn.Module):
    def __init__(self, init_channels, k, padding, spatial_size=24):
        super().__init__()
        self.k = k
        self.padding = padding
        self.init_channels = init_channels
        self.offset_conv = nn.Conv2d(init_channels, k**2 * 2, kernel_size=k, padding=padding)
        self.offset_conv.weight.data.fill_(0)
        self.offset_conv.bias.data.fill_(0)
        self.weight_conv = nn.Conv2d(init_channels, 1, kernel_size=k, padding=padding, bias=False)
        base_points_x = torch.linspace(-1, 1, spatial_size).to(torch.float32)[None].repeat(spatial_size, 1)
        base_points_y = torch.linspace(-1, 1, spatial_size).to(torch.float32)[:, None].repeat(1, spatial_size)
        self.base_points = torch.stack([base_points_x, base_points_y], dim=0)
    
    def forward(self, x):
        offsets = self.offset_conv(x).reshape(x.shape[0], 2, self.k * self.k, x.shape[2], x.shape[3])
        offsets = offsets + self.base_points[None, :, None].to(offsets)
        weight_conv = self.weight_conv(x)
        kernel_list = []
        for i in range(self.k ** 2):
            kernel_list.append(F.grid_sample(x, offsets[:, :, i].permute(0, 2, 3, 1))) # (b, c, h, w)
        
        kernel_list = torch.stack(kernel_list, dim=1).permute(0, 2, 3, 4, 1) # (b, c, h, w, k**2)
        output = (self.weight_conv.weight[0].reshape(1, self.init_channels, -1)[:, :, None, None] * kernel_list).sum(-1) # sum or mean?
        
        return output

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.q = nn.Linear(config.query_dim, config.query_dim)
        self.k = nn.Linear(config.query_dim, config.query_dim)
        self.v = nn.Linear(config.query_dim, config.query_dim)
        assert config.attention_emb_dim % config.mha_heads == 0, "mha_heads must be divisible by attention_emb_dim"
        self.mha = nn.MultiheadAttention(config.attention_emb_dim, config.mha_heads, batch_first=True)
        self.out_linear = nn.Linear(config.attention_emb_dim, config.query_dim)
    
    def forward(self, q, k, v, return_attn_maps=False):
        out, attn_maps = self.mha(self.q(q), self.k(k), self.v(v), need_weights=return_attn_maps)
        # print(len(out), out[0].shape, out[1].shape)
        out = self.out_linear(out)
        if(return_attn_maps):
            return out, attn_maps
        return out

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm1 = nn.LayerNorm(config.query_dim)
        self.norm2 = nn.LayerNorm(config.query_dim)
        # ca and sa block
        self.pos = nn.Parameter(torch.randn(1, config.num_contour_points + 1, config.query_dim))
        self.sa = Attention(config)
        self.ca = Attention(config)
        self.ff1 = nn.Linear(config.query_dim, 2*config.query_dim)
        self.ff2 = nn.Linear(2*config.query_dim, config.query_dim)
        
    def forward(self, queries, img_feats, return_attn_maps=False):
        queries = self.norm1(queries) + self.pos
        queries_new = self.sa(queries, queries, queries)
        queries = queries_new + queries
        
        queries = self.norm2(queries)
        if(return_attn_maps):
            queries_new, attn_maps = self.ca(queries, img_feats, img_feats, return_attn_maps=return_attn_maps)
        else:
            queries_new = self.ca(queries, img_feats, img_feats, return_attn_maps=return_attn_maps)
        queries = queries_new + queries
        queries = self.ff2(F.relu(self.ff1(queries))) + queries
        if(return_attn_maps):
            return queries, attn_maps
        return queries
    
def fourier_embedding(x, D):
    # freqs = torch.tensor([2**i for i in range(D // 2)], dtype=torch.float32).to(x.device)[None]
    freqs = torch.tensor([i+1 for i in range(D // 2)], dtype=torch.float32).to(x.device)[None]
    emb_sin = torch.sin(freqs * x)
    emb_cos = torch.cos(freqs * x)
    embedding = torch.cat([emb_sin, emb_cos], dim=-1)
    
    return embedding

class CenterNet(nn.Module):
    def __init__(self, config):
        super(CenterNet, self).__init__()
        self.backbone = ResNetEncoderModified(backbone=config.backbone, pretrained=True)
        self.num_classes = 2
        # self.cls_head = nn.Conv2d(256, self.num_classes + 1, kernel_size=1)
        self.cls_head = nn.Conv2d(128, self.num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(128, 4, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x) # (b, 512, 7, 7)

        cls_scores = self.cls_head(x)
        reg_scores = self.reg_head(x)
        
        return cls_scores, reg_scores
    
    def generate_gaussian_mask(self, bounding_box, cls, output, sigma=1):
        h, w = output.shape[1:]
        for i in range(len(bounding_box)):
            xc, yc = bounding_box[i, :2]
            x_diff = torch.arange(w).to(torch.float32)[None].repeat(h, 1).to(output) - xc
            y_diff = torch.arange(h).to(torch.float32)[:, None].repeat(1, w).to(output) - yc
            gauss_field = torch.exp(-1 * (x_diff ** 2 + y_diff ** 2) / (2 * sigma ** 2))
            output[cls[i]] = torch.max(output[cls[i]], gauss_field)
    
    def compute_loss(self, batch, is_train=True):
        imgs = batch['image']
        mask = batch['mask'].to(torch.bool)
        bounding_box = batch['bounding_box']

        gt_cls = bounding_box[:, :, 0].to(torch.long)
        bounding_box = bounding_box[:, :, 1:].to(torch.float32)

        cls_scores, reg_scores = self(imgs)

        feat_map_size = cls_scores.shape[2]
        gt_map_reg = torch.zeros_like(reg_scores).permute(0, 2, 3, 1)
        gt_map_cls = torch.zeros_like(cls_scores)[:, 0].to(torch.long)
        gt_map_cls_gauss = torch.zeros_like(cls_scores)
        reg_scores = reg_scores.permute(0, 2, 3, 1)
        
        for i in range(cls_scores.shape[0]):
            if(mask[i].sum() == 0):
                continue
            this_bounding_box = bounding_box[i][mask[i]] * feat_map_size
            disc_bounding_box = this_bounding_box.to(torch.long)
            self.generate_gaussian_mask(disc_bounding_box, gt_cls[i][mask[i]] - 1, gt_map_cls_gauss[i])
            gt_map_cls[i, disc_bounding_box[:, 1], disc_bounding_box[:, 0]] = gt_cls[i][mask[i]]
            to_regress = (this_bounding_box - disc_bounding_box) / feat_map_size
            to_regress[:, 2:] = this_bounding_box[:, 2:] / feat_map_size
            gt_map_reg[i][disc_bounding_box[:, 1], disc_bounding_box[:, 0], :] = to_regress

        # cls_loss = F.cross_entropy(cls_scores, gt_map_cls_gauss)
        # cls_loss = F.binary_cross_entropy_with_logits(cls_scores, gt_map_cls_gauss)
        cls_loss = self.focal_loss(cls_scores, gt_map_cls_gauss)
        bbox_loss = F.mse_loss(reg_scores[(gt_map_cls != 0)], gt_map_reg[(gt_map_cls[:] != 0)])
        
        loss = cls_loss + bbox_loss
        return {"loss" : loss}
    
    def focal_loss(self, cls_scores, gt_map_cls_gauss, beta=4, alpha=2):
        # cls_scores = F.softmax(cls_scores, dim=1)
        cls_scores = F.sigmoid(cls_scores)
        # cls_scores = torch.clamp(cls_scores, 1e-7, 1 - 1e-7)
        pt = torch.where(gt_map_cls_gauss == 1, cls_scores, 1 - cls_scores)
        weight = torch.where(gt_map_cls_gauss == 1, 1, 1 - gt_map_cls_gauss)
        loss = -1 * ((1 - pt) ** alpha) * torch.log(pt) * (weight ** beta)
        return loss.mean()
    
    def validate(self, batch):
        output = self.compute_loss(batch, is_train=False)
        return output
    
    def infer(self, img):
        cls, bbox = self(img)
        bounding_boxes = []
        cls_heatmaps = []
        # cls = F.softmax(cls, dim=1)
        for i in range(cls.shape[0]):
            this_img_boxes = []
            cls_heatmaps.append(torch.max(F.sigmoid(cls[i]), dim=0)[0].cpu().numpy())
            # fg_ind = cls[i, 0] < 0.5
            fg_ind = torch.max(cls[i], dim=0)[0] > 0
            feat_map_size = cls.shape[2]
            x_val = torch.arange(feat_map_size).to(torch.long).to(cls.device)[None, :].repeat(feat_map_size, 1)
            y_val = torch.arange(feat_map_size).to(torch.long).to(cls.device)[:, None].repeat(1, feat_map_size)
            
            fg_coords = torch.stack([x_val, y_val], dim=-1)
            fg_coords = fg_coords[fg_ind]
            fg_bbox = bbox[i].permute(1, 2, 0)[fg_ind]
            fg_bbox[:, :2] += fg_coords
            fg_bbox[:, :2] = fg_bbox[:, :2] / feat_map_size
            for box in fg_bbox:
                this_img_boxes.append(box.cpu().numpy())
            bounding_boxes.append(np.array(this_img_boxes))
        
        return {"bounding_boxes" : bounding_boxes, "cls_heatmaps" : cls_heatmaps}

class LITFSModel(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = CenterNet(config)
        
    def training_step(self, batch, idx):
        # compute the training loss and log it to wandb and return it as well to update the model
        # imgs = self.model.debug(batch)
        output = self.model.compute_loss(batch)
        train_loss = output['loss']
        
        self.log("train_loss", train_loss, sync_dist=True, prog_bar=True)
        return train_loss
    
    def validation_step(self, batch, idx):
        # log the validation_loss, visualization images to wandb
        data = batch
        output = self.model.validate(data)
        
        self.log("val_loss", output['loss'], sync_dist=True, prog_bar=True)
        
        if(idx == 0):
            pred_bbox = self.model.infer(data['image'])
            vis_imgs = self.visualize(pred_bbox, data['image'])
            for i, img in enumerate(vis_imgs):
                cv2.imwrite(f"vis/{i}.png", img)
    
    def test_step(self, batch, idx):
        metrics = self.eval_batch(batch, idx)
        self.log("test_iou", metrics['iou']/len(batch['masks']), sync_dist=True, prog_bar=True)
        return metrics['iou']
    
    def visualize(self, pred_bbox, img):
        imgs = img.permute(0, 2, 3, 1).cpu().numpy() * 255
        pred_bounding_boxes = pred_bbox['bounding_boxes']
        pred_cls_heatmaps = pred_bbox['cls_heatmaps']
        all_vis = []
        for i, img in enumerate(imgs):
            img = img.astype(np.uint8)
            this_img_boxes = pred_bounding_boxes[i] * img.shape[0]
            this_img_boxes = this_img_boxes.astype(np.int32)
            this_img_cls_heatmap = np.tile(pred_cls_heatmaps[i][..., None], (1, 1, 3)) * 255
            this_img_cls_heatmap = this_img_cls_heatmap.astype(np.uint8)
            this_img_cls_heatmap = cv2.resize(this_img_cls_heatmap, (img.shape[1], img.shape[0]))
            img = np.ascontiguousarray(img)
            for box in this_img_boxes:
                l, t, r, b = box[0] - box[2] // 2, box[1] - box[3] // 2, box[0] + box[2] // 2, box[1] + box[3] // 2
                cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)
            
            vis_img = np.hstack([img, this_img_cls_heatmap])
            all_vis.append(vis_img)
            
        return all_vis
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        return optimizer