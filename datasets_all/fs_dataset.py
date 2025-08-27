import numpy as np
import cv2
import torch
import sys
import tqdm
import matplotlib.pyplot as plt
sys.path.append("/home2/keshav06/LAV/")
import os
from basic_dataset import BasicDataset
from scipy.interpolate import interp1d

def transform_ego(ego_locs, locs, oris, bbox, typs, ego_ori, T=11):

    ego_loc = ego_locs[0]

    keys = sorted(list(locs.keys()))
    locs = np.array([locs[k] for k in keys]).reshape(-1,T,2)
    oris = np.array([oris[k] for k in keys]).reshape(-1,T)
    bbox = np.array([bbox[k] for k in keys]).reshape(-1,T,2)
    typs = np.array([typs[k] for k in keys]).reshape(-1,T)

    R = [[np.sin(ego_ori),np.cos(ego_ori)],[-np.cos(ego_ori),np.sin(ego_ori)]]
    
    locs = (locs-ego_loc) @ R
    ego_locs = (ego_locs-ego_loc) @ R
    oris = oris - ego_ori

    return ego_locs, locs, oris, bbox, typs

class FSDataset(BasicDataset):
    def __init__(self, config_path, is_train=True):
        super().__init__(config_path, is_train=is_train)
        self.max_other = 10
        self.is_train = is_train
        if(is_train):
            self.stage = "train"
        else:
            self.stage = "val"
        self.max_vehicle_radius = 100
        self.max_pedestrian_radius = 100
        self.num_plan = 20
        self.angles = []
        # print("hi")
        
    def __len__(self):
        if(self.is_train):
            return super().__len__()
        else:
            return super().__len__()

    def get_prev_bevs(self, lmdb_txn, index):
        context = self.num_prev_timesteps
        prev_bevs = []
        for t, i in enumerate(range(index-context,index)):
            if(i < 0):
                prev_bevs.append(np.zeros((4, 320, 320)))
                continue
            bev = self.__class__.load_bev(lmdb_txn, i, channels=[0,1,2,6])
            bev = (bev>0).astype(np.uint8).transpose(2,0,1)
            prev_bevs.append(bev)
        
        return prev_bevs
    
    def get_fs_segment(self, bevs, img, ego_locs, ego_oris, other_locs, idx=None, cmd=None):
        PIXELS_PER_METER = 4
        PIXELS_AHEAD_VEHICLE = 120
        img_h, img_w, _ = img.shape
        fs_mask = np.zeros((img_h, img_w)).astype(np.uint8)
        other_locs = other_locs[1:]
        other_locs_center = other_locs * PIXELS_PER_METER
        other_locs_center[:, 1] -= PIXELS_AHEAD_VEHICLE
        other_locs_center = 320/2 - other_locs_center
        
        x_original = np.linspace(0, 1, ego_locs.shape[0])
        x_new = np.linspace(0, 1, 50)

        ego_locs = np.column_stack([
            np.interp(x_new, x_original, ego_locs[:, i]) for i in range(ego_locs.shape[1])
        ])
        ego_oris = np.interp(x_new, x_original, ego_oris)
                
        x = (ego_locs[:, 0]).tolist()
        y = ego_locs[:, 1].tolist()

        # canvas = bevs[-1].transpose(1, 2, 0)
        # canvas = canvas[..., 1] + canvas[..., -1]
        # canvas = np.tile(canvas[..., None], (1, 1, 3))
        other_locs_raw = other_locs
        ego_locs_raw = ego_locs
        
        # print(other_locs)
        all_bboxes_in_bev = []
        for i, (px, py) in enumerate(zip(x, y)):
            px, py = int(320/2 - px * PIXELS_PER_METER), int(320/2 - py * PIXELS_PER_METER + PIXELS_AHEAD_VEHICLE)
            width = 6
            height = 8
            orientation = -ego_oris[i] + np.pi/2 # * 180 / np.pi
            # orientation = orientation * 0
            l, t, r, b = px - width//2, py - height//2, px + width//2, py + height//2
            # cv2.circle(canvas, (px, py), 2, (1, 0, 1), -1)
            # cv2.rectangle(canvas, (l, t), (r, b), (1, 0, 1), -1)
            R = np.array([[np.cos(orientation), -np.sin(orientation)],
                          [np.sin(orientation), np.cos(orientation)]])
            points = np.array([(-width//2, height//2), (width//2, height//2), (width//2, -height//2), (-width//2, -height//2)])
            rotated_points = (R @ points.T).T
            rotated_points[:, 1] *= -1
            rotated_points += (px, py)
            
            # for (locx, locy) in other_locs_center.astype(np.int32):
            #     cv2.circle(canvas, (locx, locy), radius=2, color=(1, 0, 1), thickness=-1)
            
            other_locs = other_locs_center - (px, py)
            other_locs[:, 1] *= -1
            other_locs = (R.T @ other_locs.T).T
            
            mask = (other_locs[:, 0] <= width//2) * (other_locs[:, 0] >= -width//2) * (other_locs[:, 1] <= height//2) * (other_locs[:, 1] >= -height//2)
            if(i == 0):
                sign = np.ones(len(other_locs))
                sign[other_locs_raw[:, 1] < 0] = -1
                distance = np.sqrt(np.sum(other_locs ** 2, -1)) * sign
                continue
                
            if(mask.sum()):
                break
            
            rotated_points = rotated_points.astype(np.int32)
            # cv2.drawContours(canvas, [rotated_points], 0, (1, 0, 1), -1)
            x_z = np.concatenate([320/2 - rotated_points[:, 0:1], 320/2 + PIXELS_AHEAD_VEHICLE - rotated_points[:, 1:2]], 1) / PIXELS_PER_METER
            all_bboxes_in_bev.append(x_z)
            # break
        
        valid = False
        if(len(all_bboxes_in_bev) == 0):
            distance = distance[distance>0]
            if(len(distance)):
                if(distance.min() < 30):
                    valid = True
        else:
            centers = np.array(all_bboxes_in_bev).mean(1)
            diff_vec = centers[-1] - centers[0]
            distance = distance[distance>0]
            if(np.linalg.norm(diff_vec) < 1e-3):
                if(len(distance)):
                    if(distance.min() < 30):
                        valid = True
            else:
                valid = True
            # valid = True

        angles = []
        try:
            centers = np.array(all_bboxes_in_bev).mean(1)
            diff_vec = centers[-1] - centers[0]
            angle = np.arctan2(diff_vec[1], diff_vec[0])
            self.angles.append(diff_vec)
        except:
            angle = np.array([0, 0])
            self.angles.append(angle)

        fov = 60
        rgb_w = 256
        rgb_h = 288
        focal = rgb_w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = rgb_w / 2.0
        K[1, 2] = rgb_h / 2.0

        # print(all_bboxes_in_bev)
        for box_in_bev in all_bboxes_in_bev:
            ego_locs = box_in_bev
            ego_locs_xyz = np.concatenate([np.zeros((ego_locs.shape[0], 1)) - 2.5, ego_locs], -1)
            ego_locs_xyz = np.concatenate([-ego_locs_xyz[:, 1:2], -ego_locs_xyz[:, 0:1], ego_locs_xyz[:, 2:3]-1.4], -1)
            
            ego_locs_xyz[:, :] = ego_locs_xyz[:, :] / ego_locs_xyz[:, -1:]
            proj_points = K @ ego_locs_xyz.T
            proj_points = (proj_points.T).astype(np.int32)
            for points in proj_points:
                x, y, _ = points
                # print(x, y)
                # cv2.circle(img, (x, y), radius=2, color=(255, 255, 255), thickness=1)
            
            if(np.all(proj_points[:, 1] > 0)):
                cv2.drawContours(fs_mask, [proj_points[:, :-1]], 0, (1, 1, 1), -1)
        
        # cv2.imwrite("base.png", canvas * 255)
        # if(ego_locs_raw[25, 0] > 2):
        # cv2.imwrite(f"/ssd_scratch/cvit/keshav/debug_segs/{idx}_{valid}_{cmd}.png", img)
        # exit(0)
        return fs_mask.astype(np.float32), valid, angle
    
    def filter_sem(self, sem, labels=[7]):
        h, w = sem.shape
        resem = np.zeros((h, w, len(labels))).astype(sem.dtype)
        for i, label in enumerate(labels):
            resem[sem==label, i] = 1

        return resem
    
    def load_contour(self, contour):
        contour = contour[0]
        mask = (contour[:, 0] == -2) & (contour[:, 1] == -2)
        contour[mask] = np.tile(np.array([0, 1])[None], (mask.sum(), 1)) # (bottom edge center)
        return contour[None]
    
    def __getitem__(self, idx):
        cache_path = f"/ssd_scratch/cvit/keshav/carla_fs_data_{self.stage}/{idx}.npz"
        if(os.path.exists(cache_path)):
            new_data = {}
            data = dict(np.load(cache_path))
            # print(data.keys())
            # exit(0)
            new_data['img'] = data['img'] * 255
            new_data['contour'] = self.load_contour(data['contour'])
            # vis_img = np.ascontiguousarray((data['img'] * 255).astype(np.uint8).transpose(1, 2, 0))
            # for point in new_data['contour'][0]:
            #     x, y = ((point+1)/2 * 256).astype(np.int32)  # Get the x, y coordinates
            #     cv2.circle(vis_img, (x, y), radius=1, color=(0, 255, 0), thickness=1)
            # new_data['vis'] = vis_img
            new_data['valid'] = data['valid']
            new_data['cmd'] = data['cmd']
            if(self.stage != "train"):
                new_data['mask'] = data['mask']
            # if(data['cmd'] == 5 and data['valid']):
            #     viz = new_data['img'].transpose(1, 2, 0)
            #     mask = data['mask']
            #     viz[mask == 1] = [0, 0, 255]
            #     cv2.imwrite(f"/ssd_scratch/cvit/keshav/lane_change/{idx}.png", viz)
            # cv2.imwrite(f"/ssd_scratch/cvit/keshav/baseline_vae/{self.stage}/{idx}.png", data['mask']*255)
            # if(new_data['cmd'] <= 2):
            #     new_data['valid'] = False
            # print(new_data['cmd'])
            # print(data['mask'].shape)
            # if(new_data['cmd'] == 3):
            # cv2.imwrite(f"img_{new_data['cmd']}.png", data['mask'][..., None] * new_data['img'].transpose(1, 2, 0))
            return new_data
        exit(0)
        # idx = 41
        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]

        # rgb1 = self.__class__.load_img(lmdb_txn, 'rgb_2', index)
        rgb2 = self.__class__.load_img(lmdb_txn, 'rgb_2', index)
        sem = self.__class__.load_img(lmdb_txn, 'sem_2', index)
        # sem = self.filter_sem(sem, labels=[4, 10, 6, 18]) # ped, cars, road_markings, traffic_light
        sem = self.filter_sem(sem, labels=list(range(20))) # ped, cars, road_markings, traffic_light
        # cv2.imwrite("mask.png", sem[..., 0]*255)
        # sem2 = self.__class__.load_img(lmdb_txn, 'sem_3', index
        
        # BEV images
        bev = self.__class__.load_bev(lmdb_txn, index, channels=[0,1,2,6])
        bev = (bev>0).astype(np.uint8).transpose(2,0,1)
        prev_bevs = self.get_prev_bevs(lmdb_txn, index)
        prev_bevs.append(bev)
        bevs = np.stack(prev_bevs)

        # rgb = np.concatenate([rgb1, rgb2], axis=1)
        # sem = np.concatenate([sem1, sem2], axis=1)

        # rgb = self.augmenter(images=rgb[...,::-1][None])[0]
        # sem = filter_sem(sem, self.seg_channels)

        # Vehicle locations/orientations
        ego_id, ego_locs, ego_oris, ego_bbox, msks, locs, oris, bbox, typs = self.__class__.filter(
            lmdb_txn, index,
            max_pedestrian_radius=self.max_pedestrian_radius,
            max_vehicle_radius=self.max_vehicle_radius,
            T=self.num_plan)
        
        # Normalize coordinates to ego frame

        ego_locs, locs, oris, bbox, typs = transform_ego(ego_locs, locs, oris, bbox, typs, ego_oris[0], self.num_plan+1)
        # print(ego_locs, ego_oris) 

        cmd = int(self.__class__.access('cmd', lmdb_txn, index, 1, dtype=np.uint8))
        nxp = self.__class__.access('nxp', lmdb_txn, index, 1).reshape(2)
        
        bbox_other = np.zeros((self.max_other, self.num_plan+1, 2))
        oris_other = np.zeros((self.max_other, self.num_plan+1))
        locs_other = np.zeros((self.max_other, self.num_plan+1, 2))
        mask_other = np.zeros(self.max_other)
        bbox_other[:bbox.shape[0]] = bbox[:self.max_other]
        locs_other[:locs.shape[0]] = locs[:self.max_other]
        oris_other[:oris.shape[0]] = oris[:self.max_other]
        mask_other[:len(bbox)] = 1

        ret = (bevs.astype(np.float32), ego_locs.astype(np.float32), oris, rgb2, locs_other.astype(np.float32), mask_other.astype(np.bool_))
        bevs = ret[0]
        img = ret[3]
        ego_locs = ret[1]
        ego_oris = ret[2][0]
        mask_other = ret[5]
        # print(mask_other)
        other_locs = ret[4][mask_other][:, 0, :] # (N, 2)
        
        mask, valid, angle = self.get_fs_segment(bevs, img, ego_locs, ego_oris, other_locs, idx=idx, cmd=cmd)
        mask = cv2.resize(mask, (256, 256))
        mask[mask > 0] = 1
        # mask[20:30, 20:30] = 1
        # sampled_coordinates = np.argwhere(mask == 1)
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # img = cv2.resize(img, (64, 64))
        # print(contours)
        img = cv2.resize(img, (256, 256))
        
        output_image = img/255 #np.tile(np.copy(mask)[..., None], (1, 1, 3))
        
        all_contours = []
        if(len(contours) == 0):
            contour = np.ones((self.num_contour_points, 2)) * -2
        else:
            max_index = np.array([len(c) for c in contours]).argmax()
            contour = contours[max_index]
            contour = self.interpolate_contour_points(contour[:, 0], self.num_contour_points)
            # for point in contour_interp:
            #     x, y = point.astype(np.int32)  # Get the x, y coordinates
            #     cv2.circle(output_image, (x, y), radius=1, color=(0, 1, 0), thickness=1)
            contour = contour / 256
            contour = contour * 2 - 1
        
        # cv2.imwrite(f"vis.png", output_image * 255)
        
        img = img.transpose(2, 0, 1).astype(np.float32)/255
        # sem = cv2.resize(sem, (64, 64)).astype(np.float32).transpose(2, 0, 1)
        data = {"img" : img, "mask": mask, "contour": contour[None].astype(np.float32)}
        # data['vis'] = output_image * 255
        data['valid'] = valid
        data['angle'] = angle
        data['ego_locs'] = ego_locs
        data['cmd'] = cmd
        data['idx'] = idx
        
        return data
    
    def interpolate_contour_points(self, contour, target_num_points):
        num_points = len(contour)
        # if num_points >= target_num_points:
        #     return contour
        indices = np.linspace(0, num_points - 1, target_num_points, dtype=np.float32)
        
        contour_interp = np.zeros((target_num_points, 2), dtype=np.float32)
        for i in range(2):
            contour_interp[:, i] = np.interp(indices, np.arange(num_points), contour[:, i])
        
        return contour_interp
    

if __name__ == '__main__':

    dataset = FSDataset('lav.yaml', is_train=True)
    dataloader = torch.utils.data.dataloader.DataLoader(dataset, batch_size=1, num_workers=16)
    from tqdm import tqdm
    import os
    output_folder = "/ssd_scratch/cvit/keshav/carla_fs_data_train/"
    os.makedirs(output_folder, exist_ok=True)

    # for i in tqdm(range(len(dataset))):
    count = {}
    for i, batch in tqdm(enumerate(dataloader)):
        # continue
        new = {}
        for k, v in batch.items():
            new[k] = v.numpy()[0]
        
        # cmd = new['cmd']
        # if(cmd in count):
        #     count[cmd] += 1
        # else:
        #     count[cmd] = 1
    
    # print(count)
    # for k, v in count.items():
    #     print(k, v, 1/v)
    # weights = []
    # for i, batch in tqdm(enumerate(dataloader)):
    #     new = {}
    #     for k, v in batch.items():
    #         new[k] = v.numpy()[0]
        
    #     cmd = new['cmd']
    #     weights.append(1/count[cmd])
    
    # np.savetxt("weights.txt", np.array(weights))
        np.savez(f"{output_folder}/{i}.npz", **new)
    
    # for i in tqdm.tqdm(range(0, 3000)):
    #     data = dataset[i]
    #     np_img = data['vis']
    #     valid = data['valid']
    #     # img = np.hstack([img, np.tile(mask[..., None], (1, 1, 3))])
    #     if(valid):
    #         cv2.imwrite(f"/ssd_scratch/cvit/keshav/img_{i}.png", np_img)
    #     # exit(0)