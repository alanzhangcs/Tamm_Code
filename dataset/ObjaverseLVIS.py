import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from utils.logger import  *
from utils.data import random_rotate_z, normalize_pc, augment_pc
import MinkowskiEngine as ME
import json
import logging
class ObjaverseLVIS(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.split, "r"))
        self.y_up = config.y_up
        self.num_points = config.num_points
        self.use_color = config.use_color
        self.normalize = config.normalize
        self.categories = sorted(np.unique([data['category'] for data in self.split]))
        self.category2idx = {self.categories[i]: i for i in range(len(self.categories))}
        # self.clip_cat_feat = np.load(config.clip_feat_path, allow_pickle=True)

        logging.info("ObjaverseLVIS: %d samples" % (len(self.split)))
        # logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        data_path = self.split[index]['data_path']
        data_path = data_path.replace("/mnt/data", 'OpenShape_data')
        data = np.load(data_path, allow_pickle=True).item()
        n = data['xyz'].shape[0]
        # if n != self.num_points:
        # idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][: self.num_points]
        rgb = data['rgb'][: self.num_points]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        assert not np.isnan(xyz).any()

        # idx = np.random.randint(data['image_feat'].shape[0])
        # img_feat = data["image_feat"][idx]

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "category": self.category2idx[self.split[index]["category"]],

        }

    def __len__(self):
        return len(self.split)


def minkowski_objaverse_lvis_collate_fn(list_data):
    return {
        "xyz": ME.utils.batched_coordinates([data["xyz"] for data in list_data], dtype=torch.float32),
        "features": torch.cat([data["features"] for data in list_data], dim=0),
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "category": torch.tensor([data["category"] for data in list_data], dtype=torch.int32),
    }


def make_objaverse_lvis(config):
    dataset = ObjaverseLVIS(config)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    return DataLoader(
        ObjaverseLVIS(config), \
        num_workers=config.num_workers, \
        collate_fn=minkowski_objaverse_lvis_collate_fn, \
        batch_size=config.batch_size, \
        pin_memory=True, \
        shuffle=False, sampler=sampler
    )