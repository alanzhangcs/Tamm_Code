import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset
import torch
from utils.logger import  *
from utils.data import random_rotate_z, normalize_pc, augment_pc
import MinkowskiEngine as ME

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc



def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetFewShot(Dataset):
    def __init__(self, config):
        self.root = config.DATA_PATH
        self.npoints = config.N_POINTS
        self.use_normals = config.USE_NORMALS
        self.num_category = config.NUM_CATEGORY
        self.process_data = True
        self.uniform = True
        self.use_color = True
        split = config.subset
        self.subset = config.subset

        self.way = config.way
        self.shot = config.shot
        self.fold = config.fold
        if self.way == -1 or self.shot == -1 or self.fold == -1:
            raise RuntimeError()

        self.pickle_path = os.path.join(self.root, f'{self.way}way_{self.shot}shot', f'{self.fold}.pkl')

        print('Load processed data from %s...' % self.pickle_path)

        with open(self.pickle_path, 'rb') as f:
            self.dataset = pickle.load(f)[self.subset]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        points, label, _ = self.dataset[index]
        points = points[:, :3]
        pt_idxs = np.arange(0, points.shape[0])  # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()

        # y up and normalize
        current_points[:, [1, 2]] = current_points[:, [2, 1]]
        current_points = normalize_pc(current_points)
        rgb = np.zeros_like(current_points)
        if self.use_color:
            features = np.concatenate([current_points, rgb], axis=1)
        else:
            features = current_points


        # current_points = torch.from_numpy(current_points).float()
        return {
            "xyz": torch.from_numpy(current_points).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),

            "category": label
        }

def minkowski_modelnet40_collate_fn(list_data):
    return {
        "xyz": ME.utils.batched_coordinates([data["xyz"] for data in list_data], dtype=torch.float32),
        "features": torch.cat([data["features"] for data in list_data], dim=0),
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "category": torch.tensor([data["category"] for data in list_data], dtype=torch.int32),
    }



from torch.utils.data import Dataset, DataLoader
def make_modelNetFewShot(config):
    dataset = ModelNetFewShot(config)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = DataLoader(
        dataset, \
        num_workers=config.num_workers, \
        collate_fn=minkowski_modelnet40_collate_fn, \
        batch_size=config.batch_size, \
        pin_memory=True, \
        shuffle=False,
        sampler=sampler
    )
    return data_loader


