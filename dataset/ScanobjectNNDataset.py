import numpy as np
import os, sys, h5py
from torch.utils.data import Dataset
import torch
from utils.data import random_rotate_z, normalize_pc, augment_pc
import MinkowskiEngine as ME



class ScanObjectNN(Dataset):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.subset = config.subset
        self.root = config.DATA_PATH
        self.use_color = True

        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])  # 2048
        # if self.subset == 'train':
        #     np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()



        # current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]
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

    def __len__(self):
        return self.points.shape[0]

class ScanObjectNN_hardest(Dataset):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.subset = config.subset
        self.root = config.DATA_PATH
        self.use_color = True

        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])  # 2048
        # if self.subset == 'train':
        #     np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()



        # current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]
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


    def __len__(self):
        return self.points.shape[0]


def minkowski_modelnet40_collate_fn(list_data):
    return {
        "xyz": ME.utils.batched_coordinates([data["xyz"] for data in list_data], dtype=torch.float32),
        "features": torch.cat([data["features"] for data in list_data], dim=0),
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "category": torch.tensor([data["category"] for data in list_data], dtype=torch.int32),
    }



from torch.utils.data import Dataset, DataLoader
def make_ScanObjectNN(config):
    dataset = ScanObjectNN(config)
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


def make_ScanObjectNN_hardest(config):
    dataset = ScanObjectNN_hardest(config)
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