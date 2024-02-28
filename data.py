import copy
import json
import logging
import os
import random

import MinkowskiEngine as ME
import numpy as np
import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset, DataLoader

# import open3d
from utils.data import random_rotate_z, normalize_pc, augment_pc

ImageFile.LOAD_TRUNCATED_IMAGES = True


class LLMDataset(Dataset):
    def __init__(self, config, phase):
        self.phase = phase
        if phase == "train":
            self.split = json.load(open(config.dataset.train_split, "r"))
            if config.dataset.train_partial > 0:
                self.split = self.split[:config.dataset.train_partial]
        else:
            raise NotImplementedError("Phase %s not supported." % phase)
        self.phase = phase
        self.y_up = config.dataset.y_up
        self.random_z_rotate = config.dataset.random_z_rotate
        self.num_points = config.dataset.num_points
        self.use_color = config.dataset.use_color
        self.normalize = config.dataset.normalize
        self.rgb_random_drop_prob = config.dataset.rgb_random_drop_prob
        self.text_source = config.dataset.text_source
        self.augment = config.dataset.augment
        self.use_knn_negative_sample = config.dataset.use_knn_negative_sample
        self.use_text_filtering = config.dataset.use_text_filtering
        self.num_imgs = config.dataset.num_imgs
        self.num_texts = config.dataset.num_texts
        self.use_prompt_engineering = config.dataset.use_prompt_engineering
        self.feat_path = config.dataset.feat_path
        self.text_embed_version = "prompt_avg" if self.use_prompt_engineering else "original"
        if self.use_knn_negative_sample:
            self.negative_sample_num = config.dataset.negative_sample_num
            self.knn = np.load(config.dataset.knn_path, allow_pickle=True).item()
            self.uid_to_index = {}
            for i, item in enumerate(self.split):
                self.uid_to_index[item['id']] = i
        if self.use_text_filtering:
            self.gpt4_filtering = json.load(open(config.dataset.gpt4_filtering_path, "r"))
        logging.info("Phase %s: %d samples" % (phase, len(self.split)))

    def get_data(self, meta):
        data_path = meta['data_path'].replace("/mnt/data/objaverse-processed/merged_for_training_final", self.feat_path)
        dataset = meta['dataset']
        uid = meta["id"]
        data = np.load(data_path, allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.phase == "train" and self.augment:
            xyz = augment_pc(xyz)
        if self.phase == "train" and self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        if self.use_color:
            if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        text_feat = []
        # print(data)
        if np.random.rand() < 0.5:
            if 'categories' in self.text_source:
                idx = np.random.randint(len(data["categories_feat"]))
                text_feat.append(data["categories_feat"][idx][self.text_embed_version])

            if 'descriptions' in self.text_source:
                idx = np.random.randint(len(data["descriptions_feat"]))
                text_feat.append(data["descriptions_feat"][idx]["original"])
            if 'llm_cate_texts' in self.text_source:
                idx = np.random.randint(len(data["llm_cate_texts_feat"]))
                text_feat.append(data["llm_cate_texts_feat"][idx]["original"])
                # print(data["llm_cate_texts_feat"][idx]["original"].shape)
        else:
            if 'text' in self.text_source:
                if dataset == "Objaverse":
                    if not (self.use_text_filtering and self.gpt4_filtering[uid]["flag"] == "N"):
                        text_feat.append(data["text_feat"][0][self.text_embed_version])
                else:
                    idx = np.random.randint(len(data["text_feat"]))
                    text_feat.append(data["text_feat"][idx][self.text_embed_version])

            if 'openshape_caption' in self.text_source:
                if np.random.rand() < 0.5:
                    if len(data["blip_caption_feat"]) > 0:
                        text_feat.append(data["blip_caption_feat"][0][self.text_embed_version])
                else:
                    if len(data["msft_caption_feat"]) > 0:
                        text_feat.append(data["msft_caption_feat"][0][self.text_embed_version])

            if 'openshape_retrieval' in self.text_source:
                if len(data["retrieval_text_feat"]) > 0:
                    idx = np.random.randint(len(data["retrieval_text_feat"]))
                    text_feat.append(
                        data["retrieval_text_feat"][idx]["original"])

        text_feat = random.sample(text_feat, self.num_texts)
        if self.num_texts > 1:
            text_feat = np.concatenate(text_feat)
        else:
            text_feat = np.array(text_feat)

        if dataset == "Objaverse":
            if self.num_imgs == 1:
                if np.random.rand() < 0.5:
                    image_feat = data['thumbnail_feat']
                else:
                    idx = np.random.randint(data['image_feat'].shape[0])
                    image_feat = data["image_feat"][idx]
            else:
                image_feat = []
                image_feat.append(data["thumbnail_feat"])
                image_feat += random.sample(list(data["image_feat"]), self.num_imgs - 1)
                image_feat = np.concatenate(image_feat)
        else:
            image_feat = random.sample(list(data["image_feat"]), self.num_imgs)
            image_feat = np.concatenate(image_feat)

        assert not np.isnan(xyz).any()

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "img_feat": torch.from_numpy(image_feat).type(torch.float32).reshape(-1),
            "text_feat": torch.from_numpy(text_feat).type(torch.float32).reshape(-1),
            "dataset": meta["dataset"],
            "group": meta["group"],
            "name": meta["id"],
            "has_text": text_feat is not None,
        }

    def __getitem__(self, index: int):
        if self.use_knn_negative_sample == False:
            return self.get_data(self.split[index])
        else:
            data_list = []
            # random select a seed shape from split
            index = random.randint(0, len(self.split) - 1)
            uid = self.split[index]['id']
            # randomly pick (negative_sample_num - 1) neighbors from 31 nearest neighbors
            knn_idx = [0] + (np.random.choice(31, self.negative_sample_num - 1, replace=False) + 1).tolist()
            for i in knn_idx:
                idx = self.uid_to_index[self.knn['name'][self.knn['index'][uid][i]]]
                data_list.append(self.get_data(self.split[idx]))
            return data_list

    def __len__(self):
        if self.use_knn_negative_sample == False:
            return len(self.split)
        else:
            return len(self.split) // self.negative_sample_num


class LLMDataset1(Dataset):
    def __init__(self, config, phase):
        self.phase = phase
        if phase == "train":
            self.split = json.load(open(config.dataset.train_split, "r"))
            if config.dataset.train_partial > 0:
                self.split = self.split[:config.dataset.train_partial]
        else:
            raise NotImplementedError("Phase %s not supported." % phase)
        self.phase = phase
        self.y_up = config.dataset.y_up
        self.random_z_rotate = config.dataset.random_z_rotate
        self.num_points = config.dataset.num_points
        self.use_color = config.dataset.use_color
        self.normalize = config.dataset.normalize
        self.rgb_random_drop_prob = config.dataset.rgb_random_drop_prob
        self.text_source = config.dataset.text_source
        self.augment = config.dataset.augment
        self.use_knn_negative_sample = config.dataset.use_knn_negative_sample
        self.use_text_filtering = config.dataset.use_text_filtering
        self.num_imgs = config.dataset.num_imgs
        self.num_texts = config.dataset.num_texts
        self.use_prompt_engineering = config.dataset.use_prompt_engineering
        self.feat_path = config.dataset.feat_path
        self.text_embed_version = "prompt_avg" if self.use_prompt_engineering else "original"
        if self.use_knn_negative_sample:
            self.negative_sample_num = config.dataset.negative_sample_num
            self.knn = np.load(config.dataset.knn_path, allow_pickle=True).item()
            self.uid_to_index = {}
            for i, item in enumerate(self.split):
                self.uid_to_index[item['id']] = i
        if self.use_text_filtering:
            self.gpt4_filtering = json.load(open(config.dataset.gpt4_filtering_path, "r"))
        logging.info("Phase %s: %d samples" % (phase, len(self.split)))

    def get_data(self, meta):
        data_path = meta['data_path'].replace("/mnt/data/objaverse-processed/merged_for_training_final", self.feat_path)
        dataset = meta['dataset']
        uid = meta["id"]
        data = np.load(data_path, allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.phase == "train" and self.augment:
            xyz = augment_pc(xyz)
        if self.phase == "train" and self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        if self.use_color:
            if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        text_feat = []
        # print(data)

        if np.random.rand() < 0.5:
            if 'categories' in self.text_source:
                idx = np.random.randint(len(data["categories_feat"]))
                text_feat.append(data["categories_feat"][idx][self.text_embed_version])
        else:
            if 'text' in self.text_source:
                if dataset == "Objaverse":
                    if not (self.use_text_filtering and self.gpt4_filtering[uid]["flag"] == "N"):
                        text_feat.append(data["text_feat"][0][self.text_embed_version])
                    else:
                        if 'categories' in self.text_source:
                            idx = np.random.randint(len(data["categories_feat"]))
                            text_feat.append(data["categories_feat"][idx][self.text_embed_version])
                else:
                    idx = np.random.randint(len(data["text_feat"]))
                    text_feat.append(data["text_feat"][idx][self.text_embed_version])

        if 'openshape_caption' in self.text_source:
            if np.random.rand() < 0.5:
                if len(data["blip_caption_feat"]) > 0:
                    text_feat.append(data["blip_caption_feat"][0][self.text_embed_version])
            else:
                if len(data["msft_caption_feat"]) > 0:
                    text_feat.append(data["msft_caption_feat"][0][self.text_embed_version])

        if np.random.rand() < 0.5:
            if np.random.rand() < 0.5:
                if 'descriptions' in self.text_source:
                    idx = np.random.randint(len(data["descriptions_feat"]))
                    text_feat.append(data["descriptions_feat"][idx]["original"])
            else:
                if 'llm_cate_texts' in self.text_source:
                    idx = np.random.randint(len(data["llm_cate_texts_feat"]))
                    text_feat.append(data["llm_cate_texts_feat"][idx]["original"])
        else:
            if 'openshape_retrieval' in self.text_source:
                if len(data["retrieval_text_feat"]) > 0:
                    idx = np.random.randint(len(data["retrieval_text_feat"]))
                    text_feat.append(
                        data["retrieval_text_feat"][idx]["original"])

        text_feat = random.sample(text_feat, self.num_texts)
        if self.num_texts > 1:
            text_feat = np.concatenate(text_feat)
        else:
            text_feat = np.array(text_feat)

        if dataset == "Objaverse":
            if self.num_imgs == 1:
                if np.random.rand() < 0.5:
                    image_feat = data['thumbnail_feat']
                else:
                    idx = np.random.randint(data['image_feat'].shape[0])
                    image_feat = data["image_feat"][idx]
            else:
                image_feat = []
                image_feat.append(data["thumbnail_feat"])
                image_feat += random.sample(list(data["image_feat"]), self.num_imgs - 1)
                image_feat = np.concatenate(image_feat)
        else:
            image_feat = random.sample(list(data["image_feat"]), self.num_imgs)
            image_feat = np.concatenate(image_feat)

        assert not np.isnan(xyz).any()

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "img_feat": torch.from_numpy(image_feat).type(torch.float32).reshape(-1),
            "text_feat": torch.from_numpy(text_feat).type(torch.float32).reshape(-1),
            "dataset": meta["dataset"],
            "group": meta["group"],
            "name": meta["id"],
            "has_text": text_feat is not None,
        }

    def __getitem__(self, index: int):
        if self.use_knn_negative_sample == False:
            return self.get_data(self.split[index])
        else:
            data_list = []
            # random select a seed shape from split
            index = random.randint(0, len(self.split) - 1)
            uid = self.split[index]['id']
            # randomly pick (negative_sample_num - 1) neighbors from 31 nearest neighbors
            knn_idx = [0] + (np.random.choice(31, self.negative_sample_num - 1, replace=False) + 1).tolist()
            for i in knn_idx:
                idx = self.uid_to_index[self.knn['name'][self.knn['index'][uid][i]]]
                data_list.append(self.get_data(self.split[idx]))
            return data_list

    def __len__(self):
        if self.use_knn_negative_sample == False:
            return len(self.split)
        else:
            return len(self.split) // self.negative_sample_num


class Four(Dataset):
    def __init__(self, config, phase):
        self.phase = phase
        if phase == "train":
            self.split = json.load(open(config.dataset.train_split, "r"))
            if config.dataset.train_partial > 0:
                self.split = self.split[:config.dataset.train_partial]
        else:
            raise NotImplementedError("Phase %s not supported." % phase)
        self.phase = phase
        self.y_up = config.dataset.y_up
        self.random_z_rotate = config.dataset.random_z_rotate
        self.num_points = config.dataset.num_points
        self.use_color = config.dataset.use_color
        self.normalize = config.dataset.normalize
        self.rgb_random_drop_prob = config.dataset.rgb_random_drop_prob
        self.text_source = config.dataset.text_source
        self.augment = config.dataset.augment
        self.use_knn_negative_sample = config.dataset.use_knn_negative_sample
        self.use_text_filtering = config.dataset.use_text_filtering
        self.use_prompt_engineering = config.dataset.use_prompt_engineering
        self.num_imgs = config.dataset.num_imgs
        self.num_texts = config.dataset.num_texts
        self.clip_embed_version = config.clip_embed_version
        self.text_embed_version = "prompt_avg" if self.use_prompt_engineering else "original"
        if self.use_knn_negative_sample:
            self.negative_sample_num = config.dataset.negative_sample_num
            self.knn = np.load(config.dataset.knn_path, allow_pickle=True).item()
            self.uid_to_index = {}
            for i, item in enumerate(self.split):
                self.uid_to_index[item['id']] = i
        if self.use_text_filtering:
            self.gpt4_filtering = json.load(open(config.dataset.gpt4_filtering_path, "r"))
        self.use_openshape_feature = config.dataset.use_openshape_feature
        logging.info("Phase %s: %d samples" % (phase, len(self.split)))

    def get_data(self, meta):
        if self.clip_embed_version == "EVACLIP":
            data_path = meta['data_path'].replace("/mnt/data/objaverse-processed/merged_for_training_final",
                                                  'data/tamm-processed/llm-feat/eva-clip')
        else:
            data_path = meta['data_path'].replace("/mnt/data", 'data')
        dataset = meta['dataset']
        uid = meta["id"]
        data = np.load(data_path, allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.phase == "train" and self.augment:
            xyz = augment_pc(xyz)
        if self.phase == "train" and self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        if self.use_color:
            if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        text_feat = []
        texts = []
        if 'text' in self.text_source:
            if dataset == "Objaverse":
                if not (self.use_text_filtering and self.gpt4_filtering[uid]["flag"] == "N"):
                    texts.append(data["text"][0])
                    text_feat.append(data["text_feat"][0][self.text_embed_version])
            else:
                idx = np.random.randint(len(data["text"]))
                texts.append(data["text"][idx])
                text_feat.append(data["text_feat"][idx][self.text_embed_version])

        if 'caption' in self.text_source:
            if np.random.rand() < 0.5:
                if len(data["blip_caption"]) > 0:
                    texts.append(data["blip_caption"])
                    if self.use_openshape_feature:
                        text_feat.append(data["blip_caption_feat"][self.text_embed_version])
                    else:
                        # print(data["blip_caption_feat"].shape)
                        text_feat.append(data["blip_caption_feat"
                                         ])
            else:
                if len(data["msft_caption"]) > 0:
                    texts.append(data["msft_caption"])
                    if self.use_openshape_feature:
                        text_feat.append(data["msft_caption_feat"][self.text_embed_version])
                    else:
                        # print(data["msft_caption_feat"].shape)
                        text_feat.append(data["msft_caption_feat"])

        if 'retrieval_text' in self.text_source:
            if len(data["retrieval_text"]) > 0:
                idx = np.random.randint(len(data["retrieval_text"]))
                texts.append(data["retrieval_text"][idx])
                text_feat.append(
                    data["retrieval_text_feat"][idx]["original"])  # no prompt engineering for retrieval text

        text_feat = random.sample(text_feat, self.num_texts)
        if self.num_texts > 1:
            text_feat = np.concatenate(text_feat)
        else:
            text_feat = np.array(text_feat)

        if dataset == "Objaverse":
            if self.num_imgs == 1:
                if np.random.rand() < 0.5:
                    image_feat = data['thumbnail_feat']
                else:
                    idx = np.random.randint(data['image_feat'].shape[0])
                    image_feat = data["image_feat"][idx]
            else:
                image_feat = []
                image_feat.append(data["thumbnail_feat"])
                image_feat += random.sample(list(data["image_feat"]), self.num_imgs - 1)
                image_feat = np.concatenate(image_feat)
        else:
            image_feat = random.sample(list(data["image_feat"]), self.num_imgs)
            image_feat = np.concatenate(image_feat)

        assert not np.isnan(xyz).any()

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "img_feat": torch.from_numpy(image_feat).type(torch.float32).reshape(-1),
            "text_feat": torch.from_numpy(text_feat).type(torch.float32).reshape(-1),
            "dataset": meta["dataset"],
            "group": meta["group"],
            "name": meta["id"],
            "texts": texts,
            "image_idx": idx,
            "has_text": text_feat is not None,
        }

    def __getitem__(self, index: int):
        if self.use_knn_negative_sample == False:
            return self.get_data(self.split[index])
        else:
            data_list = []
            # random select a seed shape from split
            index = random.randint(0, len(self.split) - 1)
            uid = self.split[index]['id']
            # randomly pick (negative_sample_num - 1) neighbors from 31 nearest neighbors
            knn_idx = [0] + (np.random.choice(31, self.negative_sample_num - 1, replace=False) + 1).tolist()
            for i in knn_idx:
                # print(self.knn['index'][uid][i], 'name')
                idx = self.uid_to_index[self.knn['name'][self.knn['index'][uid][i]]]
                data_list.append(self.get_data(self.split[idx]))
            return data_list

    def __len__(self):
        if self.use_knn_negative_sample == False:
            return len(self.split)
        else:
            return len(self.split) // self.negative_sample_num


class Four1(Dataset):
    def __init__(self, config, phase):
        self.phase = phase
        if phase == "train":
            self.split = json.load(open(config.dataset.train_split, "r"))
            if config.dataset.train_partial > 0:
                self.split = self.split[:config.dataset.train_partial]
        else:
            raise NotImplementedError("Phase %s not supported." % phase)
        self.phase = phase
        self.y_up = config.dataset.y_up
        self.random_z_rotate = config.dataset.random_z_rotate
        self.num_points = config.dataset.num_points
        self.use_color = config.dataset.use_color
        self.normalize = config.dataset.normalize
        self.rgb_random_drop_prob = config.dataset.rgb_random_drop_prob
        self.text_source = config.dataset.text_source
        self.augment = config.dataset.augment
        self.use_knn_negative_sample = config.dataset.use_knn_negative_sample
        self.use_text_filtering = config.dataset.use_text_filtering
        self.use_prompt_engineering = config.dataset.use_prompt_engineering
        self.num_imgs = config.dataset.num_imgs
        self.num_texts = config.dataset.num_texts
        self.clip_embed_version = config.clip_embed_version
        self.text_embed_version = "prompt_avg" if self.use_prompt_engineering else "original"
        if self.use_knn_negative_sample:
            self.negative_sample_num = config.dataset.negative_sample_num
            self.knn = np.load(config.dataset.knn_path, allow_pickle=True).item()
            self.uid_to_index = {}
            for i, item in enumerate(self.split):
                self.uid_to_index[item['id']] = i
        if self.use_text_filtering:
            self.gpt4_filtering = json.load(open(config.dataset.gpt4_filtering_path, "r"))
        self.use_openshape_feature = config.dataset.use_openshape_feature
        logging.info("Phase %s: %d samples" % (phase, len(self.split)))

    def get_data(self, meta):
        if self.clip_embed_version == "EVACLIP":
            data_path = meta['data_path'].replace("/mnt/data/objaverse-processed/merged_for_training_final",
                                                  'data/tamm-processed/llm-feat/eva-clip')
        else:
            data_path = meta['data_path'].replace("/mnt/data", 'data')
        dataset = meta['dataset']

        data = np.load(data_path, allow_pickle=True).item()
        n = data['xyz'].shape[0]
        idx = random.sample(range(n), self.num_points)
        xyz = data['xyz'][idx]
        rgb = data['rgb'][idx]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]
        if self.normalize:
            xyz = normalize_pc(xyz)
        if self.phase == "train" and self.augment:
            xyz = augment_pc(xyz)
        if self.phase == "train" and self.random_z_rotate:
            xyz = random_rotate_z(xyz)
        if self.use_color:
            if self.phase == "train" and np.random.rand() < self.rgb_random_drop_prob:
                rgb = np.ones_like(rgb) * 0.4
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        text_feat = []
        texts = []

        if 'text' in self.text_source:
            idx = np.random.randint(len(data["text_feat"]))
            # texts.append(data["text"][idx])
            if self.use_openshape_feature:
                text_feat.append(data["text_feat"][idx][self.text_embed_version])
            else:
                # print(data["text_feat"].shape)
                text_feat.append(data["text_feat"][idx])

        if 'caption' in self.text_source:
            if np.random.rand() < 0.5:
                if len(data["blip_caption_feat"]) > 0:
                    # texts.append(data["blip_caption"])
                    idx = np.random.randint(len(data["blip_caption_feat"]))
                    if self.use_openshape_feature:
                        text_feat.append(data["blip_caption_feat"][idx][self.text_embed_version])
                    else:
                        # print(data["blip_caption_feat"].shape)
                        text_feat.append(data["blip_caption_feat"])
            else:
                if len(data["msft_caption_feat"]) > 0:
                    # texts.append(data["msft_caption"])
                    if self.use_openshape_feature:
                        idx = np.random.randint(len(data["msft_caption_feat"]))
                        text_feat.append(data["msft_caption_feat"][idx][self.text_embed_version])
                    else:
                        # print(data["msft_caption_feat"].shape)
                        text_feat.append(data["msft_caption_feat"])

        if 'retrieval_text' in self.text_source:
            if len(data["retrieval_text_feat"]) > 0:
                idx = np.random.randint(len(data["retrieval_text_feat"]))
                # texts.append(data["retrieval_text"][idx])
                text_feat.append(
                    data["retrieval_text_feat"][idx]["original"])  # no prompt engineering for retrieval text

        text_feat = random.sample(text_feat, self.num_texts)
        if self.num_texts > 1:
            text_feat = np.concatenate(text_feat)
        else:
            text_feat = np.array(text_feat)

        if dataset == "Objaverse":
            if self.num_imgs == 1:
                if np.random.rand() < 0.5:
                    image_feat = data['thumbnail_feat']
                else:
                    idx = np.random.randint(data['image_feat'].shape[0])
                    image_feat = data["image_feat"][idx]
            else:
                image_feat = []
                image_feat.append(data["thumbnail_feat"])
                image_feat += random.sample(list(data["image_feat"]), self.num_imgs - 1)
                image_feat = np.concatenate(image_feat)
        else:
            image_feat = random.sample(list(data["image_feat"]), self.num_imgs)
            image_feat = np.concatenate(image_feat)

        assert not np.isnan(xyz).any()

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "img_feat": torch.from_numpy(image_feat).type(torch.float32).reshape(-1),
            "text_feat": torch.from_numpy(text_feat).type(torch.float32).reshape(-1),
            "dataset": meta["dataset"],
            "group": meta["group"],
            "name": meta["id"],
            "texts": texts,
            "image_idx": idx,
            "has_text": text_feat is not None,
        }

    def __getitem__(self, index: int):
        if self.use_knn_negative_sample == False:
            return self.get_data(self.split[index])
        else:
            data_list = []
            # random select a seed shape from split
            index = random.randint(0, len(self.split) - 1)
            uid = self.split[index]['id']
            # randomly pick (negative_sample_num - 1) neighbors from 31 nearest neighbors
            knn_idx = [0] + (np.random.choice(31, self.negative_sample_num - 1, replace=False) + 1).tolist()
            for i in knn_idx:
                # print(self.knn['index'][uid][i], 'name')
                idx = self.uid_to_index[self.knn['name'][self.knn['index'][uid][i]]]
                data_list.append(self.get_data(self.split[idx]))
            return data_list

    def __len__(self):
        if self.use_knn_negative_sample == False:
            return len(self.split)
        else:
            return len(self.split) // self.negative_sample_num


def minkowski_collate_fn(list_data):
    if isinstance(list_data[0], list):
        merged_list = []
        for data in list_data:
            merged_list += data
        list_data = merged_list
    return {
        "xyz": ME.utils.batched_coordinates([data["xyz"] for data in list_data], dtype=torch.float32),
        "features": torch.cat([data["features"] for data in list_data], dim=0),
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "img_feat": [data["img_feat"] for data in list_data],
        "text_feat": [data["text_feat"] for data in list_data if data["text_feat"] is not None],
        "dataset": [data["dataset"] for data in list_data],
        "group": [data["group"] for data in list_data],
        "name": [data["name"] for data in list_data],
        "image_idx": [0 for data in list_data],
        "has_text_idx": [i for i, data in enumerate(list_data) if data["text_feat"] is not None],
    }


def make(config, phase, rank, world_size):
    if config.dataset.name == "Four":
        dataset = Four(config, phase, )
        if phase == "train":
            batch_size = config.dataset.train_batch_size
            # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=True)
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            raise NotImplementedError("Phase %s not supported." % phase)
        data_loader = DataLoader(
            dataset,
            num_workers=config.dataset.num_workers,
            collate_fn=minkowski_collate_fn,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=True,
            sampler=sampler
        )

    elif config.dataset.name == "Four1":
        dataset = Four1(config, phase, )
        if phase == "train":
            batch_size = config.dataset.train_batch_size
            # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=True)
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            raise NotImplementedError("Phase %s not supported." % phase)
        data_loader = DataLoader(
            dataset,
            num_workers=config.dataset.num_workers,
            collate_fn=minkowski_collate_fn,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=True,
            sampler=sampler
        )

    elif config.dataset.name == "LLM":
        dataset = LLMDataset(config, phase, )
        if phase == "train":
            batch_size = config.dataset.train_batch_size
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            raise NotImplementedError("Phase %s not supported." % phase)
        data_loader = DataLoader(
            dataset,
            num_workers=config.dataset.num_workers,
            collate_fn=minkowski_collate_fn,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=True,
            sampler=sampler
        )
    elif config.dataset.name == "LLM1":
        dataset = LLMDataset1(config, phase, )
        if phase == "train":
            batch_size = config.dataset.train_batch_size
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            raise NotImplementedError("Phase %s not supported." % phase)
        data_loader = DataLoader(
            dataset,
            num_workers=config.dataset.num_workers,
            collate_fn=minkowski_collate_fn,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=True,
            sampler=sampler
        )

    else:
        raise NotImplementedError("Dataset %s not supported." % config.dataset.name)
    return data_loader


class ModelNet40Test(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.modelnet40.test_split, "r"))
        self.pcs = np.load(config.modelnet40.test_pc, allow_pickle=True)
        self.num_points = config.modelnet40.num_points
        self.use_color = config.dataset.use_color
        self.y_up = config.modelnet40.y_up
        clip_feat = np.load(config.modelnet40.clip_feat_path, allow_pickle=True).item()
        self.categories = list(clip_feat.keys())
        self.clip_cat_feat = []
        self.category2idx = {}
        if config.clip_embed_version == "OpenCLIP":
            for i, category in enumerate(self.categories):
                self.category2idx[category] = i
                self.clip_cat_feat.append(clip_feat[category]["open_clip_text_feat"])
        else:
            for i, category in enumerate(self.categories):
                self.category2idx[category] = i
                self.clip_cat_feat.append(clip_feat[category]["prompt_avg"])
        self.clip_cat_feat = np.concatenate(self.clip_cat_feat, axis=0)

        logging.info("ModelNet40Test: %d samples" % len(self.split))
        logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        pc = copy.deepcopy(self.pcs[index])
        n = pc['xyz'].shape[0]
        # if n != self.num_points:
        # idx = random.sample(range(n), self.num_points)
        xyz = pc['xyz'][: self.num_points]
        rgb = pc['rgb'][: self.num_points]
        rgb = rgb / 255.0  # 100, scale to 0.4 to make it consistent with the training data
        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]

        xyz = normalize_pc(xyz)

        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        assert not np.isnan(xyz).any()

        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "name": self.split[index]["name"],
            "category": self.category2idx[self.split[index]["category"]],
        }

    def __len__(self):
        return len(self.split)


def minkowski_modelnet40_collate_fn(list_data):
    return {
        "xyz": ME.utils.batched_coordinates([data["xyz"] for data in list_data], dtype=torch.float32),
        "features": torch.cat([data["features"] for data in list_data], dim=0),
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "name": [data["name"] for data in list_data],
        "category": torch.tensor([data["category"] for data in list_data], dtype=torch.int32),
    }


def make_modelnet40test(config):
    dataset = ModelNet40Test(config)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = DataLoader(
        dataset, \
        num_workers=config.modelnet40.num_workers, \
        collate_fn=minkowski_modelnet40_collate_fn, \
        batch_size=config.modelnet40.test_batch_size, \
        pin_memory=True, \
        shuffle=False,
        sampler=sampler
    )
    return data_loader


class ObjaverseLVIS(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.objaverse_lvis.split, "r"))
        self.y_up = config.objaverse_lvis.y_up
        self.num_points = config.objaverse_lvis.num_points
        self.use_color = config.objaverse_lvis.use_color
        self.normalize = config.objaverse_lvis.normalize
        self.categories = sorted(np.unique([data['category'] for data in self.split]))
        if config.clip_embed_version == "OpenCLIP":
            self.clip_cat_feat = np.load(config.objaverse_lvis.clip_feat_path, allow_pickle=True)
            # self.clip_cat_feat = np.concatenate(self.clip_cat_feat, axis=0)
            self.category2idx = {self.categories[i]: i for i in range(len(self.categories))}
        else:
            clip_feat = np.load(config.objaverse_lvis.clip_feat_path, allow_pickle=True).item()
            self.category2idx = {}
            self.clip_cat_feat = []
            for i, category in enumerate(self.categories):
                self.category2idx[category] = i
                self.clip_cat_feat.append(clip_feat[category]["prompt_avg"])
            self.clip_cat_feat = np.concatenate(self.clip_cat_feat, axis=0)

        logging.info("ObjaverseLVIS: %d samples" % (len(self.split)))
        logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        data_path = self.split[index]['data_path']
        data_path = data_path.replace("/mnt/data", 'data')
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
            "group": self.split[index]['group'],
            "name": self.split[index]['uid'],
            "category": self.category2idx[self.split[index]["category"]],
            # "image_feat": torch.from_numpy(img_feat).type(torch.float32)
        }

    def __len__(self):
        return len(self.split)


def minkowski_objaverse_lvis_collate_fn(list_data):
    return {
        "xyz": ME.utils.batched_coordinates([data["xyz"] for data in list_data], dtype=torch.float32),
        "features": torch.cat([data["features"] for data in list_data], dim=0),
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        "group": [data["group"] for data in list_data],
        "name": [data["name"] for data in list_data],
        "category": torch.tensor([data["category"] for data in list_data], dtype=torch.int32),
        # "image_feat": torch.stack([data['image_feat'] for data in list_data])
    }


def make_objaverse_lvis(config):
    dataset = ObjaverseLVIS(config)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    return DataLoader(
        ObjaverseLVIS(config), \
        num_workers=config.objaverse_lvis.num_workers, \
        collate_fn=minkowski_objaverse_lvis_collate_fn, \
        batch_size=config.objaverse_lvis.batch_size, \
        pin_memory=True, \
        shuffle=False, sampler=sampler
    )


class ScanObjectNNTest(Dataset):
    def __init__(self, config):
        self.data = np.load(config.scanobjectnn.data_path, allow_pickle=True).item()
        self.num_points = config.scanobjectnn.num_points
        self.use_color = config.dataset.use_color
        self.y_up = config.scanobjectnn.y_up
        clip_feat = np.load(config.scanobjectnn.clip_feat_path, allow_pickle=True).item()
        self.categories = ["bag", "bin", "box", "cabinet", "chair", "desk", "display", "door", "shelf", "table", "bed",
                           "pillow", "sink", "sofa", "toilet"]
        self.clip_cat_feat = []
        self.category2idx = {}
        if config.clip_embed_version == "OpenCLIP":
            for i, category in enumerate(self.categories):
                self.category2idx[category] = i
                self.clip_cat_feat.append(clip_feat[category]["open_clip_text_feat"])
        else:
            for i, category in enumerate(self.categories):
                self.category2idx[category] = i
                self.clip_cat_feat.append(clip_feat[category]["prompt_avg"])
        self.clip_cat_feat = np.concatenate(self.clip_cat_feat, axis=0)
        logging.info("ScanObjectNNTest: %d samples" % self.__len__())
        logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        xyz = copy.deepcopy(self.data['xyz'][index])
        if 'rgb' not in self.data:
            rgb = np.ones_like(xyz) * 0.4
        else:
            rgb = self.data['rgb'][index]
        label = self.data['label'][index]
        n = xyz.shape[0]
        # if n != self.num_points:
        # idx = np.random.choice(n, self.num_points)  # random.sample(range(n), self.num_points)
        xyz = xyz[: self.num_points]
        rgb = rgb[: self.num_points]

        if self.y_up:
            # swap y and z axis
            xyz[:, [1, 2]] = xyz[:, [2, 1]]

        xyz = normalize_pc(xyz)
        if self.use_color:
            features = np.concatenate([xyz, rgb], axis=1)
        else:
            features = xyz

        assert not np.isnan(xyz).any()
        return {
            "xyz": torch.from_numpy(xyz).type(torch.float32),
            "features": torch.from_numpy(features).type(torch.float32),
            "name": str(index),
            "category": label,
        }

    def __len__(self):
        return len(self.data['xyz'])


def make_scanobjectnntest(config):
    dataset = ScanObjectNNTest(config)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = DataLoader(
        dataset, \
        num_workers=config.scanobjectnn.num_workers, \
        collate_fn=minkowski_modelnet40_collate_fn, \
        batch_size=config.scanobjectnn.test_batch_size, \
        pin_memory=True, \
        shuffle=False,
        sampler=sampler
    )
    return data_loader


class OpenShape_ImageTexts(Dataset):
    def __init__(self, config, phase):
        self.phase = phase
        if phase == "train":
            self.split = json.load(open(config.dataset.train_split, "r"))
            if config.dataset.train_partial > 0:
                self.split = self.split[:config.dataset.train_partial]
        else:
            raise NotImplementedError("Phase %s not supported." % phase)
        self.phase = phase
        self.text_source = config.dataset.text_source
        # self.augment = config.dataset.augment
        self.use_text_filtering = config.dataset.use_text_filtering
        self.use_prompt_engineering = config.dataset.use_prompt_engineering
        self.use_knn_negative_sample = config.dataset.use_knn_negative_sample
        if self.use_text_filtering:
            self.gpt4_filtering = json.load(open(config.dataset.gpt4_filtering_path, "r"))
        logging.info("Phase %s: %d samples" % (phase, len(self.split)))

    def get_objaverse(self, meta):
        uid = meta["id"]
        data_path = meta['data_path'].replace("/mnt/data", 'OpenShape_data')
        data = np.load(data_path, allow_pickle=True).item()
        texts = []
        if 'text' in self.text_source:
            idx = np.random.randint(len(data["text"]))
            texts.append(data["text"][idx])

        if 'caption' in self.text_source:
            if np.random.rand() < 0.5:
                if len(data["blip_caption"]) > 0:
                    texts.append(data["blip_caption"])
            else:
                if len(data["msft_caption"]) > 0:
                    texts.append(data["msft_caption"])

        if 'retrieval_text' in self.text_source:
            if len(data["retrieval_text"]) > 0:
                idx = np.random.randint(len(data["retrieval_text"]))
                texts.append(data["retrieval_text"][idx])

        if len(texts) > 0:
            text_idx = np.random.randint(len(texts))
            texts = texts[text_idx]
        else:
            texts = None

        image_path = meta['data_path'].replace("/mnt/data/objaverse-processed/merged_for_training_final", 'Open_image')
        image_path = image_path[:-4]
        rendered_images = os.listdir(image_path)
        rendered_images_num = len(rendered_images)

        idx = np.random.randint(rendered_images_num)
        image = os.path.join(image_path, rendered_images[idx])
        images = Image.open(image).convert("RGB")

        return {
            "dataset": "Objaverse",
            "group": meta["group"],
            "name": uid,
            "texts": texts,
            "images": images
        }

    def get_others(self, meta):
        data_path = meta['data_path'].replace("/mnt/data", 'OpenShape_data')
        data = np.load(data_path, allow_pickle=True).item()

        texts = []
        if 'text' in self.text_source:
            idx = np.random.randint(len(data["text"]))
            texts.append(data["text"][idx])

        if 'caption' in self.text_source:
            if np.random.rand() < 0.5:
                if len(data["blip_caption"]) > 0:
                    texts.append(data["blip_caption"])
            else:
                if len(data["msft_caption"]) > 0:
                    texts.append(data["msft_caption"])

        if 'retrieval_text' in self.text_source:
            if len(data["retrieval_text"]) > 0:
                idx = np.random.randint(len(data["retrieval_text"]))
                texts.append(data["retrieval_text"][idx])

        if len(texts) > 0:
            text_idx = np.random.randint(len(texts))
            texts = texts[text_idx]
        else:
            texts = None

        image_path = meta['data_path'].replace("/mnt/data/objaverse-processed/merged_for_training_final", 'Open_image')
        image_path = image_path[:-4]
        rendered_images = os.listdir(image_path)
        rendered_images_num = len(rendered_images)

        idx = np.random.randint(rendered_images_num)
        image = os.path.join(image_path, rendered_images[idx])
        images = Image.open(image).convert("RGB")

        return {
            "dataset": meta["dataset"],
            "group": meta["group"],
            "name": meta["id"],
            "texts": texts,
            "images": images
        }

    def __getitem__(self, index: int):
        if self.use_knn_negative_sample == False:
            if self.split[index]['dataset'] == "Objaverse":
                return self.get_objaverse(self.split[index])
            else:
                return self.get_others(self.split[index])
        else:
            data_list = []
            # random select a seed shape from split
            index = random.randint(0, len(self.split) - 1)
            uid = self.split[index]['id']
            # randomly pick (negative_sample_num - 1) neighbors from 31 nearest neighbors
            knn_idx = [0] + (np.random.choice(31, self.negative_sample_num - 1, replace=False) + 1).tolist()
            for i in knn_idx:
                idx = self.uid_to_index[self.knn['name'][self.knn['index'][uid][i]]]
                if self.split[idx]['dataset'] == "Objaverse":
                    data_list.append(self.get_objaverse(self.split[idx]))
                else:
                    data_list.append(self.get_others(self.split[idx]))
            return data_list

    def __len__(self):
        if self.use_knn_negative_sample == False:
            return len(self.split)
        else:
            return len(self.split) // self.negative_sample_num


def open_text_images_collate_fn(list_data):
    if isinstance(list_data[0], list):
        merged_list = []
        for data in list_data:
            merged_list += data
        list_data = merged_list
    return {
        "dataset": [data["dataset"] for data in list_data],
        "group": [data["group"] for data in list_data],
        "name": [data["name"] for data in list_data],
        "texts": [data["texts"] for data in list_data],
        "images": [data["images"] for data in list_data],
    }


def make_open_text_images(config, phase):
    if config.dataset.name == "Four":
        dataset = OpenShape_ImageTexts(config, phase, )
        if phase == "train":
            batch_size = config.dataset.train_batch_size
            # sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, drop_last=True)
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            raise NotImplementedError("Phase %s not supported." % phase)
        data_loader = DataLoader(
            dataset,
            num_workers=config.dataset.num_workers,
            collate_fn=open_text_images_collate_fn,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=True,
            sampler=sampler,
        )
    else:
        raise NotImplementedError("Dataset %s not supported." % config.dataset.name)
    return data_loader


class ScannetTest(Dataset):
    def __init__(self, config):
        self.split = json.load(open(config.Scannet.split, "r"))
        self.num_points = config.Scannet.num_points
        self.use_color = config.dataset.use_color
        self.y_up = config.Scannet.y_up
        clip_feat = np.load(config.Scannet.clip_feat_path)
        self.categories = ['sink', 'chair', 'toilet', 'door', 'wall', 'desk', 'shower curtain', 'sofa', 'window',
                           'table', 'curtain', 'picture', 'cabinet', 'floor', 'refrigerator', 'bookshelf', 'bed',
                           'bathtub', 'counter']
        self.clip_cat_feat = []
        self.category2idx = {}
        for i, category in enumerate(self.categories):
            self.category2idx[category] = i
        self.clip_cat_feat = clip_feat
        logging.info("scannetTest: %d samples" % self.__len__())
        logging.info("----clip feature shape: %s" % str(self.clip_cat_feat.shape))

    def __getitem__(self, index: int):
        data_path = self.split[index]['data_path']

        # data = open3d.io.read_point_cloud(data_path)
        # xyz = np.asarray(data.points)
        #
        #
        # label = self.split[index]['category']
        #
        # n = xyz.shape[0]
        # if n < self.num_points:
        #     ratio = self.num_points / n
        #     ratio = int(ratio) + 1
        #
        #     xyz = np.repeat(xyz, ratio, axis=0)
        #     # print(xyz.shape)
        #
        # # if n != self.num_points:
        # n = xyz.shape[0]
        # rgb = np.zeros_like(xyz)
        # idx = random.sample(range(n), self.num_points)
        # xyz = xyz[idx]
        # rgb = rgb[idx]
        #
        #
        #
        #
        # if self.y_up:
        #     # swap y and z axis
        #     xyz[:, [1, 2]] = xyz[:, [2, 1]]
        #
        # xyz = normalize_pc(xyz)
        # if self.use_color:
        #     features = np.concatenate([xyz, rgb], axis=1)
        # else:
        #     features = xyz
        #
        # assert not np.isnan(xyz).any()
        # return {
        #     "xyz": torch.from_numpy(xyz).type(torch.float32),
        #     "features": torch.from_numpy(features).type(torch.float32),
        #     "category": self.category2idx[self.split[index]["category"]],
        # }

    def __len__(self):
        return len(self.split)


def minkowski_scannet_collate_fn(list_data):
    return {
        "xyz": ME.utils.batched_coordinates([data["xyz"] for data in list_data], dtype=torch.float32),
        "features": torch.cat([data["features"] for data in list_data], dim=0),
        "xyz_dense": torch.stack([data["xyz"] for data in list_data]).float(),
        "category": torch.tensor([data["category"] for data in list_data], dtype=torch.int32),
        "features_dense": torch.stack([data["features"] for data in list_data]),
        # "image_feat": torch.stack([data['image_feat'] for data in list_data])
    }


def make_scannet(config):
    dataset = ScannetTest(config)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = DataLoader(
        dataset, \
        num_workers=config.scanobjectnn.num_workers, \
        collate_fn=minkowski_scannet_collate_fn, \
        batch_size=config.scanobjectnn.test_batch_size, \
        pin_memory=True, \
        shuffle=False, sampler=sampler
    )
    return data_loader
