import logging
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.nn.functional as F
import wandb
from numpy import *
from tqdm import tqdm
# from trainers.trainer_utils import merge_two_branch_results_dist


class CLIP_Adapter_Trainer(object):
    def __init__(self, rank, config, model, logit_scale, image_proj, text_proj, optimizer,
                 scheduler, train_loader):
        self.rank = rank
        self.config = config
        self.model = model
        self.logit_scale = logit_scale
        self.image_proj = image_proj
        self.text_proj = text_proj

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.epoch = 0
        self.step = 0
        self.best_img_contras_acc = 0
        self.best_text_contras_acc = 0
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc = 0
        self.best_lvis_acc = 0
        self.config.ngpu = dist.get_world_size()

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        if self.config.training.use_text_proj:
            self.text_proj.load_state_dict(checkpoint['text_proj'])
        if self.config.training.use_image_proj:
            self.image_proj.load_state_dict(checkpoint['image_proj'])

        self.logit_scale.load_state_dict(checkpoint['logit_scale'])  # module.logit_scale = checkpoint['logit_scale']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.config.training.scheduler == "default":
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_img_contras_acc = checkpoint['best_img_contras_acc']
        self.best_text_contras_acc = checkpoint['best_text_contras_acc']
        self.best_modelnet40_overall_acc = checkpoint['best_modelnet40_overall_acc']
        self.best_modelnet40_class_acc = checkpoint['best_modelnet40_class_acc']
        self.best_lvis_acc = checkpoint['best_lvis_acc']

        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))
        logging.info("----Best img contras acc: {}".format(self.best_img_contras_acc))
        logging.info("----Best text contras acc: {}".format(self.best_text_contras_acc))
        logging.info("----Best modelnet40 overall acc: {}".format(self.best_modelnet40_overall_acc))
        logging.info("----Best modelnet40 class acc: {}".format(self.best_modelnet40_class_acc))
        logging.info("----Best lvis acc: {}".format(self.best_lvis_acc))

    def contras_loss(self, feat1, feat2, logit_scale=1, mask=None):
        if self.config.ngpu > 1:
            # i=5
            # if i<4:
            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)
            all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
            all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
            logits = logit_scale * all_feat1 @ all_feat2.T
            # print("logit", logits.shape, self.rank)
        else:
            logits = logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
        if mask is not None:
            logits = logits * mask
        labels = torch.arange(logits.shape[0]).to(self.config.device)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        # print(logits.argmax(dim=1))
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss, accuracy

    def compute_logit(self, feat1, feat2, logit_scale=1, mask=None):
        if self.config.ngpu > 1:
            # i=5
            # if i<4:
            feat1 = F.normalize(feat1, dim=1)
            feat2 = F.normalize(feat2, dim=1)
            all_feat1 = torch.cat(torch.distributed.nn.all_gather(feat1), dim=0)
            all_feat2 = torch.cat(torch.distributed.nn.all_gather(feat2), dim=0)
            logits = logit_scale * all_feat1 @ all_feat2.T
            # print("logit", logits.shape, self.rank)
        else:
            logits = logit_scale * F.normalize(feat1, dim=1) @ F.normalize(feat2, dim=1).T
        if mask is not None:
            logits = logits * mask
        labels = torch.arange(logits.shape[0]).to(self.config.device)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        # print(logits.argmax(dim=1))
        # loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return logits

    def train_one_epoch(self):
        self.model.eval()
        if self.config.training.use_text_proj:
            self.text_proj.train()
        if self.config.training.use_image_proj:
            self.image_proj.train()

        text_contras_acc_list = []
        if self.config.training.use_mask:
            k = self.config.dataset.negative_sample_num
            s = self.config.dataset.train_batch_size
            mask1 = np.eye(k * s).astype(np.bool)
            mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool)
            mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)
        img_text_pair_before = {}
        img_text_pair_after = {}
        for data in tqdm(self.train_loader):
            # print(data)
            self.step += 1
            self.optimizer.zero_grad()
            text_feat = torch.vstack(data['text_feat']).to(self.config.device)
            img_feat = torch.vstack(data['img_feat']).to(self.config.device)
            name = data["name"]
            group = data["group"]
            texts = data["texts"]
            idx = data['has_text_idx']
            image_idx = data["image_idx"]
            print(image_idx)
            logit_scale = self.logit_scale(None)
            logits = self.compute_logit(img_feat[idx], text_feat, logit_scale=logit_scale,
                                                          mask=None)
            labels = torch.arange(logits.shape[0]).to(self.config.device)
            # print("before", (logits.argmax(dim=1) == labels))
            logit_result = logits.argmax(dim=1)

            acc_before = logits.argmax(dim=1) == labels


            img_feat = self.model(img_feat)

            logits = self.compute_logit(img_feat[idx], text_feat, logit_scale=logit_scale,
                                        mask=None)
            labels = torch.arange(logits.shape[0]).to(self.config.device)
            # print("after", (logits.argmax(dim=1) == labels))
            acc_after = logits.argmax(dim=1) == labels


            for i in range(len(acc_after)):
                if acc_before[i] == False and acc_after[i] == True:
                    # print(group[i])
                    # print(image_idx[i])
                    group_id = group[i]
                    idx = image_idx[i]
                    object_name = name[i]

                    text = texts[logit_result[i]]
                    image_path = os.path.join(group_id, object_name, "colors_" + str(idx) + ".png")

                    img_text_pair_before[image_path] = text

                    img_text_pair_after[image_path] = texts[i]


            # print(img_feat[idx].shape, "image feat")
            # print(text_feat[idx].shape, "text feat")

            # logit_scale = self.logit_scale(None)
            #
            # if self.config.training.use_mask:
            #     img_text_sim = F.normalize(img_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T
            #     mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
            #     mask = torch.logical_or(mask, mask_other).detach()
            # else:
            #     mask = None
            #
            # if self.config.training.use_image_proj:
            #     img_feat = self.image_proj(img_feat)
            #
            # if self.config.training.use_text_proj:
            #     text_feat = self.text_proj(text_feat)
            #
            #
            # loss = 0
            #
            #
            # contras_loss, contras_acc = self.contras_loss(img_feat[idx], text_feat, logit_scale=logit_scale,
            #                                               mask=mask)
            #
            #
            # loss += contras_loss
            # text_contras_acc_list.append(contras_acc.item())

            # print("here, ", self.rank)
        #     loss.backward()
        #     self.optimizer.step()
        #     if self.config.training.scheduler == "cosine" or self.config.training.scheduler == "const":
        #         self.scheduler(self.step)
        #     else:
        #         self.scheduler.step()
        #
        #
        # if self.rank == 0:
        #     logging.info('Train: cotras_acc: {0}' \
        #                  .format(np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0))

    def save_model(self, name):
        torch.save({
            "state_dict": self.model.state_dict(),
            "logit_scale": self.logit_scale.state_dict(),  # module.logit_scale,
            "text_proj": self.text_proj.state_dict() if self.config.training.use_text_proj else None,
            "image_proj": self.image_proj.state_dict() if self.config.training.use_image_proj else None,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.config.training.scheduler == "default" else None,
            "epoch": self.epoch,
            "step": self.step,
        }, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))


    def train(self):
        for epoch in range(self.epoch, self.config.training.max_epoch):
            self.epoch = epoch
            if self.rank == 0:
                logging.info("Epoch: {}".format(self.epoch))
            self.train_one_epoch()

            if self.rank == 0:
                self.save_model('latest')
            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model('epoch_{}'.format(self.epoch))
