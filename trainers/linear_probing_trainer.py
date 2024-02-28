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
from trainers.trainer_utils import merge_results_dist
import torch.nn as nn


class Linear_Probing_Trainer(object):
    def __init__(self, rank, config, model, linear_layer, optimizer,
                 scheduler, train_loader, test_loader, image_branch=None, text_branch=None):
        self.rank = rank
        self.config = config
        self.model = model
        self.linear_layer = linear_layer

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epoch = 0
        self.step = 0

        self.config.ngpu = dist.get_world_size()
        self.image_branch = image_branch
        self.text_branch = text_branch
        self.criterion = nn.CrossEntropyLoss()
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc = 0

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.config.training.scheduler == "default":
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']

        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))

    def train_one_epoch(self):
        self.model.eval()
        self.linear_layer.train()
        if self.image_branch is not None and self.text_branch is not None:
            self.image_branch.eval()
            self.text_branch.eval()

        for data in tqdm(self.train_loader):
            # print(data)
            self.step += 1
            self.optimizer.zero_grad()
            loss = 0
            with torch.no_grad():
                # print(data["xyz_dense"].shape)
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                if self.image_branch is not None and self.text_branch is not None:
                    pc_image_feat = self.image_branch(pred_feat)
                    pc_text_feat = self.text_branch(pred_feat)
                    if self.config.linear_layer.feature_type == "all":
                        pred_feat = torch.cat((pc_text_feat, pc_image_feat), dim=-1)
                    elif self.config.linear_layer.feature_type == "image_branch":
                        pred_feat = pc_image_feat
                    elif self.config.linear_layer.feature_type == "text_branch":
                        pred_feat = pc_text_feat
                    else:
                        pred_feat = pred_feat
            output = self.linear_layer(pred_feat)
            label = data["category"].to(self.config.device)
            loss = self.criterion(output, label.long())
            loss.backward()

            self.optimizer.step()
            if self.config.training.scheduler == "cosine" or self.config.training.scheduler == "const":
                self.scheduler(self.step)
            else:
                self.scheduler.step()

        # test
        logits_all = []
        labels_all = []
        num_cates = self.config.dataset.NUM_CATEGORY
        per_cat_correct = torch.zeros(num_cates).cuda()
        per_cat_count = torch.zeros(num_cates).cuda()
        self.linear_layer.eval()
        with torch.no_grad():
            for data in tqdm(self.test_loader):
                with torch.no_grad():
                    if not self.config.model.get("use_dense", False):
                        pred_feat = self.model(data['xyz'], data['features'], \
                                               device=self.config.device, \
                                               quantization_size=self.config.model.voxel_size)
                    else:
                        pred_feat = self.model(data['xyz_dense'], data['features_dense'])
                    if self.image_branch is not None and self.text_branch is not None:
                        pc_image_feat = self.image_branch(pred_feat)
                        pc_text_feat = self.text_branch(pred_feat)
                        if self.config.linear_layer.feature_type == "all":
                            pred_feat = torch.cat((pc_text_feat, pc_image_feat), dim=-1)
                        elif self.config.linear_layer.feature_type == "image_branch":
                            pred_feat = pc_image_feat
                        elif self.config.linear_layer.feature_type == "text_branch":
                            pred_feat = pc_text_feat
                        else:
                            pred_feat = pred_feat

                    labels = data['category'].to(self.config.device)
                    labels_all.append(labels)

                    logits = self.linear_layer(pred_feat)

                    logits_all.append(logits)

        logits_all, labels_all = merge_results_dist(os.path.join(self.config.ckpt_dir, self.config.dataset.NAME + "_dir"),
                                                    logits_all, labels_all)
        if self.rank == 0:
            topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

            for i in range(num_cates):
                idx = (labels_all == i)
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i] = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            per_cat_acc = per_cat_correct / per_cat_count
            # for i in range(40):
            #    print(idx2category[i], per_cat_acc[i])

            if overall_acc > self.best_modelnet40_overall_acc:
                self.best_modelnet40_overall_acc = overall_acc
                self.save_model('best_modelnet40_overall')
            if per_cat_acc.mean() > self.best_modelnet40_class_acc:
                self.best_modelnet40_class_acc = per_cat_acc.mean()
                self.save_model('best_modelnet40_class')

            logging.info('Test ModelNet40: overall acc: {0}({1}) class_acc: {2}({3})'.format(overall_acc,
                                                                                             self.best_modelnet40_overall_acc,
                                                                                             per_cat_acc.mean(),
                                                                                             self.best_modelnet40_class_acc))
            logging.info(
                'Test ModelNet40: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                    topk_acc[1].item(),
                                                                                    topk_acc[2].item()))

    def save_model(self, name):
        torch.save({
            "state_dict": self.linear_layer.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.config.training.scheduler == "default" else None,
            "epoch": self.epoch,
            "step": self.step,
        }, os.path.join(self.config.ckpt_dir, '{}.pt'.format(name)))

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res, correct

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

