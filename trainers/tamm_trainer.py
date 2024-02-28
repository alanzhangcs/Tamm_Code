import logging
import os

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn
import torch.nn.functional as F
from numpy import *
from tqdm import tqdm

from trainers.trainer_utils import merge_two_branch_results_dist


class TAMM_Trainer(object):
    def __init__(self, rank, config, model, logit_scale, image_proj, text_proj, text_alignment_adapter, image_alignment_adapter, clip_adapter,
                 optimizer,
                 scheduler, train_loader, \
                 modelnet40_loader, objaverse_lvis_loader=None, scanobjectnn_loader=None):
        self.rank = rank
        self.config = config
        self.model = model
        self.logit_scale = logit_scale
        self.image_proj = image_proj
        self.text_proj = text_proj
        self.image_alignment_adapter = image_alignment_adapter
        self.text_alignment_adapter = text_alignment_adapter
        self.clip_adapter = clip_adapter
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.modelnet40_loader = modelnet40_loader
        self.objaverse_lvis_loader = objaverse_lvis_loader
        self.scanobjectnn_loader = scanobjectnn_loader
        self.epoch = 0
        self.step = 0
        self.alpha = 0.5
        self.best_img_contras_acc = 0
        self.best_text_contras_acc = 0
        self.best_modelnet40_overall_acc = 0
        self.best_modelnet40_class_acc = 0
        self.best_lvis_acc = 0
        self.config.ngpu = dist.get_world_size()

    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])

        self.image_alignment_adapter.load_state_dict(checkpoint['image_alignment_adapter'])
        self.text_alignment_adapter.load_state_dict(checkpoint['text_alignment_adapter'])

        self.logit_scale.load_state_dict(checkpoint['logit_scale'])  # module.logit_scale = checkpoint['logit_scale']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.config.training.scheduler == "default":
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch'] + 1
        self.step = checkpoint['step']

        logging.info("Loaded checkpoint from {}".format(path))
        logging.info("----Epoch: {0} Step: {1}".format(self.epoch, self.step))

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
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss, accuracy

    def train_one_epoch(self):
        self.model.train()
        if self.config.training.use_text_proj:
            self.text_proj.train()
        if self.config.training.use_image_proj:
            self.image_proj.train()
        self.image_alignment_adapter.train()
        self.text_alignment_adapter.train()

        text_contras_acc_list = []
        img_contras_acc_list = []
        intra_contras_acc_list = []
        if self.config.training.use_mask:
            k = self.config.dataset.negative_sample_num
            s = self.config.dataset.train_batch_size
            mask1 = np.eye(k * s).astype(np.bool)
            mask2 = np.kron(np.eye(s), np.ones((k, k))).astype(np.bool)
            mask_other = torch.from_numpy(np.logical_or(mask1, 1 - mask2)).bool().to(self.config.device)

        for data in tqdm(self.train_loader):
            # print(data)
            self.step += 1
            self.optimizer.zero_grad()
            loss = 0
            if not self.config.model.get("use_dense", False):
                pred_feat = self.model(data['xyz'], data['features'], \
                                       device=self.config.device, \
                                       quantization_size=self.config.model.voxel_size)
            else:
                pred_feat = self.model(data['xyz_dense'], data['features_dense'])
            logit_scale = self.logit_scale(None)

            text_feat = torch.vstack(data['text_feat'])
            img_feat = torch.vstack(data['img_feat'])

            if self.config.training.use_mask:
                img_text_sim = F.normalize(img_feat, dim=-1) @ F.normalize(text_feat, dim=-1).T
                mask = torch.diagonal(img_text_sim).reshape(-1, 1) - img_text_sim > self.config.training.mask_threshold
                mask = torch.logical_or(mask, mask_other).detach()
            else:
                mask = None

            pc_image_feat = self.image_alignment_adapter(pred_feat)
            pc_text_feat = self.text_alignment_adapter(pred_feat)

            idx = data['has_text_idx']

            if self.config.training.loss_type == "two_branch":
                for i in range(self.config.dataset.num_imgs):
                    single_img_feat = img_feat[:, i * self.config.clip_embed_dim: (i + 1) * self.config.clip_embed_dim]
                    single_img_feat = single_img_feat.to(self.config.device)

                    # use clip adapter to get new image feature
                    with torch.no_grad():
                        if self.clip_adapter is not None:
                            single_img_feat = self.clip_adapter(single_img_feat)

                    img_contras_loss, img_contras_acc = self.contras_loss(pc_image_feat, single_img_feat,
                                                                          logit_scale=logit_scale,
                                                                          mask=mask)
                    loss += img_contras_loss * self.config.training.lambda_img_contras
                    img_contras_acc_list.append(img_contras_acc.item())
                for i in range(self.config.dataset.num_texts):
                    single_text_feat = text_feat[:, i * self.config.clip_embed_dim: (i + 1) * self.config.clip_embed_dim]
                    single_text_feat = single_text_feat.to(self.config.device)
                    text_contras_loss, text_contras_acc = self.contras_loss(pc_text_feat[idx], single_text_feat,
                                                                            logit_scale=logit_scale, mask=mask)
                    loss += text_contras_loss * self.config.training.lambda_text_contras

                    text_contras_acc_list.append(text_contras_acc.item())

            loss.backward()
            self.optimizer.step()
            if self.config.training.scheduler == "cosine" or self.config.training.scheduler == "const":
                self.scheduler(self.step)
            else:
                self.scheduler.step()

        if self.rank == 0:
            logging.info('Train: text_cotras_acc: {0} image_contras_acc: {1}' \
                         .format(np.mean(text_contras_acc_list) if len(text_contras_acc_list) > 0 else 0,
                                 np.mean(img_contras_acc_list)))

    def save_model(self, name):
        torch.save({
            "state_dict": self.model.state_dict(),
            "logit_scale": self.logit_scale.state_dict(),  # module.logit_scale,
            "optimizer": self.optimizer.state_dict(),
            "image_alignment_adapter": self.image_alignment_adapter.state_dict(),
            "text_alignment_adapter": self.text_alignment_adapter.state_dict(),
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

            if epoch > self.config.training.test_epoch:
                self.test_modelnet40()
                self.test_objaverse_lvis()
                self.test_scanobjectnn()
            # if self.rank == 0:
            # self.save_model('latest')
            if self.rank == 0 and self.epoch % self.config.training.save_freq == 0:
                self.save_model('epoch_{}'.format(self.epoch))

    def test_modelnet40(self):
        self.model.eval()
        self.text_alignment_adapter.eval()
        self.image_alignment_adapter.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.modelnet40_loader.dataset.clip_cat_feat).cuda()
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(40).cuda()
        per_cat_count = torch.zeros(40).cuda()
        category2idx = self.modelnet40_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_image_all = []
        logits_text_all = []
        labels_all = []
        with torch.no_grad():
            for data in tqdm(self.modelnet40_loader):
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])

                pred_feat_text = F.normalize(self.text_alignment_adapter(pred_feat), dim=1)
                pred_feat_image = F.normalize(self.image_alignment_adapter(pred_feat), dim=1)

                logits_text = pred_feat_text @ F.normalize(clip_text_feat, dim=1).T
                logits_image = pred_feat_image @ F.normalize(clip_text_feat, dim=1).T

                labels = data['category'].to(self.config.device)
                logits_image_all.append(logits_image.detach())
                logits_text_all.append(logits_text.detach())
                labels_all.append(labels)

        logits_image_all, logits_text_all, labels_all = merge_two_branch_results_dist(
            os.path.join(self.config.ckpt_dir, "modelnet40_dir"),
            logits_image_all, logits_text_all, labels_all)

        if self.rank == 0:
            logits_all = logits_text_all + self.alpha * logits_image_all
            topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

            for i in range(40):
                idx = (labels_all == i)
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i] = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            per_cat_acc = per_cat_correct / per_cat_count

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

    def test_objaverse_lvis(self):
        self.model.eval()
        self.text_alignment_adapter.eval()
        self.image_alignment_adapter.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.objaverse_lvis_loader.dataset.clip_cat_feat).cuda()
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(1156).cuda()
        per_cat_count = torch.zeros(1156).cuda()
        category2idx = self.objaverse_lvis_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_image_all = []
        logits_text_all = []
        labels_all = []
        with torch.no_grad():
            for data in tqdm(self.objaverse_lvis_loader):
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])

                pred_feat_text = F.normalize(self.text_alignment_adapter(pred_feat), dim=1)
                pred_feat_image = F.normalize(self.image_alignment_adapter(pred_feat), dim=1)
                # print("pred_feat_text", pred_feat_text.shape)
                # print("clip_text_feat", clip_text_feat.shape)
                logits_text = pred_feat_text @ F.normalize(clip_text_feat, dim=1).T
                logits_image = pred_feat_image @ F.normalize(clip_text_feat, dim=1).T
                labels = data['category'].to(self.config.device)
                logits_image_all.append(logits_image.detach())
                logits_text_all.append(logits_text.detach())
                labels_all.append(labels)

        logits_image_all, logits_text_all, labels_all = merge_two_branch_results_dist(
            os.path.join(self.config.ckpt_dir, "objaverse_dir"),
            logits_image_all, logits_text_all, labels_all)

        if self.rank == 0:

            logits_all = 1 * logits_text_all + self.alpha * logits_image_all
            topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

            # calculate per class accuracy
            for i in torch.unique(labels_all):
                idx = (labels_all == i)
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i] = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            per_cat_acc = per_cat_correct / per_cat_count

            if overall_acc > self.best_lvis_acc:
                self.best_lvis_acc = overall_acc
                self.save_model('best_lvis')

            logging.info(
                'Test ObjaverseLVIS: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
            logging.info('Test ObjaverseLVIS: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                                topk_acc[1].item(),
                                                                                                topk_acc[2].item()))

    def test_scanobjectnn(self):
        self.model.eval()
        self.text_alignment_adapter.eval()
        self.image_alignment_adapter.eval()
        if self.config.training.use_text_proj:
            self.text_proj.eval()
        clip_text_feat = torch.from_numpy(self.scanobjectnn_loader.dataset.clip_cat_feat).to(self.config.device)
        if self.config.training.use_text_proj:
            clip_text_feat = self.text_proj(clip_text_feat)
        per_cat_correct = torch.zeros(15).to(self.config.device)
        per_cat_count = torch.zeros(15).to(self.config.device)
        category2idx = self.scanobjectnn_loader.dataset.category2idx
        idx2category = {v: k for k, v in category2idx.items()}

        logits_image_all = []
        logits_text_all = []
        labels_all = []
        with torch.no_grad():
            for data in self.scanobjectnn_loader:
                if not self.config.model.get("use_dense", False):
                    pred_feat = self.model(data['xyz'], data['features'], \
                                           device=self.config.device, \
                                           quantization_size=self.config.model.voxel_size)
                else:
                    pred_feat = self.model(data['xyz_dense'], data['features_dense'])

                pred_feat_text = F.normalize(self.text_alignment_adapter(pred_feat), dim=1)
                pred_feat_image = F.normalize(self.image_alignment_adapter(pred_feat), dim=1)

                logits_text = pred_feat_text @ F.normalize(clip_text_feat, dim=1).T
                logits_image = pred_feat_image @ F.normalize(clip_text_feat, dim=1).T

                labels = data['category'].to(self.config.device)
                logits_image_all.append(logits_image.detach())
                logits_text_all.append(logits_text.detach())
                labels_all.append(labels)

        logits_image_all, logits_text_all, labels_all = merge_two_branch_results_dist(
            os.path.join(self.config.ckpt_dir, "scanobjectnn_dir"),
            logits_image_all, logits_text_all, labels_all)

        if self.rank == 0:

            logits_all = 1 * logits_text_all + self.alpha * logits_image_all

            topk_acc, correct = self.accuracy(logits_all, labels_all, topk=(1, 3, 5,))

            # calculate per class accuracy
            for i in range(15):
                idx = (labels_all == i)
                if idx.sum() > 0:
                    per_cat_correct[i] = (logits_all[idx].argmax(dim=1) == labels_all[idx]).float().sum()
                    per_cat_count[i] = idx.sum()

            overall_acc = per_cat_correct.sum() / per_cat_count.sum()
            per_cat_acc = per_cat_correct / per_cat_count

            logging.info(
                'Test ScanObjectNN: overall acc: {0} class_acc: {1}'.format(overall_acc, per_cat_acc.mean()))
            logging.info('Test ScanObjectNN: top1_acc: {0} top3_acc: {1} top5_acc: {2}'.format(topk_acc[0].item(),
                                                                                               topk_acc[1].item(),
                                                                                               topk_acc[2].item()))

