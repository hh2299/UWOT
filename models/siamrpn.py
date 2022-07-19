from __future__ import absolute_import, division

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from collections import namedtuple
from got10k.trackers import Tracker
import os
from .ops import crop_and_resize, init_weights, read_image, show_image

from torch.utils.data import DataLoader

from data import Pair
from models.transform import SiamFCTransforms
from torch.nn import SmoothL1Loss
from  loss import SoftMaxLoss


class SiamRPN(nn.Module):

    def __init__(self, anchor_num=5):
        super(SiamRPN, self).__init__()
        self.anchor_num = anchor_num
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 192, 11, 2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(192, 512, 5, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(512, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(768, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(768, 512, 3, 1),
            nn.BatchNorm2d(512))

        self.conv_reg_z = nn.Conv2d(512, 512 * 4 * anchor_num, 3, 1)
        self.conv_reg_x = nn.Conv2d(512, 512, 3)
        self.conv_cls_z = nn.Conv2d(512, 512 * 2 * anchor_num, 3, 1)
        self.conv_cls_x = nn.Conv2d(512, 512, 3)
        self.adjust_reg = nn.Conv2d(4 * anchor_num, 4 * anchor_num, 1)

    def forward(self, z, x):
        return self.inference(x, **self.learn(z))

    def learn(self, z):
        z = self.feature(z)
        kernel_reg = self.conv_reg_z(z)
        kernel_cls = self.conv_cls_z(z)

        k = kernel_reg.size()[-1]
        kernel_reg = kernel_reg.view(4 * self.anchor_num, 512, k, k)
        kernel_cls = kernel_cls.view(2 * self.anchor_num, 512, k, k)

        return kernel_reg, kernel_cls

    def inference(self, x, kernel_reg, kernel_cls):
        x = self.feature(x)
        x_reg = self.conv_reg_x(x)
        x_cls = self.conv_cls_x(x)

        out_reg = self.adjust_reg(F.conv2d(x_reg, kernel_reg))
        out_cls = F.conv2d(x_cls, kernel_cls)

        return out_reg, out_cls


class TrackerSiamRPN(Tracker):

    def __init__(self, net_path=None, **kargs):
        super(TrackerSiamRPN, self).__init__(
            name='SiamRPN', is_deterministic=True)
        self.parse_args(**kargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = SiamRPN()
        init_weights(self.net)

        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        # define loss
        self.loss_reg = SmoothL1Loss()
        self.loss_cls = SoftMaxLoss()

    def parse_args(self, **kargs):
        self.cfg = {
            'exemplar_sz': 127,
            'instance_sz': 271,
            'total_stride': 8,
            'context': 0.5,
            'ratios': [0.33, 0.5, 1, 2, 3],
            'scales': [8, ],
            'penalty_k': 0.055,
            'window_influence': 0.42,
            'lr': 0.295}

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = namedtuple('GenericDict', self.cfg.keys())(**self.cfg)

    def init(self, image, box):
        image = np.asarray(image)

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # for small target, use larger search region
        if np.prod(self.target_sz) / np.prod(image.shape[:2]) < 0.004:
            self.cfg = self.cfg._replace(instance_sz=287)

        # generate anchors
        self.response_sz = (self.cfg.instance_sz - \
                            self.cfg.exemplar_sz) // self.cfg.total_stride + 1
        self.anchors = self._create_anchors(self.response_sz)

        # create hanning window
        self.hann_window = np.outer(
            np.hanning(self.response_sz),
            np.hanning(self.response_sz))
        self.hann_window = np.tile(
            self.hann_window.flatten(),
            len(self.cfg.ratios) * len(self.cfg.scales))

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
                    self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(image, axis=(0, 1))
        exemplar_image = crop_and_resize(
            image, self.center, self.z_sz,
            self.cfg.exemplar_sz, self.avg_color)

        # classification and regression kernels
        exemplar_image = torch.from_numpy(exemplar_image).to(
            self.device).permute([2, 0, 1]).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()
            self.kernel_reg, self.kernel_cls = self.net.learn(exemplar_image)

    def update(self, image):
        image = np.asarray(image)

        # search image
        instance_image = crop_and_resize(
            image, self.center, self.x_sz,
            self.cfg.instance_sz, self.avg_color)

        # classification and regression outputs
        instance_image = torch.from_numpy(instance_image).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()
            out_reg, out_cls = self.net.inference(
                instance_image, self.kernel_reg, self.kernel_cls)

        # offsets
        offsets = out_reg.permute(
            1, 2, 3, 0).contiguous().view(4, -1).cpu().numpy()
        offsets[0] = offsets[0] * self.anchors[:, 2] + self.anchors[:, 0]
        offsets[1] = offsets[1] * self.anchors[:, 3] + self.anchors[:, 1]
        offsets[2] = np.exp(offsets[2]) * self.anchors[:, 2]
        offsets[3] = np.exp(offsets[3]) * self.anchors[:, 3]

        # scale and ratio penalty
        penalty = self._create_penalty(self.target_sz, offsets)

        # response
        response = F.softmax(out_cls.permute(
            1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1].cpu().numpy()
        response = response * penalty
        response = (1 - self.cfg.window_influence) * response + \
                   self.cfg.window_influence * self.hann_window

        # peak location
        best_id = np.argmax(response)
        offset = offsets[:, best_id] * self.z_sz / self.cfg.exemplar_sz

        # update center
        self.center += offset[:2][::-1]
        self.center = np.clip(self.center, 0, image.shape[:2])

        # update scale
        lr = response[best_id] * self.cfg.lr
        self.target_sz = (1 - lr) * self.target_sz + lr * offset[2:][::-1]
        self.target_sz = np.clip(self.target_sz, 10, image.shape[:2])

        # update exemplar and instance sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
                    self.cfg.instance_sz / self.cfg.exemplar_sz

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box

    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            out_reg, out_cls = self.net(z, x)

            # calculate loss
            # labels = self._create_labels(responses.size())
            # loss_cls = self.loss_cls(out_cls)
            # loss_reg = self.loss_reg()

            loss = self.loss_reg()

            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)

        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)

        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()

            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_alexnet_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()

        return self.labels

    def _create_anchors(self, response_sz):
        anchor_num = len(self.cfg.ratios) * len(self.cfg.scales)
        anchors = np.zeros((anchor_num, 4), dtype=np.float32)

        size = self.cfg.total_stride * self.cfg.total_stride
        ind = 0
        for ratio in self.cfg.ratios:
            w = int(np.sqrt(size / ratio))
            h = int(w * ratio)
            for scale in self.cfg.scales:
                anchors[ind, 0] = 0
                anchors[ind, 1] = 0
                anchors[ind, 2] = w * scale
                anchors[ind, 3] = h * scale
                ind += 1
        anchors = np.tile(
            anchors, response_sz * response_sz).reshape((-1, 4))

        begin = -(response_sz // 2) * self.cfg.total_stride
        xs, ys = np.meshgrid(
            begin + self.cfg.total_stride * np.arange(response_sz),
            begin + self.cfg.total_stride * np.arange(response_sz))
        xs = np.tile(xs.flatten(), (anchor_num, 1)).flatten()
        ys = np.tile(ys.flatten(), (anchor_num, 1)).flatten()
        anchors[:, 0] = xs.astype(np.float32)
        anchors[:, 1] = ys.astype(np.float32)

        return anchors

    def _create_penalty(self, target_sz, offsets):
        def padded_size(w, h):
            context = self.cfg.context * (w + h)
            return np.sqrt((w + context) * (h + context))

        def larger_ratio(r):
            return np.maximum(r, 1 / r)

        src_sz = padded_size(
            *(target_sz * self.cfg.exemplar_sz / self.z_sz))
        dst_sz = padded_size(offsets[2], offsets[3])
        change_sz = larger_ratio(dst_sz / src_sz)

        src_ratio = target_sz[1] / target_sz[0]
        dst_ratio = offsets[2] / offsets[3]
        change_ratio = larger_ratio(dst_ratio / src_ratio)

        penalty = np.exp(-(change_ratio * change_sz - 1) * \
                         self.cfg.penalty_k)

        return penalty
