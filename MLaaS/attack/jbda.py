#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/05/15, homeway'

"""
Jacobian-based data augmentation attack.
For the detail please see paper: https://arxiv.org/abs/1602.02697
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from copy import deepcopy


class JBDA:
    def __init__(self, model, optimizer, batch_size, query_size, pred_type="soft", attack_rounds=1000, device=torch.device("cuda:0")):
        self.device = device
        self.pred_type = pred_type
        self.batch_size = batch_size
        self.query_size = query_size
        self.model = model.to(self.device)

        # blackbox return soft label
        self.criterion = F.cross_entropy
        if pred_type == "soft":
            self.criterion = F.mse_loss
        self.optimizer = optimizer
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=attack_rounds)
        print(f"-> optimizer:{self.optimizer}")

    def train(self, data_loader: DataLoader, per_epoch: int = 5):
        self.model.train()
        self.model.to(self.device)
        for epoch in range(per_epoch):
            for step, (x, y) in enumerate(data_loader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()
        self.scheduler.step()
        self.model.eval()

    def extract(self, blackbox, sub_x, sub_y, adv: callable, epoch: int = 0, per_epoch: int = 3, tag: str = None):
        """
        :param blackbox:
        :param sub_x:
        :param sub_y:
        :param adv:
        :param epoch:
        :param per_epoch:
        :param tag:
        :return: model
        """
        self.model.to(self.device)

        # Algorithm1 line 8: Jacobian-based datasets augmentation
        _x = deepcopy(sub_x[-self.query_size:])
        _y = deepcopy(sub_y[-self.query_size:])
        adv_x = adv(_x, _y).cpu()

        # Algorithm1 line 4-6: query MLaaS, label training set
        adv_y = blackbox.query(adv_x, epoch=epoch, tag=tag).cpu()
        if self.pred_type == "hard":
            adv_y = adv_y.argmax(dim=1)

        # build new loader
        sub_x = torch.cat([sub_x.cpu(), adv_x])
        sub_y = torch.cat([sub_y.cpu(), adv_y])
        dst = TensorDataset(sub_x, sub_y)
        data_loader = DataLoader(dst, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.train(data_loader=data_loader, per_epoch=per_epoch)
        return self.model.cpu(), sub_x, sub_y