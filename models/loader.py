#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/05/20, homeway'


import os
import os.path as osp
import torch
import models


class FediModel:
    def __init__(self, ckpt="model/ckpt"):
        self.ckpt = ckpt

    def get_model(self, arch, task, num_classes=10, pretrain=False):
        m_arch = arch.lower()
        if "lenet" in m_arch:
            from models.lenet import LeNet
            self.model = LeNet(num_classes=num_classes)
        elif "resnet" in m_arch:
            self.model = eval('models.resnet.{}'.format(arch))(num_classes=num_classes)
        elif "vgg" in m_arch:
            self.model = eval('models.vgg.{}'.format(arch))(num_classes=num_classes)
        elif "cnn" in m_arch:
            from models.lenet import LeNet
            self.model = LeNet(num_classes=num_classes)
        else:
            raise NotImplementedError(f"-> model:{arch} not implemented!")

        if ("resnet" in arch.lower()) and (task.upper() == "MNIST"):
            self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)

        if pretrain:
            weight = self.load(arch, task)
            if weight is not None:
                self.model.load_state_dict(weight)
        return self.model


    def save(self, model, arch, task):
        path = osp.join(self.ckpt, f"{arch}_{task}.pt")
        print("-> Save model: {}_{}.pt".format(arch, task))
        return torch.save(model.cpu().state_dict(), path)

    def load(self, arch, task):
        path = osp.join(self.ckpt, f"{arch}_{task}.pt")
        if not os.path.exists(path):
            return None
        print("-> Load pretrain model: {}_{}.pt".format(arch, task))
        return torch.load(path, map_location="cpu")