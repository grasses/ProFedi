#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/05/24, homeway'


import torch
import torch.nn.functional as F
from utils import helper
helper.set_default_seed()


def loss_soft_kd(outputs, teacher_outputs, temperature=10.0, alpha=0.003):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    T = temperature
    labels = teacher_outputs.argmax(dim=1).long()
    KD_loss = (alpha * T * T) * torch.nn.KLDivLoss(reduction="sum")(
        F.log_softmax(outputs / T, dim=1),
        F.softmax(teacher_outputs / T, dim=1)
    ) + (1. - alpha) * F.cross_entropy(outputs, labels)
    return KD_loss