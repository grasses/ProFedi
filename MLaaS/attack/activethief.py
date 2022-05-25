#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/07/01, homeway'

"""
Paper: ActiveThief: Model Extraction Using Active Learning and Unannotated Public Data
For the detail please see: https://ojs.aaai.org/index.php/AAAI/article/view/5432
"""


class ActiveThief(object):
    def __init__(self, args, blackbox, model, train_loader, test_loader):
        self.args = args
        self.blackbox = blackbox
        self.device = args.device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def extract(self, dataloader, rounds):
        pass

    def selection(self, dataset, method=""):
        """
        :param dataset:
        :param method:
        :return:
        """
        pass