#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/05/20, homeway'

"""
MLaaS is protected by defense_process
"""

import sys
import torch


class BlackBox(object):
    def __init__(self, model, save_process=None, defense_process=None, device=torch.device("cuda")):
        super(BlackBox, self).__init__()
        self.device = device
        self.model = model.to(self.device)
        self.save_process = save_process
        self.defense_process = defense_process

    def defense(self, inputs, outputs, feature_list, **kwargs):
        inputs = inputs.detach().cpu()
        # cache process
        if self.save_process is not None:
            self.save_process(inputs, outputs, feature_list[-1], **kwargs)

        # defense process
        if self.defense_process is not None:
            outputs = self.defense_process(inputs, outputs, feature_list, **kwargs)
        return outputs

    def __call__(self, inputs):
        with torch.no_grad():
            inputs = inputs.to(self.device)
            return self.model(inputs).detach().cpu()

    def query(self, inputs, **kwargs):
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs, feature_list = self.model.feature_list(inputs)

            # copy to cpu
            for idx in range(len(feature_list)):
                feature_list[idx] = feature_list[idx].detach().cpu()
            inputs = inputs.detach().cpu()
            outputs = outputs.detach().cpu()
            outputs = self.defense(inputs, outputs, feature_list, **kwargs)
        sys.stdout.flush()
        return outputs