#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/05/15, homeway'

"""
interface: Model Extraction Warning in MLaaS Paradigm
"""

import torch
import copy
import torch.nn.functional as F


class ExtractionWarning:
    def __init__(self, victim_model, proxy_model, device=torch.device("cuda")):
        self.victim_model = copy.deepcopy(victim_model.cpu())
        self.proxy_model = copy.deepcopy(proxy_model.cpu())
        self.device = device
        self.agent_model = {}
        self.agent_optim = {}
        self.agent_count = {}
        self.history_dict = {}

    def batch_query(self, x, y, user, **kwargs):
        """
        :param x:
        :param y:
        :param kwargs:
        :return:
        """
        if user not in self.agent_model.keys():
            self.agent_count[user] = 0
            self.agent_model[user] = self.get_agent(user)
            self.agent_optim[user] = torch.optim.SGD(self.agent_model[user].parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
            self.history_dict[user] = {"x": [], "y": []}

        self.history_dict[user]["x"].append(x)
        self.history_dict[user]["y"].append(y)
        self.agent_model[user] = self.train_proxy(user=user, query_data={"x": x, "y": y})
        self.agent_count[user] += 1

    def get_agent(self, user):
        if user in self.agent_model.keys():
            return self.agent_model[user]
        return copy.deepcopy(self.proxy_model)

    def train_proxy(self, user, query_data, epochs=2):
        proxy_model = self.agent_model[user].to(self.device)
        proxy_model.train()
        optimizer = self.agent_optim[user]
        for epoch in range(epochs):
            x = query_data["x"].to(self.device)
            y = query_data["y"].to(self.device)
            if len(y.shape) > 1:
                y = y.argmax(dim=1)

            optimizer.zero_grad()
            pred = proxy_model(x)
            loss = F.cross_entropy(pred, y)
            loss.backward()
            optimizer.step()
        proxy_model.eval()
        return proxy_model


















