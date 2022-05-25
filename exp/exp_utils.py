#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/04/12, homeway'


import torch
import numpy as np
import os
import os.path as osp
import sys
import inspect
import numpy as np
import torch
import copy
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils import helper
args = helper.get_args()


def plot_roc(result_array, base_path, pos_label=1, fontsize=30, linewidth=6, markersize=20):
    keys = [
        ["sample", "y", "scores"],
        ["query", "query_y", "query_scores"]
    ]

    for step, key in enumerate(keys):
        plt.figure(figsize=(16, 16), dpi=100)
        plt.cla()
        plt.grid()
        fig_path = f"{base_path}_roc_{key[0]}_"
        for method, item in result_array.items():
            fig_path = f"{fig_path}_{method}"
            (y, scores) = item[key[1]], item[key[2]]
            fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=pos_label)
            plt.plot(fpr, tpr, label=method, linewidth=linewidth, markersize=markersize)
        plt.xlabel('FRP', fontsize=fontsize)
        plt.ylabel('TPR', fontsize=fontsize)
        plt.xticks(np.arange(0, 1.2, 0.2), fontsize=fontsize)
        plt.yticks(np.arange(0, 1.2, 0.2), fontsize=fontsize)
        plt.legend(loc='lower right', fontsize=fontsize)

        fig_path += ".pdf"
        print(f"-> saving hist at:{fig_path}")
        plt.savefig(fig_path)


def test_query(fedinet, test_loader, device=args.device):
    scores = []
    fedinet = fedinet.to(device)
    with torch.no_grad():
        for step, (x, y) in enumerate(test_loader):
            pred = fedinet(x.to(device)).argmax(dim=1, keepdim=True)
            scores.append(float(100.0 * pred.sum() / len(x)))
    return np.array(scores)


def build_query_set(data_loader, query_budget, query_size):
    """
    build new query set using random selection
    :param data_loader: torch.data.utils.Dataloader
    :param query_budget: int
    :param query_size: int
    :return: dict {"x": torch.Tensor, "y": torch.Tensor}
    """
    cache_set = {"x": [], "y": []}
    for x, y in data_loader:
        cache_set["x"].append(x)
        cache_set["y"].append(y)
    cache_set["x"] = torch.cat(cache_set["x"])
    cache_set["y"] = torch.cat(cache_set["y"])

    total_size = cache_set["x"].size(0)
    total_query = int(query_budget / query_size)
    query_set = {"x": [], "y": []}
    for i in range(total_query):
        idxs = np.random.randint(low=0, high=total_size, size=[query_size])
        query_set["x"].append(cache_set["x"][idxs].clone())
        query_set["y"].append(cache_set["y"][idxs].clone())
    query_set["x"] = torch.cat(query_set["x"])
    query_set["y"] = torch.cat(query_set["y"])
    return query_set

def build_ben_dataset(data_loader, query_budget, query_size):
    set_x, set_y = [], []
    for step, (x, y) in enumerate(data_loader):
        set_x.append(x)
        set_y.append(y)
    set_x = torch.cat(set_x)
    set_y = torch.cat(set_y)

    total_query = int(query_budget / query_size)
    ben_set_x = []
    ben_set_y = []
    for i in range(total_query):
        idxs = np.random.randint(low=0, high=len(set_x), size=[query_size])
        query_x = set_x[idxs].clone()
        query_y = set_y[idxs].clone()
        ben_set_x.append(query_x)
        ben_set_y.append(query_y)
    ben_set_x = torch.cat(ben_set_x)
    ben_set_y = torch.cat(ben_set_y)
    return ben_set_x, ben_set_y