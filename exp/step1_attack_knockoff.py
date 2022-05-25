#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/05/15, homeway'


import os
import sys
import inspect
import torch
import os.path as osp
import numpy as np
from tqdm import tqdm
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils import helper
from MLaaS.server import BlackBox
from MLaaS.attack import jbda, jbda_ops
from datasets.loader import FediLoader as DataLoader
from models.loader import FediModel
from step1_pretrain import pretrain


def build_svhn_query_cache(svhn_train_loader, victim, base_path, query_size, device):
    cache_set = {"x": [], "y": []}
    for x, y in svhn_train_loader:
        cache_set["x"].append(x)
        cache_set["y"].append(y)
    cache_set["x"] = torch.cat(cache_set["x"])
    cache_set["y"] = torch.cat(cache_set["y"])
    total_size = cache_set["x"].size(0)

    victim.eval()
    victim.to(device)
    phar = tqdm(range(1, 11))
    for idx in phar:
        fname = f"ResNet34_MNIST_CNN_Knockoff-SVHN_{idx}.pt"
        fpath = osp.join(base_path, fname)
        query_set = {"x": [], "y": []}
        num_query_per_cahce = int(50000 / query_size)
        for step in range(num_query_per_cahce):
            idxs = np.random.randint(low=0, high=total_size, size=[query_size])
            x = (cache_set["x"][idxs].clone()[:, :1]).to(device)
            y = victim(x)
            query_set["x"].append(x.detach().cpu())
            query_set["y"].append(y.detach().cpu())
        query_set["x"] = torch.cat(query_set["x"])
        query_set["y"] = torch.cat(query_set["y"])
        torch.save({
            "inputs": query_set["x"],
            "outputs": query_set["y"]
        }, fpath)
        phar.set_description(f"-> save cache file:{fpath}")


def main(args):
    '''victim model'''
    victim = pretrain(args)

    '''victim dataset'''
    loader = DataLoader(data_root=args.data_root, dataset=args.atk_data, batch_size=200)
    train_loader, test_loader = loader.get_loader()
    base_path = osp.join(args.out_path, f"storage_{args.vic_data}")
    build_svhn_query_cache(train_loader, victim, base_path=base_path, query_size=args.query_size, device=args.device)


if __name__ == "__main__":
    args = helper.get_args()
    helper.set_seed(args)
    main(args)