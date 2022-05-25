#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/03/12, homeway'

import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
from MLaaS.defense import profedi
from utils import helper
args = helper.get_args()
helper.set_seed(args)


def load_anchor_feats(dataset):
    '''
    Load anchor features from .pt cache
    :param dataset: str, dataset name
    :return: torch.array, anchor_features
    '''
    root = osp.join(args.out_root, "fedi")
    if osp.exists(root):
        path = osp.join(args.out_root, f"fedi/anc_{dataset}.pt")
        if osp.exists(path):
            anchor_feats = torch.load(path)
            print(f"-> load anchor_features from:{path}")
            return anchor_feats
    raise FileNotFoundError("Anchor features not found! Please run step3 first!")


def save_anchor_feats(dataset, anchor_feats):
    '''
    save anchor features to .pt cache
    :param dataset: str, dataset name
    :param anchor_feats: torch.array, anchor_features
    :return: anchor_feats: torch.array
    '''
    root = osp.join(args.out_root, "fedi")
    if not osp.exists(root):
        os.makedirs(root)
    path = osp.join(args.out_root, f"fedi/anc_{dataset}.pt")
    print(f"-> save anchor_features to: {path}!")
    torch.save(anchor_feats, path)
    return anchor_feats


def load_fdi_feats(name):
    root = osp.join(args.out_root, "fedi")
    if osp.exists(root):
        path = osp.join(args.out_root, f"fedi/fdi_{name}.pt")
        if osp.exists(path):
            fdi_feats = torch.load(path)
            print(f"-> load FDI feats from:{path}")
            return fdi_feats
    raise FileNotFoundError("FDI feats not found! Please run step3 first!")


def save_fdi_feats(name, fdi_feats, pred_labels, fdi_labels):
    root = osp.join(args.out_root, "fedi")
    if not osp.exists(root):
        os.makedirs(root)
    path = osp.join(args.out_root, f"fedi/fdi_{name}.pt")
    data = {
        "x": fdi_feats,
        "z": pred_labels.long(),
        "y": fdi_labels
    }
    print(f"-> save FDI feats to: {path}!")
    torch.save(data, path)
    return data


def tensor2loader(x, y, batch_size=256, shuffle=False):
    dst = TensorDataset(x.float(), y.long())
    return TorchDataLoader(dst, batch_size=batch_size, shuffle=shuffle, num_workers=1)


def load_query_testset(methods, tag):
    fds_testset_methods = {}
    for method in methods:
        file_path = osp.join(args.out_root, "fedi", f"{tag}_{method}.pt")
        if osp.exists(file_path):
            print(f"-> load from cache:{file_path}")
            fds_testset = torch.load(file_path)
        else:
            raise RuntimeError(f"-> Cache file: {file_path} not found! Please build query set!")
        fds_testset_methods[method] = fds_testset
    return fds_testset_methods



























def load_distortion_cache(name):
    path = osp.join(args.out_root, f"fedi/{name}.pt")
    print(f"-> load feature distortion from:{path}")
    if not osp.exists(path):
        return None
    return torch.load(path, map_location="cpu")


def load_distortion(victim, data_loader, label, anchor_features, name, augmentated=False):
    process = profedi.compute_fdi_from_loader
    if augmentated:
        process = profedi.compute_augmentated_fds_from_loader

    cache = load_distortion_cache(name)
    if cache is None:
        fdi, pred_label = process(victim, data_loader, anchor_features, num_expand=4, eps=0.1)
        y = (torch.ones(len(fdi)) * label)
        cache = {
            "x": fdi.float(),
            "y": y.long(),
            "l": pred_label.long()
        }
        path = osp.join(args.out_root, f"fedi/{name}.pt")
        torch.save(cache, path)
    return cache