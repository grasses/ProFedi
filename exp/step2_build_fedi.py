#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/04/12, homeway'

import os
import os.path as osp
import sys
import inspect
import numpy as np
import torch

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from step1_pretrain import pretrain
from utils import helper, metric
from MLaaS.defense import profedi_ops, profedi
from datasets.loader import FediLoader as DataLoader


def build_queryset_dataloader(args, vic_model, anchor_feats, ben_test_fds, methods, query_size, query_budget):
    """
    这一步建立的是以query_size为大小的fdi的data_loader
    :param victim:
    :param anchor_features:
    :param methods:
    :param query_size:
    :return:
    """
    adv_size = 50000 * int(args.query_range[1] - args.query_range[0])
    assert query_budget == adv_size

    # build query fds for benign client
    total_query = int(query_budget/query_size)

    # benign: 随机抽取ben_test_fds，组建5000个query
    ben_fds_x = []
    ben_fds_l = []
    for i in range(total_query):
        idxs = np.random.randint(low=0, high=len(ben_test_fds["x"]), size=[query_size])
        fdi = ben_test_fds["x"][idxs].clone()
        predict_label = ben_test_fds["z"][idxs].clone()
        ben_fds_x.append(fdi)
        ben_fds_l.append(predict_label)
    ben_fds_x = torch.cat(ben_fds_x)
    ben_fds_l = torch.cat(ben_fds_l)
    ben_fds_y = torch.zeros([len(ben_fds_x)])

    # malicious: 顺序读取用户的query，组建组建5000个query
    storage_root = osp.join(args.out_root, f"storage_{args.vic_data}")
    fds_testset_methods = {}

    for method in methods:
        tag = f"{args.tag}_{method}"
        file_name = f"test_query_{tag}"
        file_path = osp.join(args.out_root, f"fedi/fdi_{file_name}.pt")
        if osp.exists(file_path):
            print(f"-> load from cache:{file_path}")
            fds_testset = torch.load(file_path)
        else:
            print(f"-> cache file: {file_path} not found!")
            adv_fds_x_list = []
            adv_fds_y_list = []
            adv_fds_l_list = []
            for i in range(args.query_range[0], args.query_range[1]):
                ori_query_fdi_path = osp.join(args.out_path, f"fedi/fdi_ori_query_{tag}_{i}.pt")
                if osp.exists(ori_query_fdi_path):
                    print(f"-> for method: {method}_{i}, read from:{ori_query_fdi_path}")
                    adv_fds_xy = torch.load(ori_query_fdi_path)
                    l = adv_fds_xy["z"]
                else:
                    # build fds dataset from query
                    cache_file_name = f"{tag}_{i}.pt"
                    fpath = osp.join(storage_root, cache_file_name)
                    print(f"-> for method: {method}_{i}, read from:{fpath}")
                    cache_query = torch.load(fpath)
                    x = cache_query["inputs"][:50000]
                    l = cache_query["outputs"][:50000].argmax(dim=1)
                    y = torch.ones([len(x)])
                    adv_test_query_loader = profedi_ops.tensor2loader(x=x, y=y, shuffle=True, batch_size=query_size)
                    adv_test_fdi, adv_test_labels = profedi.get_fdi_feats(vic_model, adv_test_query_loader, anchor_feats=anchor_feats)
                    adv_fds_xy = profedi_ops.save_fdi_feats(name=f"ori_query_{tag}_{i}",
                                               fdi_feats=adv_test_fdi,
                                               pred_labels=adv_test_labels,
                                               fdi_labels=y)
                adv_fds_x_list.append(adv_fds_xy["x"])
                adv_fds_y_list.append(adv_fds_xy["y"])
                adv_fds_l_list.append(l)
            adv_fds_x = torch.cat(adv_fds_x_list)
            adv_fds_y = torch.cat(adv_fds_y_list)
            adv_fds_l = torch.cat(adv_fds_l_list)
            fds_testset = {
                "x": torch.cat([ben_fds_x.clone(), adv_fds_x.clone()]),
                "y": torch.cat([ben_fds_y.clone(), adv_fds_y.clone()]),
                "z": torch.cat([ben_fds_l.clone(), adv_fds_l.clone()])
            }
            profedi_ops.save_fdi_feats(name=file_name,
                                       fdi_feats=fds_testset["x"],
                                       pred_labels=fds_testset["z"],
                                       fdi_labels=fds_testset["y"])
            print(f"-> save feature to: {file_path}")
        fds_testset_methods[method] = fds_testset
    return fds_testset_methods


def main(args):
    '''pretrain MLaaS DNN'''
    vic_model = pretrain(args)

    '''step1 load anchor feats'''
    anchor_feats = profedi_ops.load_anchor_feats(dataset=args.vic_data)

    '''step2 load dataset'''
    ben_loader = DataLoader(data_root=args.data_root, dataset=args.vic_data, batch_size=args.batch_size)
    ben_train_loader, ben_test_loader = ben_loader.get_loader()


    print("\n<======================testing dataset==========================>")
    ben_dataset = args.vic_data.upper()


    target_file = osp.join(args.out_path, f"fedi/fdi_test_{ben_dataset}.pt")
    if osp.exists(target_file):
        print(f"-> load cache file:{target_file}")
        ben_test_fds = torch.load(target_file)
    else:
        ben_test_fdi, ben_test_labels = profedi.get_fdi_feats(vic_model, ben_test_loader, anchor_feats=anchor_feats)
        ben_test_fds = {
            "x": ben_test_fdi,
            "z": ben_test_labels,
            "y": torch.zeros([len(ben_test_fdi)])
        }

    fds_queryset_methods = build_queryset_dataloader(
        args, vic_model, anchor_feats=anchor_feats, ben_test_fds=ben_test_fds, methods=args.atk_methods,
        query_budget=args.query_budget, query_size=args.query_size
    )
    return fds_queryset_methods


if __name__ == "__main__":
    args = helper.get_args()
    args.atk_methods = [
        #"JBDA-FGSM",
        "JBDA-DF",
        #"JBDA-PGD",
        #"Knockoff",
        #"DFME",
        #"DaST"
    ]
    main(args)
























