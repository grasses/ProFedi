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
import copy
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from step1_pretrain import pretrain
from utils import helper, metric
from MLaaS.defense import profedi_ops, profedi
from datasets.loader import FediLoader as DataLoader


def plot_solo_comparison_distortion_hist(data_array, methods, base_path, xlims, fontsize=40):
    """
    plot benign & malicious distortion hist
    :param data_array: dict
    :param methods: list
    :param base_path: str
    :return: NOne
    """
    layer_num = int(data_array["Benign"].shape[1])
    for layer in range(layer_num):
        _min, _max = 0, 0
        ben_array = (data_array["Benign"][:, layer].mean(dim=1)).clone().cpu().numpy()
        _max = np.max(ben_array)
        for method in methods:
            mal_array = (data_array[method][:, layer].mean(dim=1)).clone().cpu().numpy()
            _max = np.max(mal_array)

            plt.cla()
            plt.figure(figsize=(12, 12), dpi=100)
            bins = np.linspace(xlims[layer][0], xlims[layer][1], 50)
            kwargs = dict(
                stacked=True,
                histtype="barstacked",
                ec="black",
                alpha=0.7,
            )
            plt.hist([ben_array], bins, label=["Benign"], **kwargs)
            kwargs["alpha"] = 0.5
            plt.hist([mal_array], bins, label=[method], **kwargs)
            plt.legend(loc='upper left', fontsize=fontsize)
            plt.xlabel(f'Feature Distortion Index for Layer{layer+1}', fontsize=fontsize)
            plt.ylabel('Count', fontsize=fontsize)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            save_path = f"{base_path}_solo_{method}_l{layer + 1}.pdf"
            print(f"-> saving hist at:{save_path}")
            plt.savefig(save_path)


def plot_comparison_distortion_hist(data_array, methods, base_path, fontsize=22):
    """
    plot benign & malicious distortion hist
    :param data_array: dict
    :param methods: list
    :param base_path: str
    :return: NOne
    """
    labels = copy.deepcopy(methods)
    labels = ["Benign"] + labels
    layer_num = int(data_array["Benign"].shape[1])

    _min, _max = 0, 0
    print("-> keys:", data_array.keys())
    for layer in range(layer_num):
        layer_dict = {}
        layer_dict["Benign"] = (data_array["Benign"][:, layer].mean(dim=1)).clone().cpu().numpy()
        _max = np.max(layer_dict["Benign"])
        for method in methods:
            layer_dict[method] = (data_array[method][:, layer].mean(dim=1)).clone().cpu().numpy()
            _max = np.max(layer_dict[method])
        plt.cla()
        plt.figure(figsize=(10, 10), dpi=100)
        bins = np.linspace(_min, int(_max)+10, 50)
        kwargs = dict(
            stacked=True,
            histtype="barstacked",
            ec="black",
            alpha=0.4,
        )
        plt.hist(layer_dict.values(), bins, label=labels, **kwargs)
        plt.legend(loc='upper left', fontsize=fontsize)
        plt.xlabel('Feature Distortion', fontsize=fontsize)
        plt.ylabel('Count', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        save_path = base_path + f"_collection_l{layer + 1}_all.pdf"
        print(f"-> saving hist at:{save_path}")
        plt.savefig(save_path)


def main(args):
    # 1.load victim model
    print("-> step1: load model")
    vic_model = pretrain(args)
    vic_model = vic_model.to(args.device)

    # 2.load train_loader
    print("-> step2 load dataset")
    ben_loader = DataLoader(data_root=args.data_root, dataset=args.vic_data, batch_size=args.batch_size)
    ben_train_loader, ben_test_loader = ben_loader.get_loader()

    # 3.load feature distortion
    print("\n-> step3: load anchor feature")
    anchor_feats = profedi_ops.load_anchor_feats(dataset=args.vic_data)

    # 4.build feature distortion
    ben_dataset = args.vic_data
    path = osp.join(args.out_path, f"fedi/fdi_test_{args.vic_data}.pt")
    if not osp.exists(path):
        print("\n<======================testing dataset==========================>")
        ben_test_fdi, ben_test_labels = profedi.get_fdi_feats(vic_model, ben_test_loader, anchor_feats=anchor_feats)
        profedi_ops.save_fdi_feats(name=f"test_{ben_dataset}",
                                   fdi_feats=ben_test_fdi,
                                   pred_labels=ben_test_labels,
                                   fdi_labels=torch.zeros(len(ben_test_labels)))
        ben_test_set = {
            "x": ben_test_fdi,
            "y": torch.zeros(len(ben_test_fdi)),
            "z": ben_test_labels,
        }
    else:
        # load from cache
        ben_test_set = profedi_ops.load_fdi_feats(name=f"test_{ben_dataset}")

    if ben_dataset == "MNIST":
        xlims = [
            [50, 400],
            [100, 700],
            [50, 650],
            [50, 450],
            [0, 180],
        ]
    elif ben_dataset == "CIFAR10":
        xlims = [
            [10, 40],
            [20, 60],
            [10, 40],
            [6, 18],
            [0, 30],
        ]
    methods = [
        "JBDA-FGSM",
        "JBDA-DF",
        "JBDA-PGD",
        "Knockoff",
        "DFME",
        "DaST"
    ]

    from exp.step2_build_fedi import build_queryset_dataloader
    fds_testset_methods = build_queryset_dataloader(
        args=args, vic_model=vic_model, anchor_feats=anchor_feats, ben_test_fds=ben_test_set, methods=methods,
        query_budget=args.query_budget, query_size=args.query_size
    )
    data_array = {}
    base_path = osp.join(args.out_root, f"exp_1.2/hist_{args.vic_data}")
    for method, fds_feat in fds_testset_methods.items():
        print(f"-> test for query from: {method} query_size:{args.query_size}")
        data_array[method] = fds_feat["x"][-20000:-10000].cpu().clone()
        data_array["Benign"] = fds_feat["x"][:10000].cpu().clone()

    plot_solo_comparison_distortion_hist(data_array=data_array, methods=methods, xlims=xlims, base_path=base_path)
    plot_comparison_distortion_hist(data_array=data_array, methods=methods, base_path=base_path)



if __name__ == "__main__":
    args = helper.get_args()
    main(args)