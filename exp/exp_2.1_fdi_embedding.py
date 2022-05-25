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
import time
from sklearn.manifold import TSNE
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from tqdm import tqdm
from multiprocessing import Process

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from step1_pretrain import pretrain
from utils import helper, metric
from MLaaS.defense import profedi_ops, profedi
from datasets.loader import FediLoader as DataLoader


def plot_confusion_matrix(matrix, legends, fig_path, fontsize=20):
    mpl.style.use('seaborn')
    # sum = matrix.sum()
    # conf_arr = matrix * 100.0 / (1.0 * sum)
    df_cm = pd.DataFrame(matrix,
                         index=legends,
                         columns=legends)
    plt.figure(figsize=(16, 14), dpi=100)
    plt.clf()
    plt.grid()
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    res = sn.heatmap(df_cm, linewidths=0.8, linecolor='black', annot=True, vmin=0.0, vmax=1, fmt='.4g', cmap=cmap, annot_kws={'size': 20})
    res.invert_yaxis()

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    #plt.title('Confusion Matrix of p-value', fontsize=fontsize+10)
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()


def plot_embedding_fedi(pos, legends, file_path, fontsize=40, **kwargs):
    size = len(legends)
    split = int(pos.shape[0] / size)
    plt.figure(figsize=(16, 16), dpi=100)
    plt.cla()
    plt.grid()
    for i in range(size):
        off = int(i * split)
        legend = legends[i].lower()
        if legend == "benign" or ("test" in legend):
            plt.scatter(pos[off:off + split, 0], pos[off:off + split, 1], lw=4, s=60, marker="*", label=legends[i])
        else:
            plt.scatter(pos[off:off + split, 0], pos[off:off + split, 1], lw=4, s=60, label=legends[i])

    if "epoch" in kwargs.keys():
        x_pos = [np.min(pos[:, 0]) - 5, np.max(pos[:, 0]) + 5]
        y_pos = [np.min(pos[:, 1]) - 5, np.max(pos[:, 1]) + 5]
        plt.text(x_pos[0]+5, y_pos[1]-5, f'Epoch={kwargs["epoch"]}', fontsize=fontsize)

    plt.xlim(-30, 40)
    plt.ylim(-30, 30)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(loc="lower right", numpoints=1, fontsize=fontsize)
    plt.savefig(file_path)
    print(f"-> saving fig: {file_path}")


def run_embedding(args, fds_testset_methods, collusion_param, base_path):
    '''
    :param fds_testset_methods: {method, [ben_array, agent1, agent2...]}
    :param collusion_param:
    :param query_size:
    :param history_size:
    :param random_state:
    :return:
    '''
    random_state = args.seed
    fds_array = []
    collusion_method = []
    group = "Benign1-"

    """
    50000 samples/agent * 10 agents = 500000 adv samples
    100 history_size * 500 samples/point = 
    max_agents = 10 per attack
    """
    for step, (method, fds_feat) in enumerate(fds_testset_methods.items()):
        # fds_query_set = fds_feat["x"].view(-1, args.query_size * 500).cpu().clone().numpy()
        fds_query_set = torch.reshape(torch.mean(fds_feat["x"].cpu(), 2), [-1, 5*500]).numpy()
        if not method in collusion_param.keys():
            continue
        if step == 0:
            collusion_method.append("Benign")
            fds_array.append(fds_query_set[:args.history_size])

        agent_size = collusion_param[method]
        if int((len(fds_query_set) / 2) / args.history_size) < agent_size:
            raise RuntimeError("-> not enough queries")

        for idx in range(agent_size):
            off = args.history_size * (idx+14) + 1
            agent_name = f"{method}_Agent{idx+1}"
            agent_array = fds_query_set[off: off+args.history_size]
            fds_array.append(agent_array)
            collusion_method.append(agent_name)
            group += f"{method}{idx+1}-"
    fds_array = np.concatenate(fds_array)

    pos = TSNE(n_components=2, n_iter=1200, random_state=random_state).fit_transform(fds_array)
    file_path = f"{base_path}_h{args.history_size}_g{group[:-1]}.pdf"
    plot_embedding_fedi(pos, legends=collusion_method, fontsize=45, file_path=file_path)



def run_hypothesis_test(args, fedi_set):
    random_state = args.seed
    off_size = args.history_size

    count = 0
    result = 0.0
    range_i = range(10)
    range_j = range(10)
    phar = tqdm(range(100))
    methods = list(fedi_set.keys())
    if len(fedi_set.keys()) == 1:
        methods = {
            0: methods[0],
            1: methods[0]
        }
        range_i = range(5)
        range_j = range(5, 10)
        phar = tqdm(range(25))

    for i in range_i:
        off_i = int(off_size * i)
        a = fedi_set[methods[0]][off_i: off_i + off_size]
        for j in range_j:
            off_j = int(off_size * i)
            b = fedi_set[methods[1]][off_j: off_j + off_size]
            embedding = TSNE(n_components=2, n_iter=1000, random_state=random_state).fit_transform(np.concatenate([a, b]))
            test_res = stats.ttest_ind(embedding[:off_size], embedding[off_size:], equal_var=True)
            result += test_res.pvalue[0]
            count += 1
            phar.update(1)
            phar.set_description(f"-> [{methods[0]} vs {methods[1]}]({i}, {j})={test_res.pvalue[0]}")
    result /= (1.0 * count)
    return result


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

    print("-> load malicious dataset\n")
    path = osp.join(args.out_path, f"fedi/fdi_test_{args.vic_data}.pt")
    if not osp.exists(path):
        print("\n<======================testing dataset==========================>")
        ben_test_fdi, ben_test_labels = profedi.get_fdi_feats(vic_model, ben_test_loader, anchor_feats=anchor_feats)
        profedi_ops.save_fdi_feats(name=f"test_{args.vic_data}",
                                   fdi_feats=ben_test_fdi,
                                   pred_labels=ben_test_labels,
                                   fdi_labels=torch.zeros(len(ben_test_labels)))
        ben_test_fds = {
            "x": ben_test_fdi,
            "y": torch.zeros(len(ben_test_fdi)),
            "z": ben_test_labels,
        }
    else:
        ben_test_fds = profedi_ops.load_fdi_feats(name=f"train_{args.vic_data}")

    from exp.step2_build_fedi import build_queryset_dataloader
    fds_testset_methods = build_queryset_dataloader(
        args=args, vic_model=vic_model, anchor_feats=anchor_feats, ben_test_fds=ben_test_fds, methods=args.atk_methods,
        query_budget=args.query_budget, query_size=args.query_size
    )

    """<=================================plot embedding=================================>"""
    num_methods = len(args.atk_methods)
    print("-> plot fedi embedding, method1 vs method2, 111")
    for i in range(num_methods):
        for j in range(num_methods):
            if i == j:
                continue
            collusion_param = {}
            for method in args.atk_methods:
                collusion_param[method] = 0
            collusion_param[args.atk_methods[i]] = 1
            collusion_param[args.atk_methods[j]] = 1
            print(f"-> run with params: {collusion_param}")
            base_path = osp.join(args.out_root, "exp_2.1", f"exp_2.1_embedding_111_{args.vic_data}")
            run_embedding(args, fds_testset_methods, collusion_param, base_path=base_path)
            print()


    print("-> plot fedi embedding, method1-agent1 vs method1-agent2, 120")
    for i in range(num_methods):
        collusion_param = {}
        for method in args.atk_methods:
            collusion_param[method] = 0
        collusion_param[args.atk_methods[i]] = 2
        base_path = osp.join(args.out_root, "exp_2.1", f"exp_2.1_embedding_120_{args.vic_data}")
        run_embedding(args, fds_testset_methods, collusion_param, base_path=base_path)
    print()


    """<=================================plot confusion_matrix=================================>"""
    fedi_set = {}
    for step, (method, fds_feat) in enumerate(fds_testset_methods.items()):
        fds_feat = torch.reshape(torch.mean(fds_feat["x"].cpu(), 2), [-1, 5*500]).numpy()
        size = int(len(fds_feat) / 2)
        fedi_set[method] = fds_feat[-size:]
    fedi_set["Benign"] = fds_feat[:size]

    methods = copy.deepcopy(args.atk_methods) + ["Benign"]
    num_methods = len(methods)

    _matrix = {}
    matrix = torch.zeros([num_methods, num_methods])
    for i in range(num_methods):
        _matrix[i] = []
        for j in range(num_methods):
            group_fedi = {}
            group_fedi[methods[i]] = fedi_set[methods[i]]
            group_fedi[methods[j]] = fedi_set[methods[j]]

            p_value = run_hypothesis_test(args, group_fedi)
            _matrix[i].append(p_value)
            matrix[i, j] = p_value
            print(f"-> run hypothesis_test for: {methods[i]}-{methods[j]} avg p-value={matrix[i, j]}")
            file_path = osp.join(args.out_root, "exp_2.1", f"exp_2.1_cm_{args.vic_data}.pt")
            torch.save(matrix, file_path)
    print()
    print(_matrix)
    fig_path = osp.join(args.out_root, "exp_2.1", f"exp_2.1_cm_fig_{args.vic_data}.pt")
    plot_confusion_matrix(matrix.numpy(), legends=methods, fig_path=fig_path)


if __name__ == "__main__":
    args = helper.get_args()
    helper.set_seed(args)
    args.query_size = 100
    args.history_size = 100
    # with a total of 100*100=10000 samples
    main(args)

    data = torch.load("output/exp_2.1/exp_2.1_cm_CIFAR10.pt").numpy()
    methods = args.atk_methods + ["Benign"]
    fig_path = osp.join(args.out_root, "exp_2.1", f"exp_2.1_cm_fig_CIFAR10.pdf")
    plot_confusion_matrix(data, methods, fig_path)


    data = torch.load("output/exp_2.1/exp_2.1_cm_MNIST.pt").numpy()
    methods = args.atk_methods + ["Benign"]
    fig_path = osp.join(args.out_root, "exp_2.1", f"exp_2.1_cm_fig_MNIST.pdf")
    plot_confusion_matrix(data, methods, fig_path)



























