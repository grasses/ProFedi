#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/05/15, homeway'


import os
import sys
import os.path as osp
import inspect
import torch
from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
import numpy as np
import itertools as it
from copy import deepcopy
from sklearn.manifold import TSNE
from scipy import stats
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from step1_pretrain import pretrain
from models.loader import FediModel
from utils import helper, metric
from MLaaS.defense import profedi_ops, profedi
from datasets.loader import FediLoader as DataLoader
args = helper.get_args()
helper.set_seed(args)
random_state=args.seed


"""
step1.模拟100个agents
step2.模拟2-5个adversaries，剩下的为benign用户
step3.binary 测试
step4.colluding detection测试
"""


def simulate_agent(test_fds, agent_num, query_num, query_size, ben=True):
    """随机抽取test_fds，用于模拟agent_num个adv/ben用户的请求"""
    agent_query = {}
    if ben == True:
        y_fn = torch.zeros
        print("-> start to simulate benign agent")
    else:
        y_fn = torch.ones
        print("-> start to simulate malicious agent")
    for i in range(agent_num):
        if i % 20 == 0:
            print(f"-> simulate agent {i}")
        idxs = np.random.randint(low=0, high=len(test_fds["x"]), size=[query_size*query_num])
        fdi = test_fds["x"][idxs].clone()
        predict_label = test_fds["z"][idxs].clone()
        agent_query[i] = {
            "x": fdi,
            "y": y_fn(len(fdi)),
            "z": predict_label
        }
    return agent_query


def simulate_mal_agent(method, agent_num, query_num=1000, query_size=100):
    """
    顺序读取用户的query，组建组建query_num个query
    用于模拟agent_num个adv用户的请求
    """
    file_name = f"fdi_test_query_{args.tag}_{method}"
    file_path = osp.join(args.out_root, f"fedi/{file_name}.pt")
    if osp.exists(file_path):
        print(f"-> load from cache:{file_path}")
        adv_test_fds = torch.load(file_path)
        adv_agent_query = simulate_agent(adv_test_fds, agent_num=agent_num, query_num=query_num, query_size=query_size, ben=False)
    else:
        print("-> adv cache query not found!!")
        exit(1)
    return adv_agent_query


def binary_detection(classifier, agent_query_x, tau1):
    x = agent_query_x.to(args.device)
    output = classifier(x)
    pred = output.argmax(dim=1, keepdim=True)
    pred_conf = float(1.0 * torch.sum(pred.view(-1)) / len(pred))
    pred_vote = 1 if pred_conf > tau1 else 0
    return int(pred_vote)


def run_profedi(args, agent_num, max_adv_num=10, query_num=100, query_size=100, feature_size=5):
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

    print("-> load benign dataset\n")
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

    print("-> step4: Load FeDINet from pretrained model")
    from models.fedinet import FeDINet
    fedinet = FeDINet()
    fedinet.train()
    fedinet = fedinet.to(args.device)
    # load FeDINet from cache
    model_loader = FediModel(ckpt=args.ckpt_root)
    weights = model_loader.load(arch="FeDINet", task=args.vic_data)
    if weights is None:
        raise RuntimeError("-> Please Train ProFedi first!!!")
    fedinet.load_state_dict(weights)

    print("""-> step5.1: random agent_num+1 benign agents""")
    ben_agent_query = simulate_agent(ben_test_fds, agent_num=agent_num+1, query_num=query_num, query_size=query_size)

    # per query_size = query_size * feature_size = 100 * 500 = 50000
    # per agent select query num = query_num = 100
    print(f"-> per query_size = query_size * feature_size = 100 * 500 = {query_size * feature_size}")
    print(f"-> per agent select query num = {query_num}")

    # c is benign client
    # c = ben_agent_query[agent_num]["x"].view(-1, query_size*feature_size).numpy()
    c = ben_agent_query[agent_num]["x"].clone()
    c = torch.mean(c, dim=2).view(-1, feature_size * query_size).numpy()

    # 对所有benign agent预测
    tmp_binary_label = torch.zeros([agent_num])
    tmp_binary_pred = torch.zeros([agent_num])
    tmp_pvalue_matrix = torch.zeros([agent_num, agent_num], dtype=torch.float32)
    for idx in range(agent_num):
        tmp_binary_pred[idx] = binary_detection(classifier=fedinet, agent_query_x=ben_agent_query[idx]["x"], tau1=args.tau1)

    for method in args.atk_methods:
        saved_result = {}
        method_result_path = args.result_path + f"_{method}.pt"
        if osp.exists(method_result_path):
            saved_result = torch.load(method_result_path)

        print(f"""-> step5.2: random 2~{max_adv_num} malicious agents""")
        for adv_agent_num in range(2, max_adv_num+1):
            saved_result[adv_agent_num] = {}

            print(f"-> step5.2: current adv agent={adv_agent_num}")
            adv_agent_query = simulate_mal_agent(method, agent_num=adv_agent_num, query_num=query_num, query_size=query_size)

            print("""-> step5.3: binary testing""")
            # 从上面导入预测结果
            adv_start_idx = (agent_num - adv_agent_num)
            binary_pred = deepcopy(tmp_binary_pred)
            binary_label = deepcopy(tmp_binary_label)
            binary_label[adv_start_idx:] = 1

            # 对所有adv agent的query预测
            for idx in range(adv_agent_num):
                binary_pred[adv_start_idx+idx] = binary_detection(classifier=fedinet, agent_query_x=adv_agent_query[idx]["x"], tau1=args.tau1)
            saved_result[adv_agent_num]["binary_pred"] = binary_pred
            saved_result[adv_agent_num]["binary_label"] = binary_label

            print("""-> step5.4: select predict malicious agent""")
            mal_idx = np.reshape(np.argwhere(binary_pred.numpy() == 1), [-1])
            group = it.combinations(mal_idx, 2)

            ii = -1
            print("""-> step5.5: colluding testing""")
            for ii, (i, j) in enumerate(group):
                if i >= adv_start_idx:
                    print(f"-> method:{method} [{adv_agent_num}/{max_adv_num}] a agent idx", i - adv_start_idx)
                    a = adv_agent_query[i - adv_start_idx]["x"].cpu().clone()
                else:
                    a = ben_agent_query[i]["x"].cpu().clone()
                a = torch.mean(a, dim=2).view(-1, feature_size * query_size).numpy()

                if j >= adv_start_idx:
                    print(f"-> method:{method} [{adv_agent_num}/{max_adv_num}] b agent idx", j - adv_start_idx)
                    b = adv_agent_query[j - adv_start_idx]["x"].clone()
                else:
                    b = ben_agent_query[j]["x"].clone()
                b = torch.mean(b, dim=2).view(-1, feature_size * query_size).numpy()

                samples = np.concatenate([deepcopy(a), deepcopy(b), deepcopy(c)])
                down_samples = TSNE(n_components=1, n_iter=1000, random_state=random_state).fit_transform(samples)
                dist1 = down_samples[: query_num]
                dist2 = down_samples[query_num: 2 * query_num]
                res = stats.ttest_ind(dist1, dist2, equal_var=True)
                tmp_pvalue_matrix[i, j] = float(res.pvalue)
                print(f"-> method:{method} [{adv_agent_num}/{max_adv_num}] agent idx:{(i, j)}, p-test:{res.pvalue}\n")
                saved_result[adv_agent_num]["p_value"] = tmp_pvalue_matrix

            print(f"-> method:{method} [{adv_agent_num}/{max_adv_num}] adv_agent:{adv_agent_num} binary detected:{mal_idx} total_group:{ii+1}")
            saved_result[adv_agent_num]["group"] = np.array(list(it.combinations(mal_idx, 2)))
            torch.save(saved_result, method_result_path)


def show_result(args, agent_num, max_adv_num, tau2=0.05, read_cache=False):
    ploted_result = {}
    if osp.exists(args.result_path) and read_cache:
        ploted_result = torch.load(args.result_path)
    else:
        for method in args.atk_methods:
            ploted_result[method] = {
                "ACC": [],
                "Recall": [],
                "Precision": [],
                "F1score": []
            }
            method_result_path = args.result_path + f"_{method}.pt"
            saved_result = torch.load(method_result_path)
            for exp_idx in range(2, max_adv_num+1):
                print(f"-> method:{method} idx:{exp_idx}")
                group = it.combinations(np.arange(agent_num), 2)
                per_result = saved_result[exp_idx]
                binary_label = per_result["binary_label"]
                binary_pred = per_result["binary_pred"]
                p_value = per_result["p_value"]
                colluding_label = torch.zeros([agent_num, agent_num])
                colluding_pred = torch.zeros([agent_num, agent_num])

                for (i, j) in deepcopy(group):
                    """update colluding_label"""
                    if (binary_label[i] == 1) and (binary_label[j] == 1):
                        colluding_label[i, j] = 1
                        colluding_label[j, i] = 1
                    """update colluding_pred"""
                    if (binary_pred[i] == 1) and (binary_pred[j] == 1):
                        pred_pvalue = torch.max(p_value[i, j], p_value[j, i])
                        if pred_pvalue > tau2:
                            colluding_pred[i, j] = 1
                            colluding_pred[j, i] = 1

                TP, FP, TN, FN = 0, 0, 0, 0
                for (i, j) in deepcopy(group):
                    if (colluding_label[i, j] == 0) and (colluding_pred[i, j] == 0):
                        TP += 1
                    elif (colluding_label[i, j] == 0) and (colluding_pred[i, j] == 1):
                        FN += 1
                    elif (colluding_label[i, j] == 1) and (colluding_pred[i, j] == 1):
                        TN += 1
                    else:
                        FP += 1
                ACC = 100.0 * (TP + TN) / (TP + FP + TN + FN)
                Recall = 100.0 * (TP) / (TP + FN)
                Precision = 100.0 * (TP) / (TP + FP)
                F1score = (2.0 * Precision*Recall) / (Precision+Recall)
                ploted_result[method]["ACC"].append(ACC)
                ploted_result[method]["Recall"].append(Recall)
                ploted_result[method]["Precision"].append(Precision)
                ploted_result[method]["F1score"].append(F1score)
            print(f"-> method:{method} result:{ploted_result[method]}")
            torch.save(ploted_result, args.ploted_path + "_result.pt")

    for metric in ["ACC", "Recall", "Precision", "F1score"]:
        view_methods = []
        conf_array = {}
        for method in args.atk_methods:
            view_methods.append(method)
            conf_array[method] = np.array(ploted_result[method][metric])
        fig_path = f"{args.ploted_path}_curve_metric{metric}.pdf"
        plot_multiple_acc_curve(conf_array, view_methods, fig_path, metric=metric, max_adv_num=max_adv_num, fontsize=50)


def plot_multiple_acc_curve(res_array, methods, fig_path, metric, max_adv_num, linewidth=6, fontsize=50, markersize=20):
    markers = ["o", "*", "x", "v", "s", "d"]
    plt.figure(figsize=(16, 16), dpi=100)
    plt.cla()
    plt.grid()

    for idx, method in enumerate(methods):
        y = res_array[method]
        x = np.arange(2, max_adv_num+1, 1)
        plt.plot(x, y, label=method, linewidth=linewidth, markersize=markersize, marker=markers[idx])
    plt.plot(x, np.ones(y.shape) * 99.2, label="", color="white")

    # plt.legend(loc='lower right', fontsize=fontsize)
    # plt.xlabel('Number of Colluding Adversaries', fontsize=fontsize)
    # plt.ylabel(f'Colluding Detection {metric} (%)', fontsize=fontsize)
    plt.yticks([99.2, 99.4, 99.6, 99.8, 100], fontsize=fontsize)
    plt.xticks(np.arange(2, max_adv_num+1, 2), fontsize=fontsize)
    print(f"-> saving fig at:{fig_path}")
    plt.savefig(fig_path)


def main(args):
    args.base_root = osp.join(args.out_root, f"exp_2.2")
    args.result_path = osp.join(args.base_root, f"exp_2.2_{args.vic_data}_multiple_agents_result")
    args.ploted_path = osp.join(args.base_root, f"exp_2.2_{args.vic_data}_multiple_agents_ploted")
    # step1: run ProFeDI
    run_profedi(args=args, agent_num=100, max_adv_num=20, query_size=args.query_size, query_num=args.query_num)
    # step2: show result
    show_result(args=args, agent_num=100, max_adv_num=20, tau2=args.tau2)


if __name__ == "__main__":
    args = helper.get_args()
    # use query_num*query_size = 10000 samples to detect agent1 and agent2
    args.query_num = 100
    args.query_size = 100
    if args.vic_data == "MNIST":
        args.tau1 = 0.12
    elif args.vic_data == "CIFAR10":
        args.tau1 = 0.25
    main(args)


















