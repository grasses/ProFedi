#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/04/27, homeway'


import os
import os.path as osp
import sys
import inspect
import copy
import numpy as np
import torch
from tqdm import tqdm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from utils import helper, metric
from datasets.loader import FediLoader as DataLoader
from exp import exp_utils
from MLaaS.defense import prada
benign_result_dict = {}


def run(args, method, ben_set, shapiro_threshold):
    global benign_result_dict
    cache_base_path = osp.join(args.out_root, f"storage_{args.vic_data.upper()}")
    tag_name = f"{args.vic_model}_{args.vic_data}_{args.atk_model}_{method}"
    N = 1
    gd_agent = prada.PRADA(
        shapiro_threshold=shapiro_threshold,
        dist_metric=prada.l2,
        thr_update_rule=prada.mean_dif_std,
        N=N
    )

    if str(shapiro_threshold) in benign_result_dict.keys():
        result_dict = copy.deepcopy(benign_result_dict[str(shapiro_threshold)])
    else:
        # only run benign set one time
        result_dict = {
            "y": [],
            "pred": [],
            "query_y": [],
            "query_pred": [],
            "conf": [],
            "N": N,
            "shapiro_threshold": shapiro_threshold
        }
        num_ben_query = int(len(ben_set["x"]) / args.query_size)
        pbar = tqdm(range(num_ben_query))
        for query_idx in pbar:
            off = int(query_idx * args.query_size)
            batch_query = ben_set["x"][off: off + args.query_size].cpu().numpy()
            target_classes = ben_set["y"][off: off + args.query_size].cpu().numpy()
            alarm, prada_detect = gd_agent.batch_query(batch_query, target_classes)
            result_dict["pred"].append(prada_detect)
            result_dict["y"].append(np.zeros(len(prada_detect)))
            conf = round(100.0 * (np.sum(prada_detect) / len(prada_detect)), 2)
            result_dict["conf"].append(conf)
            query_pred = 1 if alarm else 0
            result_dict["query_y"].append(0)
            result_dict["query_pred"].append(query_pred)
            pbar.set_description(f"-> [{method}] ben query_fidx:{query_idx} prada(N={N}, threshold={shapiro_threshold}) scores:{conf}")
        benign_result_dict[str(shapiro_threshold)] = copy.deepcopy(result_dict)

    for idx in range(1, args.total_cache+1):
        cache_query_path = osp.join(cache_base_path, f"{tag_name}_{idx}.pt")
        cache = torch.load(cache_query_path)
        num_query = int(len(cache["inputs"]) / args.query_size)
        pbar = tqdm(range(num_query))
        for query_idx in pbar:
            off = int(query_idx * args.query_size)
            batch_query = cache["inputs"][off: off + args.query_size].cpu().numpy()
            target_classes = cache["outputs"][off: off + args.query_size].argmax(dim=1).cpu().numpy()
            alarm, prada_detect = gd_agent.batch_query(batch_query, target_classes)
            result_dict["pred"].append(prada_detect)
            result_dict["y"].append(np.ones(len(prada_detect)))
            conf = round(100.0 * (np.sum(prada_detect) / len(prada_detect)), 2)
            result_dict["conf"].append(conf)
            query_pred = 1 if alarm else 0
            result_dict["query_y"].append(1)
            result_dict["query_pred"].append(query_pred)
            pbar.set_description(f"-> [{method}] fidx:{idx} query_fidx:{query_idx} prada(N={N}, threshold={shapiro_threshold}) scores={conf}")
    result_dict["y"] = np.concatenate(result_dict["y"]).astype(np.int32)
    result_dict["pred"] = np.concatenate(result_dict["pred"]).astype(np.int32)
    result_dict["conf"] = np.array(result_dict["conf"]).astype(np.float64)
    result_dict["query_y"] = np.array(result_dict["query_y"], dtype=np.int32)
    result_dict["query_pred"] = np.array(result_dict["query_pred"], dtype=np.int32)

    y = result_dict["y"]
    p = result_dict["pred"]
    sample_res = metric.multi_mertic(y, p)
    print(
        f"""-> TEST_SAMPLE prada(N={N}, threshold={shapiro_threshold}) ACC:{sample_res['ACC']}% 
                Recall:{sample_res["Recall"]} Precision:{sample_res["Precision"]} F1-score:{sample_res["F1score"]}
                FPR:{sample_res['FPR100']} TPR:{sample_res['TPR100']}""")
    y = result_dict["query_y"]
    p = result_dict["query_pred"]
    query_res = metric.multi_mertic(y, p)
    result_dict.update(query_res)
    print(
        f"""-> TEST_QUERY prada(N={N}, threshold={shapiro_threshold}) ACC:{query_res['ACC']}% 
            Recall:{query_res["Recall"]} Precision:{query_res["Precision"]} F1-score:{query_res["F1score"]}
            FPR:{query_res['FPR100']} TPR:{query_res['TPR100']}""")
    result_path = osp.join(args.out_root, "exp",
                           "exp1.1_PRADA-{:.3f}_q{:04d}_{:s}.pt".format(shapiro_threshold, args.query_size, tag_name))
    torch.save(result_dict, result_path)
    print(f"-> save pred:{result_path}")
    print()
    print()


def main(args):
    print("""\n-> step1: load dataset""")
    loader = DataLoader(data_root=args.data_root, dataset=args.vic_data, batch_size=1000)
    ben_train_loader, ben_test_loader = loader.get_loader()
    ben_x, ben_y = exp_utils.build_ben_dataset(data_loader=ben_test_loader, query_budget=args.query_budget,
                                                      query_size=args.query_size)
    ben_set = {
        "x": ben_x,
        "y": ben_y
    }
    for shapiro_threshold in args.threshold_list:
        for method in args.atk_methods:
            tag_name = f"{args.vic_model}_{args.vic_data}_{args.atk_model}_{method}"
            res_path = osp.join(
                args.out_root, "exp_1.1",
                "exp1.1_PRADA_{:s}_{:.3f}_q{:04d}_{:s}.pt".format(
                    args.vic_data,
                    shapiro_threshold, args.query_size, tag_name
                )
            )
            run(args, method, ben_set, shapiro_threshold)
        print()
        print(f"-> shapiro_threshold:{shapiro_threshold}")


if __name__ == "__main__":
    args = helper.get_args()
    args.total_cache = 10

    if args.vic_data == "CIFAR10":
        if args.query_size == 100:
            args.threshold_list = [0.99]
        elif args.query_size == 1000:
            args.threshold_list = [0.96]

    elif args.vic_data == "MNIST":
        if args.query_size == 100:
            args.threshold_list = [0.99]
        elif args.query_size == 1000:
            args.threshold_list = [0.95]
    main(args)








