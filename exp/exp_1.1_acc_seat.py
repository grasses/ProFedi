#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/04/12, homeway'


import os
import os.path as osp
import sys
import inspect
import copy
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Process

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from models.vgg import vgg16_bn, vgg11_bn
from utils import helper, metric
from datasets.loader import FediLoader as DataLoader
from exp import exp_utils
from MLaaS.defense import seat, trans
benign_result_dict = {}
global_result_dict = {}


def run(args, encoder, method, ben_set, bounds, score_threshold=0.92):
    global benign_result_dict
    global global_result_dict
    encoder = encoder.to(args.device)
    SEAT = seat.SEAT(encoder, delta=args.delta, bounds=bounds, device=args.device, score_threshold=score_threshold)
    cache_base_path = osp.join(args.out_root, f"storage_{args.vic_data.upper()}")
    tag_name = f"{args.vic_model}_{args.vic_data}_{args.atk_model}_{method}"


    if str(score_threshold) in benign_result_dict.keys():
        result_dict = copy.deepcopy(benign_result_dict[str(score_threshold)])
    else:
        result_dict = {
            "y": [],
            "pred": [],
            "query_y": [],
            "query_pred": [],
            "conf": [],
            "delta": SEAT.delta,
            "score_threshold": SEAT.score_threshold
        }
        num_ben_query = int(len(ben_set["x"]) / args.query_size)
        pbar = tqdm(range(num_ben_query))
        for query_idx in pbar:
            off = int(query_idx * args.query_size)
            batch_query = ben_set["x"][off: off + args.query_size].cpu()
            alarm, seat_pred, dist = SEAT.query(batch_query)
            result_dict["pred"].append(seat_pred)
            result_dict["y"].append(np.zeros(len(seat_pred)))
            conf = round(100.0 * (1.0 - np.sum(seat_pred) / seat_pred.shape[0]), 2)
            result_dict["conf"].append(conf)
            query_pred = 1 if alarm else 0
            result_dict["query_y"].append(0)
            result_dict["query_pred"].append(query_pred)
            pbar.set_description(f"-> [{method}(τ1={score_threshold}, delta={args.delta})] ben query_fidx:{query_idx} SEAT({SEAT.delta}) conf={conf}%")
            pbar.update(1)
        global_result_dict[str(score_threshold)] = {}
        benign_result_dict[str(score_threshold)] = copy.deepcopy(result_dict)

    pbar = tqdm(range(int(args.query_budget / args.query_size)))
    for idx in range(1, 11):
        cache_query_path = osp.join(cache_base_path, f"{tag_name}_{idx}.pt")
        cache = torch.load(cache_query_path)
        num_query = int(len(cache["inputs"]) / args.query_size)
        for query_idx in range(num_query):
            off = int(query_idx * args.query_size)
            batch_query = cache["inputs"][off: off + args.query_size].cpu()
            alarm, seat_pred, dist = SEAT.query(batch_query)
            result_dict["pred"].append(seat_pred)
            result_dict["y"].append(np.ones(len(seat_pred)))
            conf = round(100.0 * (1.0 - np.sum(seat_pred) / seat_pred.shape[0]), 2)
            result_dict["conf"].append(conf)
            query_pred = 1 if alarm else 0
            result_dict["query_y"].append(1)
            result_dict["query_pred"].append(query_pred)
            pbar.set_description(
                f"-> [{method}(τ1={score_threshold}, delta={args.delta})] fidx:{idx} query_fidx:{query_idx} SEAT({SEAT.delta}): conf={conf}%")
            pbar.update(1)
    print()

    result_dict["y"] = np.concatenate(result_dict["y"]).astype(np.int32)
    result_dict["pred"] = np.concatenate(result_dict["pred"]).astype(np.int32)
    result_dict["query_y"] = np.array(result_dict["query_y"]).astype(np.int32)
    result_dict["query_pred"] = np.array(result_dict["query_pred"]).astype(np.int32)
    result_dict["conf"] = np.array(result_dict["conf"]).astype(np.float64)

    y = result_dict["y"]
    p = result_dict["pred"]
    sample_res = metric.multi_mertic(y, p)
    print(
        f"""-> TEST_SAMPLE(τ1={score_threshold}) ACC:{sample_res['ACC']}% 
            Recall:{sample_res["Recall"]} Precision:{sample_res["Precision"]} F1-score:{sample_res["F1score"]}
            FPR:{sample_res['FPR100']} TPR:{sample_res['TPR100']}""")

    y = result_dict["query_y"]
    p = result_dict["query_pred"]
    query_res = metric.multi_mertic(y, p)
    result_dict.update(query_res)
    print(
        f"""-> TEST_QUERY(τ1={score_threshold}) ACC:{query_res['ACC']}% 
        Recall:{query_res["Recall"]} Precision:{query_res["Precision"]} F1-score:{query_res["F1score"]}
        FPR:{query_res['FPR100']} TPR:{query_res['TPR100']}""")
    result_path = osp.join(args.out_root, "exp",
                           "exp1.1_SEAT-q{:04d}_{:s}.pt".format(args.query_size, tag_name))
    torch.save(result_dict, result_path)
    global_result_dict[str(score_threshold)][method] = copy.deepcopy(result_dict)
    print(f"-> save pred:{result_path}")


def main(args):
    print("""\n-> step1: load pretrained encoder""")
    if args.vic_data == "CIFAR10":
        encoder = vgg16_bn(pretrained=False)
        args.arch = "vgg16_bn"

    elif args.vic_data == "MNIST":
        #from models.lenet import LeNet
        #encoder = LeNet(num_classes=10)
        args.arch = "vgg11_bn"
        encoder = vgg11_bn(pretrained=False)
        encoder.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    else:
        raise NotImplementedError(f"-> {args.vic_data} Not implemented!!")

    weights = helper.load(arch=f"SEAT_{args.arch}", dataset=args.vic_data)
    if weights is None:
        raise FileNotFoundError(f"-> pretrain encoder: {args.arch} not found!!")
    encoder.load_state_dict(weights)

    print("""\n-> step2: load dataset""")
    print("-> load ground-truth dataset\n")
    loader = DataLoader(data_root=args.data_root, dataset=args.vic_data, batch_size=1000)
    ben_train_loader, ben_test_loader = loader.get_loader()
    ben_x, ben_y = exp_utils.build_ben_dataset(data_loader=ben_test_loader, query_budget=args.query_budget, query_size=args.query_size)
    ben_set = {
        "x": ben_x,
        "y": ben_y
    }

    tag_name = f"{args.vic_model}_{args.vic_data}_{args.atk_model}"
    global global_result_dict
    for score_threshold in args.threshold_list:
        for method in args.atk_methods:
            run(args, copy.deepcopy(encoder), method, ben_set, loader.bounds, score_threshold=score_threshold)

    global_result_path = osp.join(args.out_root, "exp_1.1",
                           "exp1.1_SEAT-{:s}_q{:04d}_{:s}.pt".format(args.vic_data, args.query_size, tag_name))
    torch.save(global_result_dict, global_result_path)
    print()



if __name__ == "__main__":
    args = helper.get_args()
    args.arch = "vgg16_bn"
    args.delta = 1e-5

    if args.vic_data == "CIFAR10":
        if args.query_size == 100:
            args.delta = 1e-4
            args.threshold_list = [0.9, 0.88]
        elif args.query_size == 1000:
            args.delta = 1e-5
            args.threshold_list = [0.86, 0.87]
    elif args.vic_data == "MNIST":

        args.delta = 1e-8
        if args.query_size == 100:
            args.threshold_list = [0.94, 0.90]
        elif args.query_size == 1000:
            args.threshold_list = [0.94, 0.90]
    main(args)