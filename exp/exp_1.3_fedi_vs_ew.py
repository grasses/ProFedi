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
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from step1_pretrain import pretrain
from exp import exp_utils
from utils import helper, metric
from MLaaS.defense import profedi_ops, profedi, ExtractionWarning
from models.loader import FediModel
from datasets.loader import FediLoader as DataLoader


def simulate_profedi_defense(args, vic_model, anchor_feats, ben_test_set, methods, extract_state={"ProFedi": []}):
    if "steps" not in extract_state.keys():
        extract_state["steps"] = args.steps

    # load FeDINet from pretrained model
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

    # load cache query file's fedi
    from exp.step2_build_fedi import build_queryset_dataloader
    fds_testset_methods = build_queryset_dataloader(
        args=args, vic_model=vic_model, anchor_feats=anchor_feats, ben_test_fds=ben_test_set, methods=methods,
        query_budget=args.query_budget, query_size=args.query_size
    )

    conf_array = {}
    for step, (method, fds_feat) in enumerate(fds_testset_methods.items()):
        start_off = int(len(fds_feat["x"]) / 2)
        fds_query_loader = profedi_ops.tensor2loader(x=fds_feat["x"][start_off:], y=fds_feat["y"][start_off:],
                                                     shuffle=False, batch_size=args.query_size)
        conf_array[method] = exp_utils.test_query(fedinet, test_loader=fds_query_loader)
        print(
            f"-> [{step+1}/{len(fds_testset_methods)}] {method} query_size:{args.query_size} score:{conf_array[method]}")

    # test benign
    fds_query_loader = profedi_ops.tensor2loader(x=fds_feat["x"][:start_off], y=fds_feat["y"][:start_off],
                                                 shuffle=False, batch_size=args.query_size)
    conf_array["Benign"] = exp_utils.test_query(fedinet, test_loader=fds_query_loader)
    methods.append("Benign")
    methods.reverse()

    steps = extract_state["steps"]
    for method in methods:
        extract_state["ProFedi"][method] = copy.deepcopy(conf_array[method][steps])
    return extract_state["ProFedi"]


def simulate_ew_defense(args, vic_model, proxy_model, train_loader, test_loader, methods, extract_state={"EW":{}}):
    if "steps" not in extract_state.keys():
        extract_state["steps"] = args.steps

    warning = ExtractionWarning(victim_model=proxy_model, proxy_model=proxy_model, device=args.device)
    extract_state["EW"] = {}
    test_acc, test_loss, fid_acc, fid_loss = 0., 0., 0., 0.
    for method in methods:
        query_count = 0
        extract_state["EW"][method] = []
        tag_name = f"{args.tag}_{method}"
        query_cache_root = osp.join(args.out_root, f"storage_{args.vic_data}")
        print(f"-> Evaluate method:{method}")
        if "benign" in method.lower():
            ben_query_set = exp_utils.build_query_set(train_loader,
                                                query_budget=args.query_budget,
                                                query_size=args.query_size)
            ben_query_loader = profedi_ops.tensor2loader(
                                                x=ben_query_set["x"],
                                                y=ben_query_set["y"],
                                                batch_size=args.query_size)
            phar = tqdm(enumerate(ben_query_loader))
            query_num = len(ben_query_loader)
            for step, (batch_x, batch_y) in phar:
                warning.batch_query(x=batch_x, y=batch_y, user=method)
                if query_count in extract_state["steps"]:
                    proxy_model = warning.get_agent(method)
                    fid_acc, fid_loss = helper.test_fidelity(
                        proxy_model, proxy_model, test_loader, args.device, epoch=query_count, debug=True
                    )
                    extract_state["EW"][method].append(fid_acc)
                    helper.save(proxy_model,
                                arch=f"EW_q{args.query_size}_c{query_count}_m{method}",
                                dataset=args.vic_data)
                    del proxy_model
                phar.set_description(f"-> [{step}/{query_num}] proxy model acc: {test_acc}% loss: {test_loss} fid_acc: {fid_acc}% fid_loss: {fid_loss}")
                query_count += 1
        else:
            for idx in range(1, 11):
                query_cache_path = osp.join(query_cache_root, f"{tag_name}_{idx}.pt")
                print(f"-> load cache query: {query_cache_path}")
                cache = torch.load(query_cache_path)
                x = cache["inputs"]
                y = cache["outputs"]
                query_num = int(x.size(0) / args.query_size)
                phar = tqdm(range(query_num))
                for step in phar:
                    off = step * args.query_size
                    batch_x = x[off: off + args.query_size].clone()
                    batch_y = y[off: off + args.query_size].clone()
                    warning.batch_query(x=batch_x, y=batch_y, user=method)
                    if query_count in extract_state["steps"]:
                        proxy_model = warning.get_agent(method)
                        fid_acc, fid_loss = helper.test_fidelity(
                            proxy_model, vic_model, test_loader, args.device, epoch=query_count, debug=False
                        )
                        test_acc, test_loss = helper.test(proxy_model, test_loader, args.device, epoch=query_count, debug=False)
                        extract_state["EW"][method].append(fid_acc)
                        helper.save(proxy_model,
                                    arch=f"EW_q{args.query_size}_c{query_count}_{method}",
                                    dataset=args.vic_data)
                        del proxy_model
                    phar.set_description(f"-> [{step}/{query_num}] proxy model acc: {test_acc}% loss: {test_loss} fid_acc: {fid_acc}% fid_loss: {fid_loss}")
                    query_count += 1
    return extract_state["EW"]


def plot_conf_curve(res_array, atk_methods, def_methods, base_path, linewidth=5, fontsize=35, markersize=25, plt_num=50):
    plt.cla()
    plt.figure(figsize=(12, 12), dpi=100)
    markers = ["o", "*", "x", "v", "s", "d"]

    for atk_method in atk_methods:
        plt.cla()
        plt.grid()
        for idx, def_method in enumerate(def_methods):
            x = copy.deepcopy(res_array["steps"][0:102:2] + 1).astype(np.int32)
            y = copy.deepcopy(res_array[def_method][atk_method][0:102:2])
            plt.plot(x, y, label=def_method, linewidth=linewidth, markersize=markersize, marker=markers[idx])

        #plt.title(f"attack_{atk_method}", fontsize=fontsize)
        #plt.legend(loc='center right', fontsize=fontsize)
        plt.xlabel('Number of Queries', fontsize=fontsize)
        plt.ylabel('Extraction Status', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(np.arange(0, 101, 20), fontsize=fontsize)

        save_path = f"{base_path}_atk{atk_method}.pdf"
        print(f"-> saving hist at:{save_path}")
        plt.savefig(save_path)


def main(args):
    # saving result to path
    base_path = osp.join(args.out_root, f"exp_1.3/exp_1.3_curve_{args.vic_data}")
    exp_path = f"{base_path}.pt"
    extract_state = {}
    if osp.exists(exp_path):
        extract_state = torch.load(exp_path)
        print(f"-> Load result cache from {exp_path}")
    else:
        # 1.load victim model
        print("-> step1: load model")
        vic_model = pretrain(args)
        vic_model = vic_model.to(args.device)

        # 2.load feature distortion
        print("\n-> step2: load anchor feature")
        anchor_feats = profedi_ops.load_anchor_feats(dataset=args.vic_data)

        # 3.load train_loader
        print("-> step3 load dataset")
        ben_loader = DataLoader(data_root=args.data_root, dataset=args.vic_data, batch_size=args.batch_size)
        ben_train_loader, ben_test_loader = ben_loader.get_loader()

        path = osp.join(args.out_path, f"fedi/fdi_test_{args.vic_data}.pt")
        if not osp.exists(path):
            print("\n<======================testing dataset==========================>")
            ben_test_fdi, ben_test_labels = profedi.get_fdi_feats(vic_model, ben_test_loader, anchor_feats=anchor_feats)
            profedi_ops.save_fdi_feats(name=f"test_{args.vic_data}",
                                       fdi_feats=ben_test_fdi,
                                       pred_labels=ben_test_labels,
                                       fdi_labels=torch.zeros(len(ben_test_labels)))
            ben_test_set = {
                "x": ben_test_fdi,
                "y": torch.zeros(len(ben_test_fdi)),
                "z": ben_test_labels,
            }
        else:
            ben_test_set = profedi_ops.load_fdi_feats(name=f"test_{args.vic_data}")

        # check key of extract_state file
        if "EW" not in extract_state.keys():
            extract_state["EW"] = {}
        if "ProFedi" not in extract_state.keys():
            extract_state["ProFedi"] = {}
        if "steps" not in extract_state.keys():
            extract_state["steps"] = args.steps

        """<=================================ProFedi=================================>"""
        if "ProFedi" in args.def_methods:
            extract_ProFedi = simulate_profedi_defense(args, vic_model, anchor_feats, ben_test_set, args.atk_methods,
                                                       extract_state=copy.deepcopy(extract_state))
            extract_state["ProFedi"] = copy.deepcopy(extract_ProFedi)

        """<=================================Extraction Warning=================================>"""
        if "EW" in args.def_methods:
            model_loader = FediModel(ckpt=args.ckpt_root)
            proxy_model = model_loader.get_model(arch=args.vic_model, task=args.vic_data, num_classes=10, pretrain=False)

            # simulate Extraction Monitor paper
            extract_EW = simulate_ew_defense(args, vic_model, proxy_model,
                                                ben_train_loader, ben_test_loader, args.atk_methods,
                                                extract_state=copy.deepcopy(extract_state))
            extract_state["EW"] = copy.deepcopy(extract_EW)

    atk_methods = (["Benign"] + args.atk_methods)
    if len(args.def_methods) > 1:
        torch.save(extract_state, exp_path)
        plot_conf_curve(extract_state,
                        atk_methods=atk_methods,
                        def_methods=args.def_methods,
                        base_path=base_path, fontsize=40, markersize=25)



if __name__ == "__main__":
    args = helper.get_args()
    args.query_size = 100
    args.steps = np.arange(0, 5000, 50)
    args.def_methods = ["ProFedi", "EW"]
    args.atk_methods = [
        "JBDA-FGSM",
        #"JBDA-DF",
        "JBDA-PGD",
        "Knockoff",
        "DFME",
        "DaST"
    ]
    main(args=args)






















