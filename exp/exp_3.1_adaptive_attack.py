#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/05/13, homeway'

import os
import os.path as osp
import sys
import inspect
import torch
import torch.nn.functional as F
from tqdm import tqdm
import copy

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from step1_pretrain import pretrain
from utils import helper
from MLaaS.defense import profedi_ops, profedi
from datasets.loader import FediLoader as DataLoader
from models.loader import FediModel
args = helper.get_args()



def retransform(x):
    if x.size(1) == 3:
        mean = (0.43768206, 0.44376972, 0.47280434)
        std = (0.19803014, 0.20101564, 0.19703615)
    else:
        mean = [0.1307, ]
        std = [0.3081, ]
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    return x.detach()


def transform(x):
    if x.size(1) == 3:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
    else:
        mean = [0.5, ]
        std = [0.5, ]
    for t, m, s in zip(x, mean, std):
        t.sub_(m).div_(s)
    return x.detach()


def transform_padding(x):
    if (x.size(1) == 1) and args.sub_model != "victim":
        t = torch.zeros(x.size(0), 3, 32, 32)
        t[:, :] = copy.deepcopy(x)
        x = copy.deepcopy(t)
    return x


def feature_correction(query_x, query_y, sub_model, anchor_feats, bounds=[-1, 1], steps=1):
    device = next(sub_model.parameters()).device
    origin_x = copy.deepcopy(query_x)
    query_x = query_x.clone()
    query_y = query_y.clone()
    sub_model.eval()
    for l in range(10):
        idx = (query_y == l).nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue
        label_feats = anchor_feats[l].to(device)
        adv_images = copy.deepcopy(query_x[idx]).to(device)
        adv_images.requires_grad = True
        optimizer = torch.optim.LBFGS([adv_images])
        for step in range(steps):
            def closure():
                optimizer.zero_grad()
                pred_feats = sub_model.avgpool(adv_images).view(adv_images.size(0), -1).to(device)
                cost = ((pred_feats - label_feats) ** 2).mean()
                cost.backward()
                return cost
            optimizer.step(closure)
            adv_images = torch.clamp(adv_images, min=bounds[0], max=bounds[1]).detach().to(device)
            adv_images.requires_grad = True
        pred_feats = sub_model.avgpool(adv_images).view(adv_images.size(0), -1).to(device)
        loss = ((pred_feats - label_feats) ** 2).mean().detach().cpu()
        query_x[idx] = adv_images.detach().cpu()
    diff = ((query_x - origin_x)**2).mean().detach().cpu()
    return query_x, diff, loss


def build_feature_correction_queryset(args, methods, anchor_feats, vic_model, sub_model, sub_sample, bounds):
    sub_anchor_feats = []
    with torch.no_grad():
        pred_y = vic_model(sub_sample).argmax(dim=1).detach().cpu()
        # padding 2-dimension for MNIST for vgg11
        sub_sample = transform_padding(sub_sample)
        class_feats = sub_model.avgpool(sub_sample).view(sub_sample.size(0), -1).detach().cpu()
        for l in range(10):
            idx = (pred_y == l).nonzero(as_tuple=True)[0]
            sub_anchor_feats.append(torch.mean(class_feats[idx], dim=0))

    # phar = tqdm(enumerate(query_loader))
    '''step2 train binary classifier'''
    weights = FediModel(ckpt=args.ckpt_root).load(arch="FeDINet", task=args.vic_data)
    if weights is None:
        raise RuntimeError("-> Please run step3 to train FeDInet")
    from models.fedinet import FeDINet
    fedinet = FeDINet()
    fedinet.load_state_dict(weights)
    fedinet.eval()

    storage_root = osp.join(args.out_root, f"storage_{args.vic_data}")
    for method in methods:
        for i in range(args.query_range[0], args.query_range[1]):
            file_path = osp.join(storage_root, f"{args.tag}_{method}_{i}.pt")
            save_path = osp.join(storage_root, f"{args.tag}_{method}-{args.adaptive_tag}_{i}.pt")
            if osp.exists(save_path):
                print(f"-> have done! bypass! {save_path}")
                continue
            if not osp.exists(file_path):
                raise FileNotFoundError(f"-> cache file:{file_path} not found!!!")

            print(f"-> for method: {method}_{i}, read from:{file_path}")
            query_set = torch.load(file_path)
            query_loader = profedi_ops.tensor2loader(x=query_set["inputs"],
                                                     y=query_set["outputs"],
                                                     batch_size=500,
                                                     shuffle=False)
            phar = tqdm(enumerate(query_loader))
            for step, (x, y) in phar:
                x = transform_padding(x)
                off = int(step * x.size(0))
                adv_x, diff, loss = feature_correction(x, y, sub_model=sub_model, anchor_feats=sub_anchor_feats, bounds=bounds, steps=2)
                if args.vic_data == "MNIST":
                    adv_x = copy.deepcopy(adv_x[:, :1, :, :])
                query_set["inputs"][off: off + x.size(0)] = adv_x.detach().cpu()
                phar.set_description(f"-> method:{method}_{i} [{step}/{len(query_loader)}] FeatC diff:{diff} loss:{loss}")
                '''
                query_x = torch.cat([x, adv_x.detach().cpu()])
                query_y = torch.ones(len(query_x))
                adv_test_query_loader = profedi_ops.tensor2loader(x=query_x, y=query_y, shuffle=False, batch_size=query_x.size(0))
                adv_test_fdi, adv_test_labels = profedi.get_fdi_feats(vic_model, adv_test_query_loader,
                                                                      anchor_feats=anchor_feats)
                print(adv_test_labels)
                size = int(len(query_x) / 2)
                pred = fedinet(adv_test_fdi).argmax(dim=1).cpu()

                pred_a = copy.deepcopy(pred[:size])
                pred_b = copy.deepcopy(pred[size:])
                label_a = copy.deepcopy(query_y[:size])
                label_b = copy.deepcopy(query_y[size:])

                a_correct = pred_a.eq(label_a).sum().item()
                b_correct = pred_b.eq(label_b).sum().item()

                a_acc = 100.0 * (a_correct / size)
                b_acc = 100.0 * (b_correct / size)

                #print(pred_a)
                #print(pred_b)
                print(f"-> {step} {method} a:{a_acc}%, b:{b_acc}%")
                print()

                if step > 10:
                    break
                    exit(1)
                '''
            print(f"-> for method: {method}_{i}, save to:{save_path}")
            torch.save(query_set, save_path)


def build_fedi_feats(args, vic_model):
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

    methods = []
    for method in args.atk_methods:
        methods.append(f"{method}-{args.adaptive_tag}")

    from exp.step2_build_fedi import build_queryset_dataloader
    build_queryset_dataloader(
        args=args, vic_model=vic_model, anchor_feats=anchor_feats, ben_test_fds=ben_test_fds, methods=methods,
        query_budget=args.query_budget, query_size=1000
    )


def main(args):
    # 1.load victim model
    print("-> step1: load model")
    vic_model = pretrain(args)
    vic_model = vic_model.to(args.device)

    # 2.load train_loader
    print("-> step2: load dataset")
    ben_loader = DataLoader(data_root=args.data_root, dataset=args.vic_data, batch_size=args.batch_size)
    ben_train_loader, ben_test_loader = ben_loader.get_loader()

    # 3.load substitute model download from model zoo
    print("""-> step3: load sub model""")
    if args.sub_model == "vgg11":
        sub_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True).to(args.device)
    elif args.sub_model == "victim":
        sub_model = pretrain(args)
    else:
        raise NotImplementedError()
    sub_model.eval()

    # 4. make assumption the adversary has a mini-batch of training data
    print("-> step4: build substitute model & anchor features")
    sub_sample, _ = iter(ben_train_loader).next()

    # 5.build feature correction queryset
    anchor_feats = profedi_ops.load_anchor_feats(dataset=args.vic_data)
    build_feature_correction_queryset(args, methods=args.atk_methods,
                                      anchor_feats=anchor_feats,
                                      vic_model=vic_model,
                                      sub_sample=sub_sample.to(args.device),
                                      sub_model=sub_model.to(args.device),
                                      bounds=ben_loader.bounds)
    # 6. build fedi vector
    args.query_size = 500
    args.batch_size = 500
    args.query_budget = 50000
    build_fedi_feats(args, vic_model)


if __name__ == "__main__":
    args.adaptive_tag = "FeatC"
    args.sub_model = "victim"
    args.query_range = [1, 2]
    main(args)








