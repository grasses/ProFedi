#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/07/21, homeway'


import os
import sys
import inspect
import torch
import random
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


def adv_obj(method, model, bounds):
    if "FGSM" in method:
        atk = jbda_ops.FGSM(model, eps=8./255, bounds=bounds)
        atk.set_mode_targeted_random()
    elif "PGD" in method:
        atk = jbda_ops.PGD(model, eps=24./255, alpha=10./255, steps=5, bounds=bounds)
        atk.set_mode_targeted_random()
    elif "DF" in method:
        steps = random.randint(20, 70)
        atk = jbda_ops.DeepFool(model, steps=steps, overshoot=0.2, bounds=bounds)
    elif "CW" in method:
        steps = random.randint(800, 1200)
        atk = jbda_ops.CW(model, c=1e-4, kappa=0, steps=steps, lr=0.01, bounds=bounds)
        atk.set_mode_targeted_random()
    else:
        raise NotImplementedError(f"-> method: {method} not implemented")
    return atk.forward


def attack(args):
    args.pred_type = "hard"
    args.batch_size = 200
    args.attack_rounds = 600
    args.query_size = 1000
    args.env_name = f"{args.env_name}_{args.atk_method}_{args.vic_data}_{args.atk_data}"
    tag = f"{args.atk_model}_{args.atk_method}"

    '''victim model'''
    vic_model = pretrain(args)

    '''victim dataset'''
    loader = DataLoader(data_root=args.data_root, dataset=args.vic_data, batch_size=200)
    train_loader, test_loader = loader.get_loader()

    '''storage'''
    from MLaaS.server.storage import Storage
    storage_root = f"output/storage"
    prefix = f"{args.vic_model}_{args.vic_data}"
    storage = Storage(storage_root=storage_root, prefix=prefix)
    blackbox = BlackBox(model=vic_model, save_process=storage.add_query, device=args.device)

    '''attack model'''
    model_loader = FediModel(ckpt=args.ckpt_root)
    atk_model = model_loader.get_model(arch=args.atk_model, task=args.vic_data, num_classes=10, pretrain=False)
    if args.vic_data.upper() in ["CIFAR10", "CIFAR100"]:
        optimizer = torch.optim.SGD(atk_model.parameters(), lr=0.05, momentum=args.momentum,
                                     weight_decay=args.weight_decay)
    elif args.vic_data.upper() in ["SVHN", "FASHIONMNIST", "MNIST"]:
        optimizer = torch.optim.SGD(atk_model.parameters(), lr=0.01, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f"-> task not implemented: {args.vic_data}")

    '''attack object'''
    atk = jbda.JBDA(model=atk_model, optimizer=optimizer,
                    batch_size=args.batch_size,
                    query_size=args.query_size,
                    attack_rounds=args.attack_rounds,
                    pred_type=args.pred_type,
                    device=args.device)
    sub_x, sub_y = iter(train_loader).next()
    if args.pred_type == "soft":
        sub_y = torch.zeros([args.batch_size, args.num_classes])

    '''training student model'''
    best_acc, best_fid_acc = 0.0, 0.0
    sampel_cnt = 0
    batch_size = args.batch_size
    phar = tqdm(range(1, 2 + args.attack_rounds))
    student = atk_model.to(args.device)
    adv = adv_obj(args.atk_method, model=student, bounds=loader.bounds)
    for round in phar:
        if len(sub_y) > args.query_size:
            batch_size = args.query_size
        sub_x = sub_x[-int(5*args.query_size):]
        sub_y = sub_y[-int(5*args.query_size):]
        student, sub_x, sub_y = atk.extract(blackbox, sub_x, sub_y, adv=adv, epoch=round, per_epoch=2, tag=tag)
        sampel_cnt += batch_size

        if round % 2 == 0:
            _epoch = int(round / 2)
            test_acc, test_loss = helper.test(student, test_loader, args.device, epoch=_epoch, debug=True)
            fid_acc, fid_loss = helper.test_fidelity(student, vic_model, test_loader, args.device, epoch=_epoch, debug=True)
            if test_acc > best_acc:
                best_acc = test_acc
            if fid_acc > best_fid_acc:
                best_fid_acc = fid_acc
            # update result
            phar.set_description(f"-> For E{_epoch} [Test] best_acc:{best_acc:.3f}% best_fid:{best_fid_acc:.3f}% samples cnt:{sampel_cnt}")
            helper.save(student, arch=f"Extract_{args.vic_data}_{args.atk_model}", dataset=args.atk_method)
            print()
        adv = adv_obj(args.atk_method, model=student.to(args.device), bounds=loader.bounds)


if __name__ == "__main__":
    args = helper.get_args()
    helper.set_seed(args)
    attack(args)






















