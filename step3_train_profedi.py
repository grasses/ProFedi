#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/04/12, homeway'


import os.path as osp
import torch
from utils import helper, metric
from step1_pretrain import pretrain
from models.loader import FediModel
from datasets.loader import FediLoader as DataLoader
from MLaaS.defense import profedi, profedi_ops


def main(args):
    '''pretrain MLaaS DNN'''
    vic_model = pretrain(args)

    '''load dataset'''
    ben_loader = DataLoader(data_root=args.data_root, dataset=args.vic_data, batch_size=args.batch_size)
    ben_train_loader, ben_test_loader = ben_loader.get_loader()
    adv_loader = DataLoader(data_root=args.data_root, dataset=args.sub_data, batch_size=args.batch_size)
    adv_train_loader, adv_test_loader = adv_loader.get_loader()

    '''step1 load anchor feats'''
    anchor_feats = profedi.get_anchor_feats(vic_model, ben_test_loader, num_classes=10, anchor_size=100)
    profedi_ops.save_anchor_feats(dataset=args.vic_data, anchor_feats=anchor_feats)

    '''step2 load ben/adv feats'''
    ben_dataset = args.vic_data
    if ben_dataset == "CIFAR10":
        adv_dataset = "CIFAR100"
    elif ben_dataset == "SVHN":
        adv_dataset = "MNIST"
    elif ben_dataset == "MNIST":
        adv_dataset = "FashionMNIST"
    else:
        raise NotImplementedError(f"-> error benign_dataset: {ben_dataset}")

    print("\n<======================training dataset==========================>")
    path = osp.join(args.out_path, f"fedi/fdi_test_{args.vic_data}.pt")
    if not osp.exists(path):
        ben_train_fdi, ben_train_labels = profedi.get_fdi_feats(vic_model, ben_train_loader, anchor_feats=anchor_feats)
        profedi_ops.save_fdi_feats(name=f"train_{ben_dataset}",
                                   fdi_feats=ben_train_fdi,
                                   pred_labels=ben_train_labels,
                                   fdi_labels=torch.zeros(len(ben_train_labels)))
        adv_train_fdi, adv_train_labels = profedi.get_fdi_feats(vic_model, adv_train_loader, anchor_feats=anchor_feats)
        profedi_ops.save_fdi_feats(name=f"train_{adv_dataset}",
                                   fdi_feats=adv_train_fdi,
                                   pred_labels=adv_train_labels,
                                   fdi_labels=torch.ones(len(adv_train_labels)))
        print("\n<======================testing dataset==========================>")
        ben_test_fdi, ben_test_labels = profedi.get_fdi_feats(vic_model, ben_test_loader, anchor_feats=anchor_feats)
        profedi_ops.save_fdi_feats(name=f"test_{ben_dataset}",
                                   fdi_feats=ben_test_fdi,
                                   pred_labels=ben_test_labels,
                                   fdi_labels=torch.zeros(len(ben_test_labels)))
        adv_test_fdi, adv_test_labels = profedi.get_fdi_feats(vic_model, adv_test_loader, anchor_feats=anchor_feats)
        profedi_ops.save_fdi_feats(name=f"test_{adv_dataset}",
                                   fdi_feats=adv_test_fdi,
                                   pred_labels=adv_test_labels,
                                   fdi_labels=torch.ones(adv_test_labels))
    else:
        # load from cache
        ben_train_fdi = profedi_ops.load_fdi_feats(name=f"train_{ben_dataset}")["x"]
        ben_test_fdi = profedi_ops.load_fdi_feats(name=f"test_{ben_dataset}")["x"]
        adv_train_fdi = profedi_ops.load_fdi_feats(name=f"train_{adv_dataset}")["x"]
        adv_test_fdi = profedi_ops.load_fdi_feats(name=f"test_{adv_dataset}")["x"]


    '''step3 train binary classifier'''
    from models.fedinet import FeDINet
    fedinet = FeDINet()
    fedinet.train()
    fedinet = fedinet.to(args.device)
    optimizer = torch.optim.Adam(fedinet.parameters(), lr=1e-4)
    fds_train_loader = profedi_ops.tensor2loader(
        x=torch.cat([ben_train_fdi[:50000], adv_train_fdi[:50000]]),
        y=torch.cat([torch.zeros(50000), torch.ones(50000)]).long(),
        shuffle=True,
        batch_size=args.batch_size
    )
    # load from cache
    model_loader = FediModel(ckpt=args.ckpt_root)
    weights = model_loader.load(arch="FeDINet", task=args.vic_data)
    if weights is None:
        for epoch in range(20):
            fedinet = metric.train(fedinet, train_loader=fds_train_loader, optimizer=optimizer)
        helper.save(fedinet, arch="FeDINet", dataset=args.vic_data)
        print()
    else:
        fedinet.load_state_dict(weights)


    '''step4 eval binary classifier'''
    ben_test_loader = profedi_ops.tensor2loader(
        x=ben_test_fdi[:10000],
        y=torch.zeros(10000).long(),
        shuffle=False,
        batch_size=args.batch_size
    )
    adv_test_loader = profedi_ops.tensor2loader(
        x=adv_test_fdi[:10000],
        y=torch.ones(10000).long(),
        shuffle=False,
        batch_size=args.batch_size
    )
    metric.test(fedinet, ben_test_loader, adv_test_loader, epoch=0, tau1=args.tau1, file_path=None)


if __name__ == "__main__":
    main(helper.get_args())



















