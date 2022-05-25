#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/04/12, homeway'


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import os.path as osp
from utils import helper, metric
from step1_pretrain import pretrain
from models.loader import FediModel
from MLaaS.defense import profedi, profedi_ops


def build_fdi_feats():
    # build fdi_feats for adversarial clients
    pass


def main(args):
    '''pretrain MLaaS DNN'''
    vic_model = pretrain(args)

    '''step1 load anchor feats'''
    profedi_ops.load_anchor_feats(dataset=args.vic_data)

    '''step2 train binary classifier'''
    weights = FediModel(ckpt=args.ckpt_root).load(arch="FeDINet", task=args.vic_data)
    if weights is None:
        raise RuntimeError("-> Please run step3 to train FeDInet")
    from models.fedinet import FeDINet
    fedinet = FeDINet()
    fedinet.load_state_dict(weights)
    fedinet.eval()

    '''step3 load ben/adv feats'''
    methods = ["JBDA-FGSM", "JBDA-DF", "Knockoff", "DFME", "DaST"]
    tag = f"fdi_test_query_{args.vic_model}_{args.vic_data}_{args.atk_model}"

    for method in methods:
        fdi_feats = profedi_ops.load_query_testset(methods=[method], tag=tag)[method]
        print(f"-> Test query from: {method} query_size:{args.query_size} query_budget:{args.query_budget}")
        half_size = int(len(fdi_feats["x"]) / 2)
        ben_query_loader = profedi_ops.tensor2loader(x=fdi_feats["x"][:half_size].clone(),
                                                y=fdi_feats["y"][:half_size].clone(),
                                                shuffle=False, batch_size=args.query_size)
        adv_query_loader = profedi_ops.tensor2loader(x=fdi_feats["x"][half_size:].clone(),
                                                y=fdi_feats["y"][half_size:].clone(),
                                                shuffle=False, batch_size=args.query_size)
        file_path = osp.join(args.out_root, "exp",
                             "exp1.1_binary_ProFeDI-qs{:04d}_{:.3f}_{:s}.pt".format(
                                 args.query_size,
                                 args.tau1,
                                 tag
                             )
                    )
        metric.test(fedinet,
                         ben_test_loader=ben_query_loader,
                         adv_test_loader=adv_query_loader,
                         tau1=args.tau1,
                         file_path=file_path
                    )
        print()



if __name__ == "__main__":
    main(helper.get_args())










