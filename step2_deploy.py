#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/03/31, homeway'


from utils import helper
from step1_pretrain import pretrain


def deploy(args):
    '''pretrain MLaaS DNN'''
    vic_model = pretrain(args)
    from web import server
    from MLaaS import Storage, BlackBox

    '''storage'''
    storage_root = f"output/storage"
    prefix = f"{args.vic_model}_{args.vic_data}"
    storage = Storage(storage_root=storage_root, prefix=prefix)
    '''load defense method'''
    defense_process = None
    '''load MLaaS blackbox service'''
    blackbox = BlackBox(vic_model, save_process=storage.add_query, defense_process=defense_process, device=args.device)
    '''upload DNN model to MLaaS'''
    app = server(blackbox=blackbox)
    app.run(debug=True, threaded=True, host=args.host, port=args.port)


if __name__ == "__main__":
    deploy(helper.get_args())