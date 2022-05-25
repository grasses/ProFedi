#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/01/25, homeway'


from utils import helper
from step1_pretrain import pretrain


def main(args):
    '''pretrain MLaaS DNN'''
    vic_model = pretrain(args)

    if args.type == "client":
        assert args.attack in ["JBA_DF", "JBA_PGD", "RS", "Knockoff", "ActiveTheft", "DFME", "MAZE"]
        if args.attack == "JBA_DF":
            pass
        elif args.attack == "JBA_PGD":
            pass
        elif args.attack == "RS":
            pass
        elif args.attack == "Knockoff":
            pass
        elif args.attack == "ActiveTheft":
            pass
        elif args.attack == "DFME":
            pass
        elif args.attack == "MAZE":
            pass
        else:
            raise NotImplementedError(f"-> attack method:{args.attack} not implemented!!")

    elif args.type == "demo":
        from web import server
        from MLaaS import Storage, BlackBox
        '''storage'''
        storage_root = f"output/storage_{args.vic_data.upper()}"
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
    main(helper.get_args())