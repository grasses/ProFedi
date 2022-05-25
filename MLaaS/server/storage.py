#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/05/20, homeway'


import os
import os.path as osp
import torch
import os.path as osp


class Storage:
    def __init__(self, storage_root, prefix, cache_size=50000):
        self.tags = set()
        self.caches = {}
        self.last_process = []
        self.cache_size = cache_size
        self.prefix = prefix
        self.storage_root = storage_root
        if not osp.exists(storage_root):
            os.makedirs(storage_root)

    def __save_data__(self, storage_root, prefix, cache):
        if len(cache["inputs"]) == 0:
            print("-> save error", cache["tag"])
            return

        save_name = f"{prefix}_{cache['tag']}_{cache['save_idx']}.pt"
        print(f"-> Saving data at:{storage_root}/{prefix}/{save_name}")
        cache["inputs"] = torch.cat(cache["inputs"])[:self.cache_size]
        cache["outputs"] = torch.cat(cache["outputs"])[:self.cache_size]
        cache["features"] = torch.cat(cache["features"])[:self.cache_size]
        torch.save(cache, osp.join(storage_root, f"{save_name}"))

    def __init_client__(self, tag):
        if tag not in self.tags:
            cache = {
                "tag": tag,
                "step": 0,
                "size": 0,
                "clock": 1,
                "save_idx": 1,
                "inputs": [],
                "outputs": [],
                "features": []
            }
            self.tags.add(tag)
            self.caches[tag] = cache

    def __push_data__(self, inputs, outputs, features, **kwargs):
        tag = kwargs["tag"]
        # 当数据大于50000时候，存储缓冲区
        if self.caches[tag]["size"] >= self.cache_size:
            self.__flush_cache__(tag)
            self.__init_client__(tag)

        # 缓冲区中添加数据
        self.caches[tag]["step"] += 1
        self.caches[tag]["clock"] = 0
        self.caches[tag]["size"] += int(inputs.size(0))
        self.caches[tag]["inputs"].append(inputs.cpu())
        self.caches[tag]["outputs"].append(outputs.cpu())
        self.caches[tag]["features"].append(features.cpu())

        # 针对没有请求满50000的用户，定时清空缓区
        for _tag in self.tags:
            size = self.caches[_tag]["size"]
            clock = self.caches[_tag]["clock"]
            if (_tag != tag) and (size > 0):
                self.caches[_tag]["clock"] = clock + 1
                if clock >= 250:
                    self.caches[_tag]["size"] = 0
                    self.__flush_cache__(_tag)

    def __flush_cache__(self, tag):
        print(f"-> Flush storage for:{tag} step:{self.caches[tag]['step']} size:{len(self.caches[tag]['inputs'])}")
        self.__save_data__(self.storage_root, self.prefix, self.caches[tag])
        self.caches[tag]["size"] = 0
        self.caches[tag]["clock"] = 0
        self.caches[tag]["save_idx"] += 1
        self.caches[tag]["inputs"] = []
        self.caches[tag]["outputs"] = []
        self.caches[tag]["features"] = []

    def add_query(self, inputs, outputs, features, **kwargs):
        assert "tag" in kwargs.keys()
        tag = kwargs["tag"]
        self.__init_client__(tag)
        self.__push_data__(inputs, outputs, features, **kwargs)