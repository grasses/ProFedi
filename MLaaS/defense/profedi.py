#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/03/31, homeway'


from tqdm import tqdm
import torch
import numpy as np
from utils import helper
args = helper.get_args()
helper.set_seed(args)


def get_fdi_feats(model, data_loader, anchor_feats, **kwargs):
    """
    extract feature distortion from data_loader
    :param victim:
    :param data_loader:
    :param anchor_features:
    :return:
    """
    model.eval()
    victim = model.to(args.device)
    feature_distortion_index = []
    predict_label = []
    phar = tqdm(enumerate(data_loader))
    for step, (x, y) in phar:
        x = x.to(args.device)
        out_logits, out_list = victim.feature_list(x)
        out_pred = out_logits.argmax(dim=1).cpu()
        dist = get_feature_distortion(out_pred.numpy(), out_list, anchor_feats)
        predict_label.append(out_pred)
        feature_distortion_index.append(dist)
        phar.set_description(f"Get FDI for feature layers: [{step}/{len(data_loader)}]")
        phar.update(1)
    label = torch.cat(predict_label).cpu()
    fdi = torch.cat(feature_distortion_index).cpu()
    return fdi, label


def get_feature_distortion(pred_labels, out_features, anchor_features):
    """
    :param pred_labels:
    :param out_features:
    :param anchor_features:
    :return: batch feature distortion [batch_size, layer_size, anchor_size]
    """
    anchor_size = len(anchor_features[0][0])
    layer_size = len(anchor_features[0])
    batch_size = int(pred_labels.shape[0])
    batch_distortion = torch.zeros([batch_size, layer_size, anchor_size])
    for fidx in range(layer_size):
        out_features[fidx] = out_features[fidx].detach().to(args.device)
    for fidx in range(layer_size):
        per_feats = out_features[fidx].view(batch_size, -1)
        for idx in range(batch_size):
            label = int(pred_labels[idx])
            anchor_fts = anchor_features[label][fidx].to(args.device)
            repeated_feats = torch.repeat_interleave(per_feats[idx].view(1, -1), repeats=anchor_size, dim=0)
            batch_distortion[idx, fidx] = torch.norm(repeated_feats - anchor_fts, dim=1, p=2)
    return batch_distortion.detach().cpu()


def get_anchor_feats(model, test_loader, num_classes, anchor_size=100):
    """
    :param model:
    :param test_loader:
    :param anchor_size:
    :return: anchor_features
    anchor_features = {label: [torch.tensor]}
    """
    device = next(model.parameters()).device
    model = model.to(device)
    label_cnt = np.zeros([num_classes], dtype=np.int32)
    anchor_features = {}
    for i in range(num_classes):
        anchor_features[i] = {}
    max_iteration = int((anchor_size - 1) * num_classes)

    with torch.no_grad():
        for step, (x, y) in enumerate(test_loader):
            x = x.to(device)
            logits, out_list = model.feature_list(x)
            pred = logits.argmax(dim=1).view_as(y)
            prob = torch.softmax(logits, dim=1).detach().cpu()
            # 初始化features字典 [label_idx, layer_idx, numpy.array()]
            for layer_idx in range(len(out_list)):
                out_list[layer_idx] = out_list[layer_idx].detach().cpu()
                if step == 0:
                    for c in range(num_classes):
                        anchor_features[c][layer_idx] = torch.zeros([anchor_size, out_list[layer_idx].shape[1]])
            # 获取layer feature space并放入
            for sample_idx in range(len(x)):
                label = int(pred[sample_idx])
                label_idx = int(label_cnt[label])
                if float(prob[sample_idx][label]) < 0.98:
                    continue
                if label_cnt[label] == anchor_size - 1:
                    continue

                for layer_idx in range(len(out_list)):
                    val = out_list[layer_idx][sample_idx].view(-1)
                    anchor_features[label][layer_idx][label_idx] = val
                label_cnt[label] += 1
            if np.sum(label_cnt) == max_iteration:
                break
    return anchor_features