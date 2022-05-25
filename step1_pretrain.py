#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2022/04/12, homeway'

"""
train a pre
"""

import copy
import torch
import torch.backends.cudnn as cudnn
from datasets.loader import FediLoader as DataLoader
from models.loader import FediModel
from utils import helper
helper.set_default_seed()
best_acc = 0.0


def train(model, optimizer, train_loader, epoch, device, backends):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if backends:
            inputs = inputs.cuda()
            targets = targets.cuda()
        else:
            inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        helper.progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return model


def test(model, test_loader, device, backends):
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            if backends:
                inputs = inputs.cuda()
                targets = targets.cuda()
            else:
                inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            helper.progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% Best_Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, best_acc, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    return acc


def save_dnn(arch, task, model, backends, acc):
    global best_acc
    if acc > best_acc:
        best_acc = acc
        dnn = copy.deepcopy(model)
        if backends:
            dnn = copy.deepcopy(model.module)
        helper.save(dnn.cpu(), arch=arch, dataset=task)


def pretrain(args):
    global best_acc

    # load model
    model_loader = FediModel(ckpt=args.ckpt_root)
    model = model_loader.get_model(arch=args.vic_model, task=args.vic_data, num_classes=10, pretrain=False)
    # load dataset
    loader = DataLoader(data_root=args.data_root, dataset=args.vic_data, batch_size=128)
    train_loader, test_loader = loader.get_loader()

    # load from pretrain cache!
    weights = model_loader.load(arch=args.vic_model, task=args.vic_data)
    if weights is not None:
        print("-> load model from pretrained ckpt!")
        model.load_state_dict(weights)
        return model

    if args.backends:
        model = model.cuda()
        model = torch.nn.DataParallel(model, device_ids=[0, 2])
        cudnn.benchmark = True
        print("-> running using backends")
    else:
        model = model.to(args.device)


    start_epoch = 0
    split_step = [150, 250, 350]
    dataset = args.vic_data.lower()
    if dataset == "imagenet32x":
        split_step = [50, 100, 150]
    if dataset == "svhn":
        split_step = [30, 60, 100]
    if dataset == "mnist":
        split_step = [10, 20, 30]
        args.lr = 0.01
    if dataset == "caltech256":
        split_step = [200, 300, 400]
    print("-> split_step", split_step)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    for epoch in range(start_epoch, start_epoch + split_step[0]):
        train(model, optimizer, train_loader, epoch, device=args.device, backends=args.backends)
        acc = test(model, test_loader, device=args.device, backends=args.backends)
        save_dnn(args.vic_model, args.vic_data, model, args.backends, acc)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr/10.0, momentum=0.9, weight_decay=args.weight_decay)
    for epoch in range(start_epoch + split_step[0], start_epoch + split_step[1]):
        train(model, optimizer, train_loader, epoch, device=args.device, backends=args.backends)
        acc = test(model, test_loader, device=args.device, backends=args.backends)
        save_dnn(args.vic_model, args.vic_data, model, args.backends, acc)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr/100.0, momentum=0.9, weight_decay=args.weight_decay)
    for epoch in range(start_epoch + split_step[1], start_epoch + split_step[2]):
        train(model, optimizer, train_loader, epoch, device=args.device, backends=args.backends)
        acc = test(model, test_loader, device=args.device, backends=args.backends)
        save_dnn(args.vic_model, args.vic_data, model, args.backends, acc)
    return model


if __name__ == "__main__":
    args = helper.get_args()
    helper.set_seed(args)
    pretrain(args)























