#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/05/20, homeway'

import os
import sys
import time
import yaml
import argparse
from os import path as osp
import random
import numpy as np
import datetime
import torch.nn.functional as F
global step
step = 1


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_default_seed():
    set_seed(get_args())


def get_args():
    ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    parser = argparse.ArgumentParser()
    # system params
    parser.add_argument("--type", type=str, choices=["cli", "server", "client", "demo"], default="cli", help="running type of program")
    parser.add_argument("--conf", type=str, help="config file")
    parser.add_argument("--ckpt_root", default=osp.join(ROOT, "models/ckpt"))
    parser.add_argument("--out_root", default=osp.join(ROOT, "output"))
    parser.add_argument("--data_root", default=osp.join(ROOT, "datasets/data"))

    # machine learning params
    parser.add_argument("--momentum", default=0.5, type=float, help="momentum (default: 0.5)")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight_decay (default: 0.0005)")
    parser.add_argument("--lr", default=0.1, type=float, help="Learning rate (default: 0.1)")
    parser.add_argument("--batch_size", default=128, type=int, help="Blackbox server cache size")
    parser.add_argument("--cache_size", default=50000, type=int, help="Blackbox server cache size")
    parser.add_argument("--seed", default=1000, type=int, help="Default seed of numpy/pyTorch")
    parser.add_argument("--backends", default=False, type=bool)
    parser.add_argument("--device", default=0, type=int, help="GPU device id")

    # web params
    parser.add_argument("--host", default="0.0.0.0", type=str, help="Server host for MLaaS")
    parser.add_argument("--port", default=5678, type=int, help="Server port for MLaaS")

    # init system
    args = parser.parse_args()
    set_seed(args)
    format_time = str(datetime.datetime.now().strftime("%Y%m%d_%H%M"))
    args.env_name = str(format_time)

    # load yaml
    path = osp.join(ROOT, "conf", args.conf)
    assert os.path.exists(path)
    for k, v in yaml.safe_load(open(path)).items():
        setattr(args, k, v)

    if args.backends:
        args.device = torch.device("cuda")
    else:
        args.device = get_device(args)

    # build directory
    args.ROOT = ROOT
    args.out_path = osp.join(ROOT, "output")
    args.data_path = osp.join(ROOT, "datasets/data")
    args.recv_path = osp.join(ROOT, "output/query")
    args.fedi_path = osp.join(ROOT, "output/fedi")
    args.exp_path = [osp.join(ROOT, "output/exp"), osp.join(ROOT, "output/exp_1.1"), osp.join(ROOT, "output/exp_1.2"),
                     osp.join(ROOT, "output/exp_1.3"), osp.join(ROOT, "output/exp_2.1"),
                     osp.join(ROOT, "output/exp_2.2")]
    path_list = [args.out_root, args.data_root, args.ckpt_root, args.recv_path, args.fedi_path] + args.exp_path
    for path in path_list:
        try:
            os.makedirs(path)
        except Exception as e:
            pass
    args.tag = f"{args.vic_model}_{args.vic_data}_{args.atk_model}"
    return args


def get_device(args):
    device = torch.device(f"cpu")
    if torch.cuda.is_available():
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        device = torch.device(f"cuda:{args.device}")
    return device


def save(model, arch, dataset):
    args = get_args()
    path = osp.join(args.ckpt_root, f"{arch}_{dataset}.pt")
    print("-> Save model: {}_{}.pt".format(arch, dataset))
    return torch.save(model.cpu().state_dict(), path)


def load(arch, dataset):
    args = get_args()
    path = osp.join(args.ckpt_root, f"{arch}_{dataset}.pt")
    if not os.path.exists(path):
        raise RuntimeError(f"File:{path} not found!")
        return None
    print("-> Load pretrain model: {}_{}.pt".format(arch, dataset))
    return torch.load(path, map_location="cpu")


step = 0
import io
import torch
import pickle
import requests
def query(inputs, device=torch.device("cuda:0"), action="query", host="192.168.1.140", port=9898, **kwargs):
    global step
    assert action in ["query", "info"]
    assert "tag" in kwargs.keys()
    url = f"http://{host}:{port}/query?tag={kwargs['tag']}&step={step}"

    assert "epoch" in kwargs.keys()
    try:
        headers = {"content-type": "application/json;charset=UTF-8"}
        # send pickle file
        buff = io.BytesIO()
        data = {
            "tag": kwargs['tag'],
            "epoch": kwargs['epoch'],
            "inputs": inputs.cpu()
        }
        buff.write(pickle.dumps(data))
        # receive pickle file
        response = requests.post(url, data=buff.getvalue(), headers=headers)
        buff = io.BytesIO(response.content)
        res = pickle.load(buff)
        outputs = res["outputs"].to(device)
    except Exception as e:
        print(e)
        exit(1)
    step += 1
    return outputs


def test_fidelity(student, blackbox, test_loader, device, epoch=0, debug=False):
    test_loss = 0.0
    correct = 0.0
    student.to(device)
    blackbox.to(device)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            output = student(x).cpu()
            output_T = blackbox(x).cpu()
            loss = F.mse_loss(output, output_T)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            t_label = output_T.argmax(dim=1, keepdim=True)
            correct += pred.eq(t_label.view_as(pred)).sum().item()
    test_loss /= (1.0 * len(test_loader.dataset))
    acc = 100.0 * correct / len(test_loader.dataset)

    msg = "-> For E{:d}, [Test] fidelity_loss={:.5f}, fidelity_acc={:.3f}%".format(
            int(epoch),
            test_loss,
            acc
    )
    if debug: print(msg)
    return acc, test_loss


def test(model, test_loader, device, epoch=0, debug=False):
    test_loss = 0.0
    correct = 0.0
    model.to(device)
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= (1.0 * len(test_loader.dataset))
    acc = 100.0 * correct / len(test_loader.dataset)

    msg = "-> For E{:d}, [Test] loss={:.5f}, acc={:.3f}%".format(
            int(epoch),
            test_loss,
            acc
    )
    if debug: print(msg)
    return acc, test_loss


_, term_width = ['40', '181']#os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)
    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f














