#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'homeway'
__copyright__ = 'Copyright © 2021/05/15, homeway'

"""
Show example to query & response in MLaaS platform
"""

import io
import pickle
import torch
import requests
from MLaaS.server.blackbox import BlackBox
from MLaaS.server.storage import Storage
from flask import Flask, request, abort, make_response


step = 0
def query(inputs, device=torch.device("cuda:0"), action="query", host="192.168.1.140", port=9898, **kwargs):
    global step
    assert action in ["query", "info"]
    assert "tag" in kwargs.keys()
    url = f"http://{host}:{port}/query?tag={kwargs['tag']}&step={step}"
    assert "epoch" in kwargs.keys()
    try:
        headers = {"content-type": "application/json;charset=UTF-8"}
        # send
        buff = io.BytesIO()
        data = {
            "tag": kwargs['tag'],
            "epoch": kwargs['epoch'],
            "inputs": inputs.cpu()
        }
        buff.write(pickle.dumps(data))

        # receive
        response = requests.post(url, data=buff.getvalue(), headers=headers)
        buff = io.BytesIO(response.content)
        res = pickle.load(buff)
        outputs = res["outputs"].to(device)
    except Exception as e:
        print(e)
        return exit(1)
    step += 1
    return outputs


def run_server(args, blackbox):
    app = Flask(__name__)
    @app.route("/query", methods=["POST"])
    def predict():
        if request.method == "POST":
            # receive
            req = pickle.load(io.BytesIO(request.data))
            tag = req["tag"]
            inputs = req["inputs"].to(args.device)
            epoch = 0 if "epoch" not in req.keys() else int(req["epoch"])
            outputs = blackbox.query(inputs, tag=tag, epoch=epoch)

            # send
            buff = io.BytesIO()
            data = {
                "idxs": [],
                "outputs": outputs.cpu()
            }
            buff.write(pickle.dumps(data))
            res = make_response(buff.getvalue())
            res.headers.set("Content-Type", "application/octet-stream")
            res.headers.set('Content-Disposition', 'attachment', filename='outputs')
            return res
        else:
            abort(403)
    app.run(debug=False,  threaded=True, port=args.port, host=args.host)