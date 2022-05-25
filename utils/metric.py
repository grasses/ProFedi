

import os
from sklearn import metrics
import torch
import torch.nn.functional as F
import numpy as np
from utils import helper
from tqdm import tqdm
args = helper.get_args()
helper.set_seed(args)


def multi_mertic(y, p, scores=None):
    result_dict = {}
    t_idx = np.where(y == 0)[0]
    f_idx = np.where(y == 1)[0]
    TP = len(np.where(p[t_idx] == 0)[0])
    FP = len(np.where(p[f_idx] == 0)[0])
    FN = len(np.where(p[t_idx] == 1)[0])
    TN = len(np.where(p[f_idx] == 1)[0])
    print(f"-> TP={TP}, FP={FP}, FN={FN}, TN={TN}")

    result_dict["TP"] = TP
    result_dict["FP"] = FP
    result_dict["FN"] = FN
    result_dict["TN"] = TN
    result_dict["FPR100"] = 100.0 * FP / (FP + TN)
    result_dict["TPR100"] = 100.0 * TP / (TP + FN)
    result_dict["ACC"] = round(100.0 * (TP + TN) / (TP + FP + TN + FN), 5)
    result_dict["Recall"] = round(100.0 * (TP) / (TP + FN), 5)
    result_dict["Precision"] = round(100.0 * (TP) / (TP + FP), 5)
    result_dict["F1score"] = round((2.0 * result_dict["Precision"] * result_dict["Recall"]) /
                                   (result_dict["Precision"] + result_dict["Recall"]), 5)
    if scores is not None:
        fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
        result_dict["FPR"] = fpr
        result_dict["TPR"] = tpr
        result_dict["thresholds"] = thresholds
    return result_dict


def test(classifier, ben_test_loader, adv_test_loader, epoch=0, tau1=0.5, file_path=None):
    classifier.eval()
    classifier = classifier.to(args.device)
    total, sum_query_correct, sum_correct, sum_loss = 0, 0, 0, 0.0,
    result_dict = {
        "y": [],
        "pred": [],
        "query_y": [],
        "query_pred": [],
        "scores": [],
        "query_scores": [],
        "conf": []
    }
    with torch.no_grad():
        step = 0
        for (x, y) in ben_test_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            output = classifier(x)
            loss = F.cross_entropy(output, y)
            total += y.size(0)
            sum_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            scores = F.softmax(output)[:, 1].detach().cpu()

            result_dict["scores"].append(scores)
            result_dict["y"].append(y.detach().cpu())
            result_dict["pred"].append(pred.view([-1]).detach().cpu())
            sum_correct += pred.eq(y.view_as(pred)).sum().item()

            query_scores = float(1.0 * pred.sum() / len(pred))
            detect_flag = 1 if query_scores > tau1 else 0
            truth_flag = 1 if (1.0 * y.sum() / len(y)) > tau1 else 0
            result_dict["query_y"].append(truth_flag)
            result_dict["query_pred"].append(detect_flag)
            result_dict["query_scores"].append(query_scores)
            if detect_flag == truth_flag:
                sum_query_correct += 1
            info = "[Test] epoch:{:d} Loss: {:.6f} Acc:{:.3f}%".format(
                epoch,
                sum_loss / total,
                100.0 * sum_correct / total
            )
            helper.progress_bar(step, 2*len(adv_test_loader), info)
            step += 1

        for (x, y) in adv_test_loader:
            x = x.to(args.device)
            y = y.to(args.device)
            output = classifier(x)
            loss = F.cross_entropy(output, y)
            total += y.size(0)
            sum_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            scores = F.softmax(output)[:, 1].detach().cpu()
            result_dict["scores"].append(scores)
            result_dict["y"].append(y.detach().cpu())
            result_dict["pred"].append(pred.view([-1]).detach().cpu())
            sum_correct += pred.eq(y.view_as(pred)).sum().item()
            query_scores = float(1.0 * pred.sum() / len(pred))
            detect_flag = 1 if query_scores > tau1 else 0
            truth_flag = 1 if (1.0 * y.sum() / len(y)) > tau1 else 0
            result_dict["query_y"].append(truth_flag)
            result_dict["query_pred"].append(detect_flag)
            result_dict["query_scores"].append(query_scores)
            if detect_flag == truth_flag:
                sum_query_correct += 1
            info = "[Test] epoch:{:d} Loss: {:.6f} Acc:{:.3f}%".format(
                epoch,
                sum_loss / total,
                100.0 * sum_correct / total
            )
            helper.progress_bar(step, 2*len(adv_test_loader), info)
            step += 1

    result_dict["y"] = torch.cat(result_dict["y"]).detach().cpu().numpy()
    result_dict["pred"] = torch.cat(result_dict["pred"]).detach().cpu().numpy()
    result_dict["scores"] = torch.cat(result_dict["scores"]).detach().cpu().numpy()
    result_dict["query_y"] = torch.tensor(result_dict["query_y"]).detach().cpu().numpy()
    result_dict["query_pred"] = torch.tensor(result_dict["query_pred"]).detach().cpu().numpy()
    result_dict["query_scores"] = np.array(result_dict["query_scores"])

    y = result_dict["y"]
    p = result_dict["pred"]
    sample_res = multi_mertic(y, p)

    y = result_dict["query_y"]
    p = result_dict["query_pred"]
    query_res = multi_mertic(y, p)

    print(
        f"""-> TEST_SAMPLE(τ1={tau1}) ACC:{sample_res['ACC']}% 
    Recall:{sample_res["Recall"]} Precision:{sample_res["Precision"]} F1-score:{sample_res["F1score"]}
    FPR:{sample_res['FPR100']} TPR:{sample_res['TPR100']}""")
    print(
        f"""-> TEST_QUERY(τ1={tau1}) ACC:{query_res['ACC']}% 
    Recall:{query_res["Recall"]} Precision:{query_res["Precision"]} F1-score:{query_res["F1score"]}
    FPR:{query_res['FPR100']} TPR:{query_res['TPR100']}""")

    result_dict.update(query_res)
    if file_path is not None:
        print("-> save result", file_path)
        torch.save(result_dict, file_path)
    return result_dict




def train(classifier, train_loader, optimizer):
    classifier.train()
    classifier = classifier.to(args.device)
    for step, (x, y) in enumerate(train_loader):
        x = x.to(args.device).view(-1, 500)
        y = y.to(args.device)
        optimizer.zero_grad()
        out = classifier(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
    return classifier