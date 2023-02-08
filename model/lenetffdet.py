import os
import pathlib
from abc import abstractmethod
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from PIL import Image
from empatches import EMPatches
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch import nn, optim, ops
from torchvision.transforms import transforms
from tqdm import tqdm

from model.FocalLoss import FocalLoss
from model.dcn import DeformableConv2d
from model.lenetdet import evaluate_mAP, calc_mAP
from model.lenetff import LeNetFF, overlay_y_on_x_batch, get_batch_of_hybrid_image
from options import Options


def evaluate(writer, net, opt, test_loader, classes, op_dir, batch_idx):
    with torch.no_grad():
        print("\n\tTesting...")
        acc_list = []
        fpr_list = []
        tpr_list = []
        f1_list = []
        pred_y = []
        actual_y = []
        tps = []
        tns = []
        fps = []
        fns = []
        for idx, ((batch_x, batch_y), meta) in enumerate(tqdm(test_loader)):
            for x, y in zip(batch_x, batch_y):
                one_goodness, y_pred = net.predict(x, len(classes))
                pred_y.append(one_goodness.item())
                actual_y.append(y.item())
        actual_y, pred_y = np.array(actual_y), np.array(pred_y)
        for class_idx in range(len(classes)):
            one_vs_rest_actual_y = actual_y == class_idx
            one_vs_rest_pred_y = pred_y == class_idx
            # evaluate
            confusion_mat = sklearn.metrics.confusion_matrix(one_vs_rest_actual_y, one_vs_rest_pred_y)
            tp, fn, fp, tn = confusion_mat.flatten()
            acc = np.sum((tp, tn)) / np.sum((tn, fp, fn, tp))
            f1 = (2 * tp) / np.sum((2 * tp, fp, fn))
            fpr = fp / np.sum((tn, fp))
            tpr = tp / np.sum((tp, fn))

            acc_list.append(acc)
            fpr_list.append(fpr)
            f1_list.append(f1)
            tpr_list.append(tpr)

            tps.append(tp)
            tns.append(tn)
            fps.append(fp)
            fns.append(fn)

            writer.add_scalars('Accuracy', {classes[class_idx]: acc}, batch_idx + 1)
            writer.add_scalars('F1-Score', {classes[class_idx]: f1}, batch_idx + 1)
            writer.add_scalars('FPR', {classes[class_idx]: fpr}, batch_idx + 1)
            writer.add_scalars('TPR', {classes[class_idx]: tpr}, batch_idx + 1)

        writer.add_scalars('Evaluation', {
            "µAccuracy": np.mean(acc_list),
            'µF1-Score': np.mean(f1_list),
            'µFPR': np.mean(fpr_list),
            'µTPR': np.mean(tpr_list)
        }, batch_idx + 1)
        print({
            "µAccuracy": np.mean(acc_list),
            'µF1-Score': np.mean(f1_list),
            'µFPR': np.mean(fpr_list),
            'µTPR': np.mean(tpr_list)
        })

        acc_list.append(np.mean(acc_list))
        f1_list.append(np.mean(f1_list))
        fpr_list.append(np.mean(fpr_list))
        tps.append(0)
        tns.append(0)
        fps.append(0)
        fns.append(0)
        try:
            pd.DataFrame(dict(
                Accuracy=acc_list,
                F1=f1_list,
                FPR=fpr_list,
                TPR=tpr_list,
                TP=tps,
                FP=fps,
                TN=tns,
                FN=fns,
            )).to_csv(str(op_dir / "eval.csv"))
        except:
            ...
        with open(str(op_dir / "model_summary.txt"), "a+") as fp:
            s = 'Accuracy: %s\n' % acc_list
            fp.write(s)
            s = 'F1-Score: %s\n' % f1_list
            fp.write(s)
            s = 'FPR: %s\n' % fpr_list
            fp.write(s)
            s = 'TPR: %s\n' % tpr_list
            fp.write(s)
            s = 'TP: %s\n' % tps
            fp.write(s)
            s = 'TN: %s\n' % tns
            fp.write(s)
            s = 'FP: %s\n' % fps
            fp.write(s)
            s = 'FN: %s\n' % fns
            fp.write(s)


def evaluate_mAP(classes, epoch_idx, num_epochs, model, test_loader, meta_dict, opt, detailed_output=False,
                 iou_threshold=0.5):
    print("\nTesting... epoch:[%d/%d]" % (epoch_idx + 1, num_epochs))
    model.eval()
    labels_all, probs_all, gt_bbox_all, prob_bbox_all = [], [], [], []
    with torch.no_grad():
        for (batch_x, batch_y, bboxs), meta in tqdm(test_loader, leave=False, total=len(test_loader)):
            if opt.ffnegalg == "overlay":
                x_pos = overlay_y_on_x_batch(batch_x, batch_y, len(classes))
                rnd = torch.randperm(batch_x.size(0))
                x_neg = overlay_y_on_x_batch(batch_x, batch_y[rnd], len(classes))
            elif opt.ffnegalg == "hybrid":
                x_pos = batch_x
                x_neg = get_batch_of_hybrid_image(batch_x, batch_y, opt, classes)
            bboxs_pred, logits_pred = model(x_pos)
            probs = torch.softmax(logits_pred, dim=-1)
            probs_all.append(probs.cpu().numpy())
            labels_all.append(batch_y.cpu().numpy())
            gt_bbox_all.append(bboxs.cpu().numpy())
            prob_bbox_all.append(bboxs_pred.cpu().numpy())
    # flatten the items
    labels_all = np.concatenate(labels_all)
    probs_all = np.concatenate(probs_all)
    gt_bbox_all = np.concatenate(gt_bbox_all)
    prob_bbox_all = np.concatenate(prob_bbox_all)
    n_classes = probs_all.shape[1]
    auc_list = []
    auc_dict = defaultdict(lambda: [])
    for class_idx in tqdm(range(n_classes), leave=False):
        labels_binary = (labels_all == class_idx).astype(int)
        # auc score
        try:
            auc = roc_auc_score(labels_binary, probs_all[:, class_idx])
        except ValueError:
            # this happens when a batch contains negative or positive samples as ground_truth
            auc = 0.0
        auc_list.append(auc)
        auc_dict[class_idx].append(auc)
    mean_auc = np.mean(auc_list)
    for key, values in auc_dict.items():
        auc_dict[key] = np.mean(values)

    # mAP calculation
    # mAP score
    mAP = calc_mAP(gt_bbox_all, prob_bbox_all, probs_all, iou_threshold=iou_threshold)

    model.eval(False)
    if detailed_output:
        return mean_auc, mAP, auc_list, auc_dict
    else:
        return mean_auc, mAP


def train_loop(opt, classes, writer, train_loader, test_loader, val_loader):
    now = datetime.now()
    dt_string_start = now.strftime("%d/%m/%Y %H:%M:%S\n")
    num_batches = len(train_loader)
    sample_batch = torch.rand((opt.batchsize, opt.nc, opt.isize, opt.isize))

    class_balance_weights = 1.0 - train_loader.dataset.class_proportion
    class_balance_weights = torch.from_numpy(class_balance_weights.astype('float32')).to(opt.device)
    # loss functions
    if opt.loss == 'SFL':
        criterion_class_loss = lambda pred, tgt: ops.sigmoid_focal_loss(pred, tgt, alpha=class_balance_weights,
                                                                        reduction='mean')
    elif opt.loss == 'FL':
        criterion_class_loss = FocalLoss(alpha=class_balance_weights, gamma=2.0)
    elif opt.loss == 'BCE':
        criterion_class_loss = nn.CrossEntropyLoss(
            weight=class_balance_weights)
    else:
        # default cross entropy loss
        criterion_class_loss = nn.CrossEntropyLoss()
    # loss for bbox regressor
    criterion_bbox_loss = nn.SmoothL1Loss()

    def calculate_classification_loss(logits_pred, labels, loss):
        # calculate the loss for classification
        class_loss_value = criterion_class_loss(logits_pred.view(-1, len(classes)), labels.view(-1))
        return loss + class_loss_value

    def calculate_regression_loss(bboxes_pred, bboxes, loss):
        bbox_loss_value = criterion_bbox_loss(bboxes_pred, bboxes)
        return loss + bbox_loss_value

    net = LeNetFF(sample_batch=sample_batch, num_classes=len(classes), epoch=opt.niter,
                  channels=opt.nc,
                  goodness=opt.ffgoodness, negative_image_algo=opt.ffnegalg,
                  batchnorm=opt.batchnorm,
                  layernorm=opt.layernorm,
                  dropout=opt.dropout,
                  num_anchors=1,  # for detection
                  additional_class_loss=calculate_classification_loss,
                  additional_bbox_loss=calculate_regression_loss,
                  lr=opt.lr, deformable=opt.deformable, writer=writer)

    print("================Model Summary===============")
    op_dir = pathlib.Path(opt.outf) / opt.name
    os.makedirs(str(op_dir), exist_ok=True)
    with open(str(op_dir / "model_summary.txt"), "w") as fp:
        s = "\n%s\n" % net
        fp.write(opt.name)
        fp.write(s)
        print(s)
    print("============================================")
    net.to(opt.device)
    print("\nLoading batch[%d/%d]..." % (1, len(train_loader)))
    max_auc = 0.0
    max_auc_batch=0
    max_mAP_batch=0
    max_mAP = 0.0
    perclass_max_auc = defaultdict(lambda: 0.0)
    weight_dir = pathlib.Path(opt.outf) / opt.name / "weight"
    os.makedirs(str(weight_dir), exist_ok=True)
    for batch_idx, ((batch_x, batch_y, bboxs), meta) in enumerate(train_loader):
        meta_dict = dict(
            total_train_loss=0.0,
            train_running_loss=0.0,
            test_running_loss=0.0,
            batch_ctr=0,
            step_ctr=0,
        )
        print("Training Batch: [%d/%d]" % (batch_idx + 1, num_batches))
        if opt.ffnegalg == "overlay":
            x_pos = overlay_y_on_x_batch(batch_x, batch_y, len(classes))
            rnd = torch.randperm(batch_x.size(0))
            x_neg = overlay_y_on_x_batch(batch_x, batch_y[rnd], len(classes))
        elif opt.ffnegalg == "hybrid":
            x_pos = batch_x
            x_neg = get_batch_of_hybrid_image(batch_x, batch_y, opt, classes)

        # write to tensorboard
        writer.add_image('Positive Images', torchvision.utils.make_grid(x_pos))
        writer.add_image('Negative Images', torchvision.utils.make_grid(x_neg))

        # x_pos = x_pos.view(-1, 3 * 128 * 128)  # flatten(but batch wise)
        # x_neg = x_neg.view(-1, 3 * 128 * 128)  # flatten(but batch wise)
        net.set_detection_input(batch_y, bboxs)
        net.train(x_pos, x_neg, batch_idx)

        # unlike backprop, in FF we evaluate per-batch (not per-epoch)
        # draw bbox
        # draw_bbox(writer, net, opt)
        # test
        detailed = True
        auc, mAP, auc_ls, auc_dict = evaluate_mAP(classes, batch_idx, len(train_loader), net, test_loader, meta_dict,
                                                  opt,
                                                  detailed_output=detailed, iou_threshold=opt.iou)
        if auc > max_auc \
                or any([cls_auc > perclass_max_auc[cls_idx] for cls_idx, cls_auc in auc_dict.items()]) \
                or mAP > max_mAP:
            if auc > max_auc:
                max_auc = auc
                max_auc_batch = batch_idx + 1
            if mAP > max_mAP:
                max_mAP = mAP
                max_mAP_batch = batch_idx + 1
            perclass_max_auc = auc_dict

            # save the model
            torch.save({'epoch': batch_idx, 'state_dict': net.state_dict()},
                       f'{str(weight_dir)}/net_%d_%f.pth' % (batch_idx, auc))
        writer.add_scalar('Testing AUC', auc, batch_idx)
        writer.add_scalar('Testing mAP', mAP, batch_idx)
        for cls_idx, auc in auc_dict.items():
            writer.add_scalar('Testing AUC[%s]' % classes[cls_idx], auc, batch_idx)
        print(f"Testing epoch:[{batch_idx + 1}/{len(train_loader)}] Micro-averaged "
              f"AUC:{auc:.2f} mAP:{mAP:.2f} ",
              end="Class wise AUC [")
        for class_idx, cls_auc in auc_dict.items():
            print("%s:%0.2f, " % (classes[class_idx], cls_auc), end="")
        print(
            f"]. MAX AUC:{max_auc:.2f}@epoch-{max_auc_batch} MAX mAP{max_mAP:.2f}@epoch-{max_mAP_batch}")
    with open(str(op_dir / "model_summary.txt"), "a+") as fp:
        now = datetime.now()
        dt_string_end = now.strftime("%d/%m/%Y %H:%M:%S\n")
        fp.write(dt_string_start)
        fp.write(dt_string_end)
        print(dt_string_start.strip())
        print(dt_string_end.strip())
    # # save the model
    # torch.save({'epoch': epoch_idx, 'state_dict': net.state_dict()},
    #            f'{str(weight_dir)}/net_last.pth' % (epoch_idx, auc))
