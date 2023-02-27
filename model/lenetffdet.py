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
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix
from torch import nn, optim, ops
from torchmetrics.detection import MeanAveragePrecision
from torchvision.transforms import transforms
from tqdm import tqdm

from model.FocalLoss import FocalLoss
from model.dcn import DeformableConv2d
from model.lenetdet import calc_mAP
from model.lenetff import LeNetFF, overlay_y_on_x_batch, get_batch_of_hybrid_image
from options import Options


# helper function
def add_pr_curve_tensorboard(writer, class_name, class_index, test_probs, test_label, global_step, csv_writer=None):
    '''
    Takes in a "class_index" from 0 to 10 and plots the corresponding
    precision-recall curve
    '''
    y_true = test_label == class_index
    y_pred_proba = test_probs[:, 0, class_index].view(-1, 1)  # probabilities

    # average performance sampled over 127 performance metricsZ
    accs, f1s, fprs, tprs, prs, rcs = [], [], [], [], [], []
    numerical_stability = 1e-20
    num_thresholds = 127
    for idx, th in enumerate(np.linspace(0.0, 1.0, num_thresholds)):
        y_pred = y_pred_proba > th
        confusion_mat = confusion_matrix(y_true.view(-1), y_pred.view(-1), labels=[False, True])
        # print(confusion_mat.shape, len(y_true.view(-1)), len(y_pred.view(-1)), idx, th, global_step)
        ((tn, fp), (fn, tp)) = confusion_mat
        acc = np.sum((tp, tn)) / np.sum((tn, fp, fn, tp))
        f1 = (2 * tp) / (np.sum((2 * tp, fp, fn)) + numerical_stability)
        fpr = fp / (np.sum((tn, fp)) + numerical_stability)
        tpr = tp / (np.sum((tp, fn)) + numerical_stability)
        pr = tp / (np.sum((tp, fp)) + numerical_stability)
        rc = tp / (np.sum((tp, fn)) + numerical_stability)
        accs.append(acc)
        f1s.append(f1)
        fprs.append(fpr)
        tprs.append(tpr)
        prs.append(pr)
        rcs.append(rc)
        aggregator = lambda x: np.mean(x)
        uacc, uf1, ufpr, utpr = aggregator(accs), aggregator(f1s), aggregator(fprs), aggregator(tprs)
        upr, urc = aggregator(prs), aggregator(rcs)

        writer.add_scalars('Evaluation-%s' % class_name, {
            "precision": upr,
            'recall': urc,
            'f1_score': uf1,
            'accuracy': uacc,
            'fpr': ufpr,
            'tpr': utpr,
        }, global_step)

        writer.add_pr_curve(tag=class_name,
                            labels=y_true,
                            predictions=y_pred_proba,
                            num_thresholds=num_thresholds,
                            global_step=global_step)


def evaluate_mAP(writer, classes, num_classes, step, total_step, model, test_loader, opt, iou_threshold,
                 detailed_output=False, csv_writer=None):
    labels_all, probs_all, gt_bbox_all, prob_bbox_all = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch_idx, ((batch_x, batch_y, bboxs), meta) in enumerate(test_loader):
            print("Testing Epoch:[%d/%d] batch :[%d/%d]....." % (step + 1, total_step, batch_idx + 1, len(test_loader)),
                  end="\b\r")
            if opt.ffnegalg == "overlay":
                x_pos = overlay_y_on_x_batch(batch_x, batch_y, num_classes)
                rnd = torch.randperm(batch_x.size(0))
                x_neg = overlay_y_on_x_batch(batch_x, batch_y[rnd], num_classes)
            elif opt.ffnegalg == "hybrid":
                x_pos = batch_x
                x_neg = get_batch_of_hybrid_image(batch_x, batch_y, opt, classes)
            bboxs_pred, logits_pred = model(x_pos)
            probs_pred = torch.softmax(logits_pred, dim=-1)
            # draw prediction box
            # draw_bbox(bboxs, bboxs_pred, batch_y, probs_pred, classes)
            # prepare list
            probs_all.append(probs_pred.cpu().numpy())
            labels_all.append(batch_y.cpu().numpy())
            gt_bbox_all.append(bboxs.cpu().numpy())
            prob_bbox_all.append(bboxs_pred.cpu().numpy())

            # draw curve
            for class_idx, class_name in enumerate(classes):
                add_pr_curve_tensorboard(writer, class_name, class_idx,
                                         probs_pred.detach().cpu(), batch_y.detach().cpu(), step, csv_writer)
    # flatten the items
    labels_all = np.concatenate(labels_all)
    probs_all = np.concatenate(probs_all)
    gt_bbox_all = np.concatenate(gt_bbox_all)
    prob_bbox_all = np.concatenate(prob_bbox_all)
    n_classes = probs_all.shape[1]
    auc_list = []
    auc_dict = defaultdict(lambda: [])
    # calculate AUC
    for class_idx in range(n_classes):
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

    # mAP score
    # mAP = calc_mAP(gt_bbox_all, prob_bbox_all, probs_all, iou_threshold=iou_threshold)

    preds = []
    target = []
    for gt_blb, prob, gt_bbox, pred_bbox in zip(labels_all, probs_all, gt_bbox_all, prob_bbox_all):
        preds.append(dict(
            boxes=torch.tensor(pred_bbox),
            scores=torch.tensor([prob[0][gt_blb[0]]]),
            labels=torch.tensor(gt_blb),
        ))
        target.append(dict(
            boxes=torch.tensor(gt_bbox),
            labels=torch.tensor(gt_blb),
        ))
    metric0 = MeanAveragePrecision(box_format='xywh', iou_type='bbox', iou_thresholds=[0.2],
                                   class_metrics=True)
    metric1 = MeanAveragePrecision(box_format='xywh', iou_type='bbox', iou_thresholds=[iou_threshold],
                                   class_metrics=True)
    metric2 = MeanAveragePrecision(box_format='xywh', iou_type='bbox',
                                   iou_thresholds=list(np.arange(0.5, 0.95, 0.05, dtype="float")), class_metrics=True)
    metric3 = MeanAveragePrecision(box_format='xywh', iou_type='bbox',
                                   iou_thresholds=list(np.arange(0.05, 0.5, 0.05, dtype="float")), class_metrics=True)
    metric0.update(preds, target)
    metric1.update(preds, target)
    metric2.update(preds, target)
    metric3.update(preds, target)

    result0 = metric0.compute()
    result1 = metric1.compute()
    result2 = metric2.compute()
    result3 = metric3.compute()

    map0 = result0['map'].item()
    map1 = result1['map'].item()
    map2 = result2['map'].item()
    map3 = result3['map'].item()

    writer.add_scalars('Testing_mAP', {
        "_02": map0,
        "_%0.2f" % iou_threshold: map1,
        "_[005_05]": map3,
        "_[05_095]": map2
    }, batch_idx)
    print(result0, result1, result2, result3)
    writer.add_scalars('Testing_Class_mAP_02',
                       {classes[idx]: k.item() for idx, k in enumerate(result0['map_per_class'])}, batch_idx)
    writer.add_scalars('Testing_Class_mAP_%f' % iou_threshold,
                       {classes[idx]: k.item() for idx, k in enumerate(result1['map_per_class'])}, batch_idx)

    writer.add_scalars('Testing_Class_mAP[05_095]',
                       {classes[idx]: k.item() for idx, k in enumerate(result2['map_per_class'])}, batch_idx)
    writer.add_scalars('Testing_Class_mAP[005_05]',
                       {classes[idx]: k.item() for idx, k in enumerate(result3['map_per_class'])}, batch_idx)
    model.eval(False)
    if detailed_output:
        return mean_auc, map1, map2, auc_list, auc_dict
    else:
        return mean_auc, map1, map2


def fix_batch_of_image_for_rendering(images, invert=False):
    batch_like = torch.empty((images.shape[0], 1, images.shape[2], images.shape[3]))
    for idx, img in enumerate(images.detach().clone().cpu()):
        img = img.numpy()
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            img = img.reshape(img.shape[1], img.shape[2])

        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        batch_like[idx, 0, :, :] = torch.from_numpy(255 - img) if invert else torch.from_numpy(img)
    return batch_like


def train_loop(opt, classes, writer, train_loader, test_loader, val_loader):
    print("FF Detection...")
    now = datetime.now()
    dt_string_start = now.strftime("%d/%m/%Y %H:%M:%S\n")
    num_batches = len(train_loader)
    sample_batch = torch.rand((opt.batchsize, opt.nc, opt.isize, opt.isize))

    num_classes = len(classes)  # as detection skip "NORMAL0"

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
        class_loss_value = criterion_class_loss(logits_pred.view(-1, num_classes), labels.view(-1))
        return loss + class_loss_value

    def calculate_regression_loss(bboxes_pred, bboxes, loss):
        bbox_loss_value = criterion_bbox_loss(bboxes_pred.view(-1, 4), bboxes.view(-1, 4))
        return loss + bbox_loss_value

    net = LeNetFF(sample_batch=sample_batch, num_classes=num_classes, epoch=opt.niter,
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

    max_auc = 0.0
    max_auc_batch = 0
    max_mAP_batch = 0
    max_mAP = 0.0
    perclass_max_auc = defaultdict(lambda: 0.0)
    weight_dir = pathlib.Path(opt.outf) / opt.name / "weight"
    os.makedirs(str(weight_dir), exist_ok=True)
    csv_writer = open(weight_dir / "performance.csv", "w+")
    csv_writer.write("precision, recall, f1_score, global_step, class_name\n")
    num_epochs = opt.niter
    print('Training...')
    for epoch_idx in range(num_epochs):
        # ONE EPOCH START
        for batch_idx, ((batch_x, batch_y, bboxs), meta) in enumerate(train_loader):
            print(
                "Training Epoch: [%d/%d] Batch:[%d/%d]......" % (epoch_idx + 1, num_epochs, batch_idx + 1, num_batches),
                end='\b\r', flush=True)
            if opt.ffnegalg == "overlay":
                x_pos = overlay_y_on_x_batch(batch_x, batch_y, num_classes)
                rnd = torch.randperm(batch_x.size(0))
                x_neg = overlay_y_on_x_batch(batch_x, batch_y[rnd], num_classes)
            elif opt.ffnegalg == "hybrid":
                x_pos = batch_x
                x_neg = get_batch_of_hybrid_image(batch_x, batch_y, opt, classes)

            # write to tensorboard
            img_pos = fix_batch_of_image_for_rendering(x_pos, True)
            writer.add_image('Positive Images', torchvision.utils.make_grid(img_pos))
            img_neg = fix_batch_of_image_for_rendering(x_neg, True)
            writer.add_image('Negative Images', torchvision.utils.make_grid(img_neg))

            # x_pos = x_pos.view(-1, 3 * 128 * 128)  # flatten(but batch wise)
            # x_neg = x_neg.view(-1, 3 * 128 * 128)  # flatten(but batch wise)
            net.set_detection_input(batch_y, bboxs)
            net.train(x_pos, x_neg, batch_idx)

        # ONE EPOCH END

        # unlike backprop, in FF we evaluate per-batch (not per-epoch)
        # draw bbox
        # draw_bbox(writer, net, opt)
        # test
        detailed = True
        auc, map1, map2, auc_ls, auc_dict = evaluate_mAP(writer, classes, num_classes, epoch_idx, num_epochs, net,
                                                         test_loader, opt, iou_threshold=opt.iou,
                                                         detailed_output=detailed,
                                                         csv_writer=csv_writer)
        if auc > max_auc \
                or any([cls_auc > perclass_max_auc[cls_idx] for cls_idx, cls_auc in auc_dict.items()]) \
                or map1 > max_mAP:
            if auc > max_auc:
                max_auc = auc
                max_auc_batch = batch_idx + 1
            if map1 > max_mAP:
                max_mAP = map1
                max_mAP_batch = batch_idx + 1
            perclass_max_auc = auc_dict

            # save the model
            torch.save({'epoch': batch_idx, 'state_dict': net.state_dict()},
                       f'{str(weight_dir)}/net_%d_%f.pth' % (batch_idx, auc))
        writer.add_scalar('Testing AUC', auc, batch_idx)
        for cls_idx, auc in auc_dict.items():
            writer.add_scalar('Testing AUC[%s]' % classes[cls_idx], auc, batch_idx)
        print(f"Testing epoch:[{batch_idx + 1}/{len(train_loader)}] Micro-averaged "
              f"AUC:{auc:.2f} mAP:{map1:.2f} ",
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

    csv_writer.close()
    writer.close()
