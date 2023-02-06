import os
import pathlib
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch import optim, ops
from torch.functional import F
from tqdm import tqdm

from model.FocalLoss import FocalLoss
from model.dcn import DeformableConv2d

# ######################################################################################
# helper function to show an image
# (used in the `plot_classes_preds` function below)
from model.lenet import LeNet


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = cv2.normalize(img.numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # img = img / 2 + 0.5  # unnormalize
    # npimg = img.numpy()
    if one_channel:
        plt.imshow(img, cmap="Greys")
    else:
        plt.imshow(np.transpose(img, (1, 2, 0)))


def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels, classes):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=False)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


import numpy as np
from sklearn.metrics import average_precision_score


def calculate_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x1_min, y1_min, x1_max, y1_max = x1, y1, x1 + w1, y1 + h1
    x2_min, y2_min, x2_max, y2_max = x2, y2, x2 + w2, y2 + h2

    intersect_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    intersect_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersect_area = intersect_w * intersect_h

    union_area = w1 * h1 + w2 * h2 - intersect_area

    iou = intersect_area / union_area
    return iou


def calc_mAP(gt_bboxes, pred_bboxes, pred_scores, iou_threshold=0.5):
    n_images = gt_bboxes.shape[0]
    avg_precision_scores = []
    for i in range(n_images):
        gt_bbox = gt_bboxes[i]
        pred_bbox = pred_bboxes[i]
        pred_score = pred_scores[i]
        # Calculate IoU scores
        iou_scores = []
        for j in range(gt_bbox.shape[0]):
            for k in range(pred_bbox.shape[0]):
                iou = calculate_iou(gt_bbox[j], pred_bbox[k])
                iou_scores.append((iou, j, k))
        # Sort the IoU scores
        iou_scores = sorted(iou_scores, key=lambda x: x[0], reverse=True)
        # Assign the highest IoU score to each ground truth bbox
        assigned_pred_bboxes = np.zeros((gt_bbox.shape[0],), dtype=bool)
        assigned_pred_scores = np.zeros((gt_bbox.shape[0],), dtype=float)
        for iou, gt_idx, pred_idx in iou_scores:
            if iou < iou_threshold:
                break
            if not assigned_pred_bboxes[gt_idx]:
                assigned_pred_bboxes[gt_idx] = True
                assigned_pred_scores[gt_idx] = pred_score[0][pred_idx]
        # Calculate the average precision score
        avg_precision_scores.append(average_precision_score(assigned_pred_bboxes, assigned_pred_scores))
    return np.mean(avg_precision_scores)


def evaluate(epoch_idx, num_epochs, model, test_loader, meta_dict, detailed_output=False, iou_threshold=0.5):
    print("\nTesting... epoch:[%d/%d]" % (epoch_idx + 1, num_epochs))
    model.eval()
    labels_all, probs_all, gt_bbox_all, prob_bbox_all = [], [], [], []
    with torch.no_grad():
        for (images, labels, bboxs), meta in tqdm(test_loader, leave=False, total=len(test_loader)):
            bboxs_pred, logits_pred = model(images)
            probs = torch.softmax(logits_pred, dim=-1)
            probs_all.append(probs.numpy())
            labels_all.append(labels.numpy())
            gt_bbox_all.append(bboxs.numpy())
            prob_bbox_all.append(bboxs_pred.numpy())
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

    model.train()
    if detailed_output:
        return mean_auc, mAP, auc_list, auc_dict
    else:
        return mean_auc, mAP


def train_one_batch(num_classes, model, images, labels, bboxes, optimizer, criterion_class_loss, criterion_bbox_loss):
    optimizer.zero_grad()
    bboxes_pred, logits_pred = model(images)
    # logits_pred = model(images)  # forward
    # calculate the loss for bounding box regression
    bbox_loss_value = criterion_bbox_loss(bboxes_pred, bboxes)
    # calculate the loss for classification
    class_loss_value = criterion_class_loss(logits_pred.view(-1, num_classes), labels.view(-1))
    # combine the loss
    loss = bbox_loss_value + class_loss_value
    # loss = class_loss_value
    # backprop
    loss.backward()
    # update weights
    optimizer.step()
    return loss


def train_one_epoch(num_classes, opt, writer, epoch_idx, num_epoch, model, meta_dict, train_loader, optimizer, criterion_class_loss,
                    criterion_bbox_loss):
    print("\nTraining... epoch:[%d/%d]" % (epoch_idx + 1, num_epoch))
    model.train()  # activate training mode
    meta_dict["train_running_loss"] = 0.0
    for batch_idx, data in enumerate(tqdm(train_loader, leave=False, total=len(train_loader))):
        meta_dict["step_ctr"] += opt.batchsize
        (inputs, targets, bboxs), meta = data
        loss = train_one_batch(num_classes, model, inputs, targets, bboxs, optimizer, criterion_class_loss, criterion_bbox_loss)
        # create grid of images
        img_grid = torchvision.utils.make_grid(inputs)
        # write to tensorboard
        writer.add_image('Batch Images', img_grid)
        meta_dict["train_running_loss"] += loss.item()
        meta_dict["batch_ctr"] += 1
        if meta_dict["batch_ctr"] % opt.print_freq == 0:
            # log the running loss
            writer.add_scalar('training loss', meta_dict["train_running_loss"] / opt.print_freq,
                              meta_dict["step_ctr"])
            meta_dict["total_train_loss"] = (meta_dict["train_running_loss"] / opt.print_freq)
            meta_dict["train_running_loss"] = 0.0
    print("Training Loss:", meta_dict["total_train_loss"], "Epoch:", epoch_idx + 1)


def train_loop(opt, classes, writer, train_loader, test_loader, val_loader):
    # Define the number of training epochs
    num_epochs = opt.niter

    # initialize the model
    sample_batch = torch.rand((opt.batchsize, opt.nc, opt.isize, opt.isize))
    net = LeNet(sample_batch, channels=opt.nc, num_classes=len(classes),
                num_anchors=1,
                deformable=opt.deformable, dilation=opt.dilation)
    print("================Model Summary===============")
    op_dir = pathlib.Path(opt.outf) / opt.name
    os.makedirs(str(op_dir), exist_ok=True)
    with open(str(op_dir / "model_summary.txt"), "w") as fp:
        s = "\n%s\n" % net
        fp.write(opt.name)
        fp.write(s)
        print(s)
    print("============================================")
    # Define the optimizer
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    net.to(opt.device)  # move to device

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
    criterion_bbox_loss = nn.SmoothL1Loss()

    # Train the network
    max_auc = 0.0
    max_auc_epoch = 0
    max_mAP = 0.0
    max_mAP_epoch = 0
    perclass_max_auc = defaultdict(lambda: 0.0)
    weight_dir = pathlib.Path(opt.outf) / opt.name / "weight"
    os.makedirs(str(weight_dir), exist_ok=True)
    for epoch_idx in range(num_epochs):
        meta_dict = dict(
            total_train_loss=0.0,
            train_running_loss=0.0,
            test_running_loss=0.0,
            batch_ctr=0,
            step_ctr=0,
        )

        # train
        train_one_epoch(len(classes), opt, writer, epoch_idx, num_epochs, net, meta_dict, train_loader, optimizer,
                        criterion_class_loss, criterion_bbox_loss)

        # test
        detailed = True
        auc, mAP, auc_ls, auc_dict = evaluate(epoch_idx, num_epochs, net, test_loader, meta_dict,
                                              detailed_output=detailed, iou_threshold=0.2)
        if auc > max_auc \
                or any([cls_auc > perclass_max_auc[cls_idx] for cls_idx, cls_auc in auc_dict.items()]) \
                or mAP > max_mAP:
            if auc > max_auc:
                max_auc = auc
                max_auc_epoch = epoch_idx + 1
            if mAP > max_mAP:
                max_mAP = mAP
                max_mAP_epoch = epoch_idx + 1
            perclass_max_auc = auc_dict

            # save the model
            torch.save({'epoch': epoch_idx, 'state_dict': net.state_dict()},
                       f'{str(weight_dir)}/net_%d_%f.pth' % (epoch_idx, auc))
        writer.add_scalar('Testing AUC', auc, epoch_idx)
        writer.add_scalar('Testing mAP', mAP, epoch_idx)
        for cls_idx, auc in auc_dict.items():
            writer.add_scalar('Testing AUC[%s]' % classes[cls_idx], auc, epoch_idx)
        print(f"Testing epoch:[{epoch_idx + 1}d/{num_epochs}] Micro-averaged "
              f"AUC:{auc:.2f} mAP:{mAP:.2f}",
              end="Class wise AUC [")
        for class_idx, cls_auc in auc_dict.items():
            print("%s:%0.2f, " % (classes[class_idx], cls_auc), end="")
        print(
            f"]. MAX AUC:{max_auc:.2f}@epoch-{max_auc_epoch} MAX mAP{max_mAP:.2f}@epoch-{max_mAP_epoch}")
    print('Finished LeNet Training')
