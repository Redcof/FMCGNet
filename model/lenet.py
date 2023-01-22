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
from torch.optim import optimizer
from tqdm import tqdm
from torchvision import transforms

from customdataset.atz.dataloader import load_atz_data
from model.dcn import DeformableConv2d


class LeNet(nn.Module):
    def __init__(self, if0=32, channels=1, num_classes=2, num_anchors=0, deformable=False):
        """
        LeNet input image size in 32x32  1ch
        if - input feature
        f - feature, k-kernel, s-stride, p-padding
        ps-pooling stride, fci - fully connected in
        fco - fully connected out
        """
        super(LeNet, self).__init__()
        k1, f1, s1, p1, d1 = 5, 6, 1, 0, 2
        k2, f2, s2, p2, d2 = 5, 16, 1, 0, 2
        ps = 2
        fc1o = 120
        fc2o = 84
        self.f2 = f2
        self.num_classes = num_classes
        # As original LeNet designed for 32x32
        # images patches, we need to calculate
        # feature map size for other image dimensions.
        # Calculating the feature map size after
        # convolution and pooling before moving
        # into fc layers
        fx = 1 + (if0 - (k1 * d1 - 1) + (2 * p1)) // s1
        fx = fx // ps
        fx = 1 + (fx - (k1 * d2 - 1) + (2 * p2)) // s2
        self.fx = fx = fx // ps
        # calculation ends.
        conv2 = DeformableConv2d if deformable else nn.Conv2d

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=f1, kernel_size=k1, stride=s1, padding=p1,
                               dilation=d1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=ps)  # non-learning layer, we can reuse

        self.conv2 = conv2(in_channels=f1, out_channels=f2, kernel_size=k2, stride=s2, padding=p2,
                           dilation=d2)

        self.fc1 = nn.Linear(fx * fx * f2, fc1o)
        self.fc2 = nn.Linear(fc1o, fc2o)
        self.fc3 = nn.Linear(fc2o, num_classes)

        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.fc4 = nn.Linear(in_features=512 * 3 * 3, out_features=1024)
        # self.fc5 = nn.Linear(in_features=1024, out_features=num_anchors * 4)  # 4 for bounding box regression
        # self.fc6 = nn.Linear(in_features=1024, out_features=num_anchors * num_classes)  # for classification
        # self.num_anchors = num_anchors
        # self.num_classes = num_classes
        self.activation = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=num_classes)

    def forward(self, x):
        # Perform the forward pass of the LeNet architecture
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(-1, self.fx * self.fx * self.f2)  # flatten(but batch wise)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        # x = self.softmax(x)

        # # Additional layers for object detection
        # x = self.activation(self.conv3(x))
        # x = self.activation(self.conv4(x))
        # x = self.activation(self.conv5(x))
        # x = self.activation(self.conv6(x))
        # x = self.activation(self.conv7(x))
        # x = x.view(x.size(0), -1)
        # x = self.activation(self.fc4(x))
        #
        # # Bounding box regression
        # bboxes = self.fc5(x)
        # bboxes = bboxes.view(bboxes.size(0), self.num_anchors, 4)

        # Classification
        # logits = self.fc6(x)
        # logits = logits.view(logits.size(0), self.num_anchors, self.num_classes)
        # return bboxes, logits
        return x


def train_loop(opt, classes, writer, train_loader, test_loader, val_loader):
    # helper function to show an image
    # (used in the `plot_classes_preds` function below)
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

    def compute_multiclass_auc(epoch_idx, num_epochs, model, test_loader, meta_dict, detailed_output=False):
        print("\nTesting... epoch:[%d/%d]" % (epoch_idx + 1, num_epochs))
        model.eval()
        labels_all, probs_all = [], []
        with torch.no_grad():
            for (images, labels), meta in tqdm(test_loader, leave=False, total=len(test_loader)):
                logits = model(images)
                probs = torch.softmax(logits, dim=-1)
                probs_all.append(probs.numpy())
                labels_all.append(labels.numpy())
        labels_all = np.concatenate(labels_all)
        probs_all = np.concatenate(probs_all)
        n_classes = probs_all.shape[1]
        auc_list = []
        auc_dict = defaultdict(lambda: [])
        for class_idx in tqdm(range(n_classes), leave=False):
            labels_binary = (labels_all == class_idx).astype(int)
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
        model.train()
        if detailed_output:
            return mean_auc, auc_list, auc_dict
        else:
            return mean_auc

    def train_one_batch(model, images, labels):
        optimizer.zero_grad()
        # bboxes_pred, logits_pred = net(inputs)
        logits_pred = model(images)  # forward
        # calculate the loss for bounding box regression
        # bbox_loss_value = criterion_bbox_loss(bboxes_pred, bboxes)
        # calculate the loss for classification
        class_loss_value = criterion_class_loss(logits_pred, labels)
        # combine the loss
        # loss = bbox_loss_value + class_loss_value
        loss = class_loss_value
        # backprop
        loss.backward()
        # update weights
        optimizer.step()
        return loss

    def train_one_epoch(epoch_id, num_epoch, model, meta_dict):
        print("\nTraining... epoch:[%d/%d]" % (epoch_idx + 1, num_epochs))
        net.train()  # activate training mode
        meta_dict["train_running_loss"] = 0.0
        for batch_idx, data in enumerate(tqdm(train_loader, leave=False, total=len(train_loader))):
            meta_dict["step_ctr"] += opt.batchsize
            (inputs, targets), meta = data
            loss = train_one_batch(model, inputs, targets)
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
                meta_dict["train_running_loss"] = 0.0

    # Define the number of training epochs
    num_epochs = opt.niter

    # initialize the model
    net = LeNet(if0=opt.isize, channels=opt.nc, num_classes=len(classes), deformable=opt.deformable)

    # Define the optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    net.to(opt.device)  # move to device

    # loss functions
    if opt.loss == 'SFL':
        criterion_class_loss = ops.sigmoid_focal_loss()
    elif opt.loss == 'BEC':
        class_balance_weights = 1.0 - train_loader.dataset.class_proportion
        criterion_class_loss = nn.CrossEntropyLoss(weight=class_balance_weights)
    else:
        # default cross entropy loss
        criterion_class_loss = nn.CrossEntropyLoss()
    # criterion_bbox_loss = nn.SmoothL1Loss()

    # Train the network
    for epoch_idx in range(num_epochs):
        meta_dict = dict(
            train_running_loss=0.0,
            test_running_loss=0.0,
            batch_ctr=0,
            step_ctr=0,
        )
        # 82466 14011

        # train
        train_one_epoch(epoch_idx, num_epochs, net, meta_dict)

        # test
        detailed = True
        auc, auc_ls, auc_dict = compute_multiclass_auc(epoch_idx, num_epochs, net, test_loader, meta_dict,
                                                       detailed_output=detailed)
        writer.add_scalar('Testing AUC', auc, epoch_idx)
        for cls_idx, auc in auc_dict.items():
            writer.add_scalar('Testing AUC[%s]' % classes[cls_idx], auc, epoch_idx)
        print(f"Testing epoch:[{epoch_idx + 1}d/{num_epochs}] Micro-averaged "
              f"AUC score:{auc:.2f}")
    print('Finished LeNet Training')
