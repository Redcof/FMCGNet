import os
import pathlib
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
import cv2
import numpy as np
import torch
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

from customdataset.atz.dataloader import CLASSE_IDX, load_atz_data
from model.evaluate import roc
from model.lenet_wth_gaussaff import LeNet
from options import Options
from torch.functional import F

# constant for classes
classes = CLASSE_IDX


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


def plot_classes_preds(net, images, labels):
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


def compute_multiclass_auc(epoch_idx, num_epochs, model, test_loader, meta_dict):
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
    for class_idx in tqdm(range(n_classes), leave=False):
        labels_binary = (labels_all == class_idx).astype(int)
        try:
            auc = roc_auc_score(labels_binary, probs_all[:, class_idx])
        except ValueError:
            # this happens when a batch contains negative or positive samples as ground_truth
            auc = 0.0
        auc_list.append(auc)
    auc = np.mean(auc_list)
    model.train()
    return auc


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
            writer.add_scalar('training loss', meta_dict["train_running_loss"] / opt.print_freq, meta_dict["step_ctr"])
            meta_dict["train_running_loss"] = 0.0


opt = Options().parse()

# default `log_dir` is "runs" - we'll be more specific here
dir_ = pathlib.Path(opt.outf) / opt.name
os.makedirs(str(dir_), exist_ok=True)
writer = SummaryWriter(str(dir_))

# Define the LeNet architecture
C = len(CLASSE_IDX)
net = LeNet(if0=opt.isize, channels=opt.nc, num_classes=C)

# Define the Gaussian Affinity Loss function
criterion_class_loss = nn.CrossEntropyLoss()
criterion_bbox_loss = nn.SmoothL1Loss()

# Define the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Define the number of training epochs
num_epochs = opt.niter
batch_size = opt.batchsize

# Define the data loading and preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_loader, test_loader, val_loader = load_atz_data(opt)

net.to(opt.device)  # move to device

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
    auc = compute_multiclass_auc(epoch_idx, num_epochs, net, test_loader, meta_dict)
    writer.add_scalar('Testing AUC', auc, epoch_idx)
    print(f"Testing epoch:[{epoch_idx + 1}d/{num_epochs}] Micro-averaged "
          f"One-vs-Rest ROC AUC score:{auc:.2f}")

print('Finished Training')
