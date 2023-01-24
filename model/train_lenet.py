import os
import pathlib
import shutil

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
import cv2
import numpy as np
import torch
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt
from torch import nn, ops
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from tqdm import tqdm

from customdataset.atz.dataloader import CLASSE_IDX, load_atz_data
from model.evaluate import roc

from options import Options
from torch.functional import F

# constant for classes
classes = CLASSE_IDX

if __name__ == '__main__':

    opt = Options().parse()
    opt.device = "cuda:0" if opt.device != 'cpu' else "cpu"

    torch.manual_seed(opt.manualseed)

    # default `log_dir` is "runs" - we'll be more specific here
    dir_ = pathlib.Path(opt.outf) / opt.name
    os.makedirs(str(dir_), exist_ok=True)  # recreate
    writer = SummaryWriter(str(dir_))

    # Define the data loading and preprocessing
    transform = transforms.Compose([
        transforms.Resize((opt.isize, opt.isize)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_loader, test_loader, val_loader = load_atz_data(opt, transform)

    if opt.ff:
        from model.lenetff import train_loop

        train_loop(opt, CLASSE_IDX, writer, train_loader, test_loader, val_loader)
    else:
        from model.lenet import train_loop

        train_loop(opt, CLASSE_IDX, writer, train_loader, test_loader, val_loader)
