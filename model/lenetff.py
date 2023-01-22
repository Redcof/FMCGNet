import cv2
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from tqdm import tqdm

from model.dcn import DeformableConv2d


def overlay_y_on_x_batch(x_batch, y_batch, n_classes):
    x_batch = x_batch.clone()
    for idx, (x, y) in enumerate(zip(x_batch, y_batch)):
        x = overlay_y_on_x(x, y, n_classes, inplace=True)
        x_batch[idx, :, :, :] = x
    return x_batch


def overlay_y_on_x(x, y, n_classes, inplace=False):
    if inplace is False:
        x = x.clone()
    x[:, :n_classes] *= 0.0
    try:
        x[range(x.shape[0]), y] = x.max()
    except Exception as e:
        print("=>>", x.shape, range(x.shape[0]), y, torch.tensor(0, dtype=torch.uint8), x.max())
        raise e
    return x


class LeNetFF(nn.Module):
    LAYER_NO = 0

    def __init__(self, epoch, num_class, dims=(784, 500, 500)):
        super().__init__()
        self.num_class = num_class
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [
                FCFFOrigLayer(dims[d], dims[d + 1], epoch=epoch)
            ]

    def predict(self, batch_x, n_classes):
        goodness_per_label = []
        for label in range(n_classes):
            y_label = [label] * len(batch_x)
            batch_h = overlay_y_on_x_batch(batch_x, y_label, self.num_class)
            batch_h = batch_h.view(-1, 3 * 128 * 128)  # flatten(but batch wise)
            goodness = []
            for layer in self.layers:
                batch_h = layer(batch_h)
                goodness += [batch_h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1), goodness_per_label

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for idx, layer in enumerate(self.layers):
            print('Layer:[%02d/%02d]...' % (idx, LeNetFF.LAYER_NO), end="\b\r")
            h_pos, h_neg = layer.train(h_pos, h_neg)


class FFLayer(nn.Module):
    def __init__(self, goodness_threshold=2.0, epoch=10, activation=None, criterion=None):
        super().__init__()
        LeNetFF.LAYER_NO += 1
        self.layer_no = LeNetFF.LAYER_NO
        self.threshold = goodness_threshold
        self.num_epochs = epoch
        self.activation = torch.nn.ReLU() if activation is None else activation
        self.criterion = self.calculate_loss if criterion is None else criterion

    def calculate_loss(self, f_pos, f_neg):
        goodness_pos = f_pos.pow(2).mean(1)
        goodness_neg = f_neg.pow(2).mean(1)
        # The following loss pushes pos (neg) samples to
        # values larger (smaller) than the self.threshold.
        loss = torch.log(1 + torch.exp(torch.cat([
            -goodness_pos + self.threshold,
            goodness_neg - self.threshold]))).mean()
        return loss

    def train(self, x_pos, x_neg):
        for epoch_idx in range(self.num_epochs):
            f_pos = self.forward(x_pos)
            f_neg = self.forward(x_neg)
            # calculate loss
            loss = self.criterion(f_pos, f_neg)
            self.optimizer.zero_grad()  # set gradients to 0
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()  # backprop loss in layer
            self.optimizer.step()  # update weights
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class FCFFOrigLayer(FFLayer):
    def __init__(self, in_features, out_features, dtype=None,
                 activation=None, goodness_threshold=2.0, epoch=10, optimizer=None, lr=0.03, criterion=None):
        super().__init__(goodness_threshold=goodness_threshold, epoch=epoch, activation=activation, criterion=criterion)
        self.fc = nn.Linear(in_features, out_features, dtype=dtype)
        self.optimizer = optim.Adam(self.fc.parameters(), lr=lr) if optimizer is None else optimizer

    def forward(self, x):
        x = x / (x.norm(2, 1, keepdim=True) + 1e-4) if self.layer_norm else x
        x = self.activation(self.fc(x))
        return x


class FCFFLayer(FFLayer):
    def __init__(self, in_features, out_features, dtype=None,
                 activation=None, goodness_threshold=2.0, epoch=10, optimizer=None, lr=0.03,
                 criterion=None):
        super().__init__(goodness_threshold=goodness_threshold,
                         epoch=epoch, activation=activation, criterion=criterion)
        self.fc = nn.Linear(in_features, out_features, dtype=dtype)
        self.optimizer = optim.Adam(self.fc.parameters(), lr=lr) if optimizer is None else optimizer

    def forward(self, x):
        x = self.activation(self.self.fc(x))
        return x


class ConvLayer(FFLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, deformable=False,
                 activation=None, goodness_threshold=2.0, epoch=10, optimizer=None, lr=0.03, criterion=None):
        super().__init__(goodness_threshold=goodness_threshold, epoch=epoch, activation=activation, criterion=criterion)
        # selection of convolution
        conv = DeformableConv2d if deformable else nn.Conv2d
        self.conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation)
        self.optimizer = optim.Adam(self.conv.parameters(), lr=lr) if optimizer is None else optimizer

    def forward(self, x):
        x = self.activation(self.self.conv(x))
        return x


def train_loop(opt, classes, writer, train_loader, test_loader, val_loader):
    num_batches = len(train_loader)
    net = LeNetFF(opt.niter, len(classes), dims=(3 * 128 * 128, 500, 50))
    net.to(opt.device)
    for idx, ((batch_x, batch_y), meta) in enumerate(train_loader):
        print("Batch: [%d/%d]" % (idx, num_batches))
        try:
            x_pos = overlay_y_on_x_batch(batch_x, batch_y, len(classes))
            rnd = torch.randperm(batch_x.size(0))
            x_neg = overlay_y_on_x_batch(batch_x, batch_y[rnd], len(classes))
            x_pos = x_pos.view(-1, 3 * 128 * 128)  # flatten(but batch wise)
            x_neg = x_neg.view(-1, 3 * 128 * 128)  # flatten(but batch wise)
            net.train(x_pos, x_neg)
        except Exception as e:
            print(e)

    with torch.no_grad():
        print("Testing..")
        num_batches = len(test_loader)
        for idx, ((batch_x, batch_y), meta) in enumerate(test_loader):
            print("Batch: [%d/%d]" % (idx + 1, num_batches))
            try:
                goodness = net.predict(batch_x, len(classes))
                print(goodness)
            except Exception as e:
                print(e)


def train_loop2(opt, classes, writer, train_loader, test_loader, val_loader):
    from forward_forward import train_with_forward_forward_algorithm
    import os
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trained_model = train_with_forward_forward_algorithm(
        model_type="progressive",
        n_layers=3,
        hidden_size=2000,
        lr=0.03,
        device=device,
        epochs=100,
        batch_size=opt.batchsize,
        theta=2.,
    )
